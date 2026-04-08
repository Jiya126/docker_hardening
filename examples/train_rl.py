#!/usr/bin/env python3
"""
Docker Hardening RL — GRPO Training Script
===========================================
Trains a code LLM (via LoRA) to generate security-hardened Dockerfiles using
Group Relative Policy Optimization (GRPO) via HuggingFace TRL.

The policy model learns to read a Dockerfile + vulnerability report
and produce a patched Dockerfile that reduces vulnerabilities.
The environment's mock scanner provides the training signal — no Docker
daemon or external APIs are needed.

Install training dependencies first:
    pip install trl transformers torch datasets accelerate peft

Usage:
    # Mock-mode training with LoRA on GPU (recommended starting point):
    python examples/train_rl.py

    # Specific base model:
    python examples/train_rl.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

    # More generations per prompt (better GRPO signal, more compute):
    python examples/train_rl.py --num-generations 8

    # Multi-GPU with accelerate:
    accelerate launch examples/train_rl.py
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Severity, VulnReport
from tools.scanner import scan_mock, check_best_practices, detect_antipatterns
from curriculum import CURRICULUM, get_sample_dockerfile, get_sample_image


# ---------------------------------------------------------------------------
# Prompt templates for the policy model
# ---------------------------------------------------------------------------

POLICY_SYSTEM_PROMPT = """\
You are a Docker security hardening expert.
Given a Dockerfile and its vulnerability scan report, produce a patched
Dockerfile that fixes as many vulnerabilities as possible.

Rules:
- Output ONLY the patched Dockerfile content — no explanations, no markdown fences.
- Prefer upgrading the base image to a newer tag (e.g. python:3.12-slim).
- Run `apt-get update && apt-get upgrade -y` to fix system package CVEs.
- Add a non-root USER after build steps.
- Add a real HEALTHCHECK (not CMD true).
- Remove secrets from ENV/ARG.
- Use COPY instead of ADD.
- Clean apt cache: rm -rf /var/lib/apt/lists/*.
- Use pip --no-cache-dir.
- Do NOT pin exact apt package versions — you don't know what's available.
- Do NOT modify requirements.txt or other repo files from the Dockerfile.
- Keep the Dockerfile syntactically valid.\
"""


def build_policy_prompt(dockerfile: str, report: VulnReport) -> list[dict]:
    """Build a chat-format prompt for the policy model."""
    vuln_lines = []
    for v in sorted(
        report.vulnerabilities,
        key=lambda x: list(Severity).index(x.severity),
    ):
        fix = f" -> fix: {v.fixed_version}" if v.fixed_version else " (no fix)"
        vuln_lines.append(
            f"  [{v.severity.value}] {v.cve_id} | "
            f"{v.package_name} {v.installed_version}{fix}"
        )
    vuln_text = "\n".join(vuln_lines[:30])

    bp = check_best_practices(dockerfile)
    ap = detect_antipatterns(dockerfile)

    bp_lines = [f"  [{'PASS' if v else 'FAIL'}] {k}" for k, v in bp.items()]
    ap_lines = [f"  [WARN] {w}" for w in ap]

    user_msg = (
        f"Current Dockerfile:\n```\n{dockerfile}\n```\n\n"
        f"Vulnerability Report ({report.total_count} total, "
        f"CRITICAL: {report.critical_count}, HIGH: {report.high_count}):\n"
        f"{vuln_text}\n\n"
        f"Best Practices:\n" + "\n".join(bp_lines) + "\n\n"
    )
    if ap_lines:
        user_msg += "Anti-Patterns Detected:\n" + "\n".join(ap_lines) + "\n\n"
    user_msg += "Produce a patched Dockerfile that addresses the highest-severity issues."

    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Dataset generation from curriculum
# ---------------------------------------------------------------------------

def generate_training_dataset(
    num_samples_per_level: int = 20,
) -> list[dict]:
    """Create training prompts from curriculum levels."""
    samples = []
    for level, lvl_config in sorted(CURRICULUM.items()):
        for idx in range(len(lvl_config.sample_dockerfiles)):
            dockerfile = get_sample_dockerfile(level, index=idx)
            image = get_sample_image(level, index=idx)
            report = scan_mock(image, difficulty=level)

            prompt = build_policy_prompt(dockerfile, report)
            for _ in range(num_samples_per_level):
                samples.append({
                    "prompt": prompt,
                    "difficulty": level,
                    "original_dockerfile": dockerfile,
                    "image_tag": image,
                })
    return samples


# ---------------------------------------------------------------------------
# Reward function — evaluates a model completion against the environment
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {
    Severity.CRITICAL:   4.0,
    Severity.HIGH:       3.0,
    Severity.MEDIUM:     2.0,
    Severity.LOW:        1.0,
    Severity.NEGLIGIBLE: 0.5,
    Severity.UNKNOWN:    1.0,
}

W_VULN = 0.50
W_BP = 0.30
W_AP = 0.20


def _weighted_vuln_score(vulns: list) -> float:
    return sum(_SEVERITY_WEIGHT.get(v.severity, 1.0) for v in vulns)


def compute_completion_reward(
    original_dockerfile: str,
    patched_dockerfile: str,
    difficulty: int,
    image_tag: str,
) -> float:
    """Evaluate a single patched Dockerfile against the environment.

    Returns a reward in roughly [-1.0, 1.0] range.
    Uses the same scoring logic as the environment server.
    """
    initial_report = scan_mock(image_tag, difficulty=difficulty)
    new_report = scan_mock(
        "grpo:eval", difficulty=difficulty, current_dockerfile=patched_dockerfile,
    )

    initial_bp = check_best_practices(original_dockerfile)
    new_bp = check_best_practices(patched_dockerfile)
    initial_ap = detect_antipatterns(original_dockerfile)
    new_ap = detect_antipatterns(patched_dockerfile)

    # Vulnerability improvement (severity-weighted)
    initial_weight = _weighted_vuln_score(initial_report.vulnerabilities)
    current_weight = _weighted_vuln_score(new_report.vulnerabilities)
    vuln_improvement = max(0.0, (initial_weight - current_weight) / max(initial_weight, 1.0))

    # Best practices improvement
    init_bp_sat = sum(1 for v in initial_bp.values() if v)
    curr_bp_sat = sum(1 for v in new_bp.values() if v)
    improvable_bp = len(new_bp) - init_bp_sat
    bp_improvement = (
        max(0.0, (curr_bp_sat - init_bp_sat) / max(improvable_bp, 1))
        if improvable_bp > 0 else 0.0
    )

    # Anti-pattern improvement
    ap_improvement = (
        max(0.0, (len(initial_ap) - len(new_ap)) / max(len(initial_ap), 1))
        if len(initial_ap) > 0 else 0.0
    )

    score = W_VULN * vuln_improvement + W_BP * bp_improvement + W_AP * ap_improvement
    return min(1.0, max(-1.0, score))


def extract_dockerfile(completion: str) -> str | None:
    """Extract a Dockerfile from raw model output."""
    text = completion.strip()

    if "```" in text:
        blocks = re.findall(
            r"```(?:dockerfile|docker|Dockerfile)?\s*\n(.*?)```",
            text, re.DOTALL,
        )
        if blocks:
            text = blocks[0].strip()

    if not text or "FROM" not in text.upper():
        return None
    return text


def _completion_to_text(completion) -> str:
    """Extract plain text from a TRL completion (string or message list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if completion and isinstance(completion[-1], dict):
            return completion[-1].get("content", "")
    return str(completion)


def make_reward_fn():
    """Factory: returns a reward function compatible with TRL's GRPOTrainer.

    The reward function scores each completion (patched Dockerfile) against
    the environment's mock scanner and best-practice checks.
    """

    def dockerfile_reward(
        completions,
        original_dockerfile: list[str] | None = None,
        difficulty: list[int] | None = None,
        image_tag: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            orig_df = original_dockerfile[i] if original_dockerfile else ""
            diff = difficulty[i] if difficulty else 1
            img = image_tag[i] if image_tag else "python:3.9-slim"

            patched = extract_dockerfile(text)
            if patched is None:
                rewards.append(-1.0)
                continue

            if patched.strip() == orig_df.strip():
                rewards.append(-0.5)
                continue

            try:
                r = compute_completion_reward(orig_df, patched, diff, img)
            except Exception:
                r = -1.0
            rewards.append(r)
        return rewards

    return dockerfile_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Docker Hardening RL — GRPO Training",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        default="output/docker-hardening-grpo",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--lora", action="store_true", default=True,
        help="Use LoRA adapters (default: True)",
    )
    parser.add_argument(
        "--no-lora", action="store_false", dest="lora",
        help="Full fine-tuning (requires much more VRAM)",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="G in GRPO — completions per prompt")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--samples-per-level", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as exc:
        sys.exit(
            f"Missing training dependency: {exc}\n"
            "Install with:\n"
            "  pip install 'openenv-docker_hardening[train]'\n"
            "  or: pip install trl transformers torch datasets accelerate peft"
        )

    print("=" * 60)
    print("Docker Hardening RL — GRPO Training")
    print("=" * 60)
    print("\nGenerating training dataset from curriculum levels...")
    raw = generate_training_dataset(num_samples_per_level=args.samples_per_level)
    dataset = Dataset.from_list(raw)
    print(f"  {len(dataset)} training samples across {len(CURRICULUM)} difficulty levels")

    reward_fn = make_reward_fn()

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        print(f"\n  LoRA: rank={args.lora_r}, alpha={args.lora_alpha}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    effective_batch = max(args.batch_size, args.num_generations)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=effective_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=50,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        report_to="none",
    )

    print(f"\n  Model: {args.model}")
    print(f"  Generations per prompt (G): {args.num_generations}")
    print(f"  Precision: {'bf16' if use_bf16 else ('fp16' if torch.cuda.is_available() else 'fp32')}")
    print(f"  Output dir: {args.output_dir}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("\nStarting GRPO training...\n")
    trainer.train()

    save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(save_path)
    if peft_config:
        print(f"\nLoRA adapters saved to {save_path}")
        print("To merge into a full model:")
        print(f"  from peft import AutoPeftModelForCausalLM")
        print(f'  model = AutoPeftModelForCausalLM.from_pretrained("{save_path}")')
        print(f'  model = model.merge_and_unload()')
    else:
        print(f"\nFull model saved to {save_path}")

    print("\nTraining complete!")
    print(f"Evaluate with: python examples/eval_rl.py --model {save_path}")


if __name__ == "__main__":
    main()
