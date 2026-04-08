#!/usr/bin/env python3
"""
Docker Hardening RL — Evaluation Script
========================================
Evaluates a trained (or base) model on the Docker hardening environment
and compares it against the rule-based mock patcher baseline.

Usage:
    # Evaluate base model (no training):
    python examples/eval_rl.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct

    # Evaluate a trained checkpoint:
    python examples/eval_rl.py --model output/docker-hardening-grpo/final

    # Compare against mock patcher baseline only:
    python examples/eval_rl.py --baseline-only
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Severity
from tools.scanner import scan_mock, check_best_practices, detect_antipatterns
from tools.patch_agent import LLMPatchAgent
from curriculum import CURRICULUM, get_sample_dockerfile, get_sample_image
from examples.train_rl import (
    build_policy_prompt,
    extract_dockerfile,
    compute_completion_reward,
)


@dataclass
class EvalResult:
    level: int
    episode: int
    reward: float
    vulns_before: int
    vulns_after: int
    build_success: bool
    source: str  # "model" or "baseline"
    elapsed_s: float = 0.0


@dataclass
class LevelSummary:
    level: int
    name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reward for r in self.results) / len(self.results)

    @property
    def avg_vuln_reduction(self) -> float:
        if not self.results:
            return 0.0
        reductions = [r.vulns_before - r.vulns_after for r in self.results]
        return sum(reductions) / len(reductions)

    @property
    def build_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.build_success) / len(self.results)


# ---------------------------------------------------------------------------
# Baseline: rule-based mock patcher
# ---------------------------------------------------------------------------

def run_baseline_episode(difficulty: int) -> EvalResult:
    """Run one episode using the mock patcher (rule-based baseline)."""
    dockerfile = get_sample_dockerfile(difficulty)
    image = get_sample_image(difficulty)
    report = scan_mock(image, difficulty=difficulty)

    agent = LLMPatchAgent(provider="claude", use_mock=True)
    attempt = agent.patch(dockerfile, report, [], cycle=1)

    if not attempt.success or not attempt.patched_dockerfile:
        return EvalResult(
            level=difficulty, episode=0, reward=-1.0,
            vulns_before=report.total_count, vulns_after=report.total_count,
            build_success=False, source="baseline",
        )

    reward = compute_completion_reward(
        dockerfile, attempt.patched_dockerfile, difficulty, image,
    )
    new_report = scan_mock(
        "baseline:eval", difficulty=difficulty,
        current_dockerfile=attempt.patched_dockerfile,
    )
    return EvalResult(
        level=difficulty, episode=0, reward=reward,
        vulns_before=report.total_count, vulns_after=new_report.total_count,
        build_success=True, source="baseline",
    )


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def run_model_episode(
    model,
    tokenizer,
    difficulty: int,
) -> EvalResult:
    """Run one episode using the trained policy model."""
    import torch

    dockerfile = get_sample_dockerfile(difficulty)
    image = get_sample_image(difficulty)
    report = scan_mock(image, difficulty=difficulty)

    prompt_messages = build_policy_prompt(dockerfile, report)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )

    t0 = time.time()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    elapsed = time.time() - t0

    patched = extract_dockerfile(completion)
    if patched is None:
        return EvalResult(
            level=difficulty, episode=0, reward=-1.0,
            vulns_before=report.total_count, vulns_after=report.total_count,
            build_success=False, source="model", elapsed_s=elapsed,
        )

    reward = compute_completion_reward(dockerfile, patched, difficulty, image)
    new_report = scan_mock(
        "model:eval", difficulty=difficulty, current_dockerfile=patched,
    )
    return EvalResult(
        level=difficulty, episode=0, reward=reward,
        vulns_before=report.total_count, vulns_after=new_report.total_count,
        build_success=True, source="model", elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    model_summaries: dict[int, LevelSummary],
    baseline_summaries: dict[int, LevelSummary],
):
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    header = (
        f"{'Level':<8} {'Source':<10} {'Avg Reward':>12} "
        f"{'Vuln Reduction':>16} {'Build Success':>14}"
    )
    print(header)
    print("-" * 70)

    for level in sorted(set(list(model_summaries.keys()) + list(baseline_summaries.keys()))):
        lvl_config = CURRICULUM.get(level)
        level_name = lvl_config.name if lvl_config else f"Level {level}"

        if level in baseline_summaries:
            bs = baseline_summaries[level]
            print(
                f"  {level:<6} {'baseline':<10} {bs.avg_reward:>12.2f} "
                f"{bs.avg_vuln_reduction:>16.1f} {bs.build_success_rate:>13.0%}"
            )

        if level in model_summaries:
            ms = model_summaries[level]
            print(
                f"  {level:<6} {'model':<10} {ms.avg_reward:>12.2f} "
                f"{ms.avg_vuln_reduction:>16.1f} {ms.build_success_rate:>13.0%}"
            )

        if level in model_summaries and level in baseline_summaries:
            delta = model_summaries[level].avg_reward - baseline_summaries[level].avg_reward
            label = "better" if delta > 0 else "worse"
            print(f"  {'':<6} {'delta':<10} {delta:>+12.2f} ({label})")

        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Docker Hardening RL — Evaluation",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per difficulty level")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run the rule-based baseline")
    args = parser.parse_args()

    # Baseline evaluation
    print("Running baseline (rule-based mock patcher)...")
    baseline_summaries: dict[int, LevelSummary] = {}
    for level, lvl_config in sorted(CURRICULUM.items()):
        summary = LevelSummary(level=level, name=lvl_config.name)
        for ep in range(args.episodes):
            result = run_baseline_episode(level)
            result.episode = ep + 1
            summary.results.append(result)
            print(
                f"  Level {level} ep {ep+1}: reward={result.reward:.2f} "
                f"vulns {result.vulns_before}->{result.vulns_after}"
            )
        baseline_summaries[level] = summary

    if args.baseline_only:
        print_report({}, baseline_summaries)
        return

    # Model evaluation
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install transformers torch"
        )

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    print(f"Running model evaluation ({args.episodes} episodes per level)...")
    model_summaries: dict[int, LevelSummary] = {}
    for level, lvl_config in sorted(CURRICULUM.items()):
        summary = LevelSummary(level=level, name=lvl_config.name)
        for ep in range(args.episodes):
            result = run_model_episode(model, tokenizer, level)
            result.episode = ep + 1
            summary.results.append(result)
            print(
                f"  Level {level} ep {ep+1}: reward={result.reward:.2f} "
                f"vulns {result.vulns_before}->{result.vulns_after} "
                f"({result.elapsed_s:.1f}s)"
            )
        model_summaries[level] = summary

    print_report(model_summaries, baseline_summaries)


if __name__ == "__main__":
    main()
