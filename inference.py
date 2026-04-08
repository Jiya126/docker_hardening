#!/usr/bin/env python3
"""
Inference script for the Docker Hardening environment.

Runs an LLM agent against all three tasks (patch_easy, patch_medium,
patch_hard) and outputs results in the required [START]/[STEP]/[END] format.

Required environment variables:
    HF_TOKEN       — API key (mandatory)
    API_BASE_URL   — LLM endpoint  (default: router.huggingface.co/v1)
    MODEL_NAME     — model id      (default: Qwen/Qwen2.5-72B-Instruct)
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import DockerHardeningEnv
from models import DockerHardeningAction, DockerHardeningObservation

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME   = os.getenv("IMAGE_NAME")

ENV_NAME    = "docker_hardening"
TASKS       = ["patch_easy", "patch_medium", "patch_hard"]
MAX_STEPS   = 5
MAX_TOKENS  = 2048

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Docker security expert. You will receive a Dockerfile and a
    security scan report. Your job is to patch the Dockerfile to fix the
    reported vulnerabilities and improve its overall security posture.

    Rules:
    - Output ONLY the complete patched Dockerfile content, nothing else.
    - Do NOT wrap it in markdown code fences.
    - Do NOT explain your changes.
    - Do NOT remove packages that the application might need.
    - Do NOT install new language runtimes (Java, Node, etc.).
    - Do NOT pin exact apt package versions (e.g. curl=7.88.1-10).

    Security best practices:
    - Upgrade the base image to a modern version (e.g. python:3.12-slim).
    - Run apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*.
    - Add a non-root USER instruction AFTER all RUN/COPY/ADD steps.
    - Add a real HEALTHCHECK (not CMD true), e.g. HEALTHCHECK CMD python -c "import sys"
    - Remove secrets from ENV/ARG instructions entirely.
    - Use COPY instead of ADD.
    - Use pip --no-cache-dir on all pip install lines.
    - Remove or do not EXPOSE database/cache ports (5432, 6379, 3306).
    - Remove pipe-to-shell patterns (curl|sh, wget|bash).
    - Remove ADD from URL patterns.

    Read the scan report carefully and address EVERY finding.
    The report contains specific fix instructions — follow them.\
""")


class _InlineResult:
    def __init__(self, observation: DockerHardeningObservation, reward: float, done: bool):
        self.observation = observation
        self.reward = reward
        self.done = done


class _InlineEnv:

    def __init__(self):
        from server.docker_hardening_environment import DockerHardeningEnvironment
        self._env = DockerHardeningEnvironment()

    async def reset(self) -> _InlineResult:
        obs = self._env.reset()
        return _InlineResult(observation=obs, reward=0.0, done=obs.done)

    async def step(self, action: DockerHardeningAction) -> _InlineResult:
        obs = self._env.step(action)
        return _InlineResult(observation=obs, reward=obs.reward or 0.0, done=obs.done)

    async def close(self) -> None:
        pass


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_short = action.replace("\n", "\\n")[:120]
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _build_user_prompt(
    dockerfile: str,
    vuln_summary: str,
    step: int,
    history: List[dict],
) -> str:

    parts = [f"Step {step}/{MAX_STEPS}.\n"]

    parts.append(f"Current Dockerfile:\n```\n{dockerfile}\n```\n")
    parts.append(f"Vulnerability Report:\n{vuln_summary}\n")

    if history:
        prev_reward = history[-1].get("reward", 0)

        if prev_reward <= 0 and len(history) >= 2:
            parts.append(
                "IMPORTANT: Your previous patches did NOT reduce vulnerabilities. "
                "Check the scan report for specific fix instructions and ensure "
                "you are addressing all remaining findings.\n"
            )
        elif prev_reward <= 0:
            parts.append(
                "NOTE: Your last patch did not improve the score. "
                "Look at the remaining vulnerabilities and anti-patterns carefully.\n"
            )

        parts.append("Previous steps:")
        for h in history[-3:]:
            parts.append(
                f"  Step {h['step']}: {h['vulns_remaining']} vulns remaining, "
                f"reward={h['reward']:.2f}"
            )
            if h.get("error"):
                parts.append(f"    Error: {h['error']}")
            if h.get("still_failing"):
                parts.append(f"    Still failing: {', '.join(h['still_failing'][:5])}")
        parts.append("")

    parts.append("Output the complete patched Dockerfile now.")

    return "\n".join(parts)


def _extract_failing_checks(vuln_summary: str) -> List[str]:
    failing = []
    for line in vuln_summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("[FAIL]"):
            failing.append(stripped[7:].strip())
        if "(Python dependency" in stripped:
            pkg = stripped.split("|")[1].strip().split()[0] if "|" in stripped else ""
            if pkg:
                failing.append(f"pip: {pkg}")
    return failing


def get_patched_dockerfile(
    client: OpenAI,
    dockerfile: str,
    vuln_summary: str,
    step: int,
    history: List[dict],
) -> str:
    user_prompt = _build_user_prompt(dockerfile, vuln_summary, step, history)

    temperature = 0.3 if step == 1 else min(0.3 + 0.15 * (step - 1), 0.7)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else dockerfile
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return dockerfile


async def run_task(task_name: str, client: OpenAI) -> float:
    os.environ["SCA_GYM_TASK"] = task_name
    os.environ.setdefault("SCA_GYM_MODE", "eval")

    if IMAGE_NAME:
        env = await DockerHardeningEnv.from_docker_image(IMAGE_NAME)
    elif os.getenv("ENV_BASE_URL"):
        env = DockerHardeningEnv(base_url=os.environ["ENV_BASE_URL"])
    else:
        env = _InlineEnv()

    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            patched = get_patched_dockerfile(
                client, obs.current_dockerfile, obs.vulnerability_summary,
                step, history,
            )

            action = DockerHardeningAction(patched_dockerfile=patched)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step
            score = obs.score

            log_step(step=step, action=f"patch({obs.current_vuln_count} vulns remaining)",
                     reward=reward, done=done, error=error)

            still_failing = _extract_failing_checks(obs.vulnerability_summary)
            history.append({
                "step": step,
                "vulns_remaining": obs.current_vuln_count,
                "reward": reward,
                "error": error,
                "still_failing": still_failing,
            })

            if done:
                break

        success = score > 0.5

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = {}
    for task in TASKS:
        scores[task] = await run_task(task, client)

    print("\n--- Summary ---", flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  Average: {avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
