#!/usr/bin/env python3
"""
Inference script for the Docker Hardening environment.

Runs an LLM agent against all three tasks (patch_easy, patch_medium,
patch_hard) using the 2-step episode design:
  Step 1 — Analyze: identify security issues
  Step 2 — Patch:   submit a fixed Dockerfile

Outputs results in the required [START]/[STEP]/[END] format.

Required environment variables:
    HF_TOKEN       — API key (mandatory)
    API_BASE_URL   — LLM endpoint  (default: router.huggingface.co/v1)
    MODEL_NAME     — model id      (default: Qwen/Qwen2.5-72B-Instruct)
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests

API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

ENV_NAME = "docker_hardening"
TASKS = ["patch_easy", "patch_medium", "patch_hard"]
MAX_RETRIES = 3

# ── Prompts ──────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = textwrap.dedent("""\
    You are a Docker security expert analyzing a Dockerfile and its scan report.

    Your task is to IDENTIFY all security issues present. Do NOT fix anything yet.
    Produce a JSON object with exactly these fields:
    {
      "identified_issues": [
        "description of issue 1",
        "description of issue 2",
        ...
      ],
      "identified_categories": [
        "category1",
        "category2",
        ...
      ]
    }

    Valid categories (use exact strings):
    - "supply_chain" — pipe-to-shell (curl|bash), ADD from URL, get.docker.com
    - "best_practice" — missing USER, missing HEALTHCHECK, no cache cleanup, pip cache, layer merging, copy order, no-install-recommends
    - "network_security" — exposed database/cache ports (5432, 3306, 6379)
    - "secret_exposure" — secrets in ENV/ARG, base64 secrets, credentials in URLs
    - "vulnerability" — outdated base image, known CVEs, pip package vulnerabilities
    - "permissions" — chmod 777, overly permissive file permissions

    Be thorough. Check for ALL of these patterns:
    1. Secrets in ENV or ARG (passwords, tokens, API keys, base64 encoded values)
    2. Credentials embedded in URLs (user:pass@host)
    3. Pipe-to-shell patterns (curl|bash, wget|sh)
    4. ADD from URL instead of COPY
    5. Docker-in-Docker (get.docker.com)
    6. Missing non-root USER instruction
    7. Missing or useless HEALTHCHECK (CMD true is useless)
    8. No apt-get cache cleanup (rm -rf /var/lib/apt/lists/*)
    9. No --no-cache-dir on pip install
    10. No --no-install-recommends on apt-get install
    11. Exposed database/cache ports (5432, 3306, 6379)
    12. Old base image (python < 3.12)
    13. Consecutive RUN commands that should be merged
    14. COPY . . before COPY requirements.txt (bad cache order)
    15. chmod 777 (overly permissive)

    Output ONLY valid JSON. No explanation, no markdown fences.\
""")

PATCH_PROMPT = textwrap.dedent("""\
    You are a Docker security expert. You will receive:
    1. A vulnerable Dockerfile
    2. A security scan report WITH analysis feedback showing which issues
       were correctly identified and which were missed.

    Your job is to produce a FIXED Dockerfile that resolves ALL identified issues.

    Rules:
    - Output ONLY the complete patched Dockerfile content, nothing else.
    - Do NOT wrap it in markdown code fences.
    - Do NOT explain your changes.
    - Do NOT remove packages that the application might need.
    - Do NOT install new language runtimes (Java, Node, etc.).
    - Do NOT pin exact apt package versions (e.g. curl=7.88.1-10).

    Security best practices to apply:
    - Upgrade the base image to python:3.12-slim.
    - Run apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
    - Use --no-install-recommends on apt-get install.
    - Add a non-root USER instruction AFTER all RUN/COPY/ADD steps.
    - Add a real HEALTHCHECK (not CMD true), e.g. HEALTHCHECK CMD python -c "import sys"
    - Remove ALL secrets from ENV/ARG instructions entirely.
    - Remove credentials from URLs.
    - Remove base64-encoded secrets from ENV.
    - Use COPY instead of ADD.
    - Use pip --no-cache-dir on all pip install lines.
    - Remove or do not EXPOSE database/cache ports (5432, 6379, 3306).
    - Remove pipe-to-shell patterns (curl|sh, wget|bash).
    - Remove ADD from URL patterns.
    - Remove Docker-in-Docker installation.
    - Remove chmod 777.
    - Merge consecutive RUN commands.
    - Ensure COPY requirements.txt comes before COPY . .

    Pay special attention to the analysis feedback — fix every CONFIRMED issue
    and also address any MISSED issues mentioned in the feedback.\
""")


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _llm_call(system_prompt: str, user_content: str, temperature: float = 0.3) -> str:
    """Call the LLM via the OpenAI-compatible chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    url = f"{API_BASE_URL}/chat/completions"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            _debug(f"LLM call attempt {attempt + 1} failed: {exc}")
            if attempt == MAX_RETRIES - 1:
                return ""
            time.sleep(2 ** attempt)
    return ""


def _parse_llm_json(text: str) -> Dict:
    """Extract JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"identified_issues": [text], "identified_categories": []}


# ── Environment helpers ──────────────────────────────────────────────────────

def _env_health() -> bool:
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _env_reset(task_name: str) -> Dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _env_step(action: Dict, task_name: str) -> Dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "task_name": task_name},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Logging ──────────────────────────────────────────────────────────────────

def _debug(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def log_start(task: str, model: str):
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action_summary: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action_summary} "
        f"reward={reward:.4f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    score = max(0.01, min(0.99, score))
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(task_name: str) -> float:
    """Run a single 2-step episode: analyze then patch."""

    log_start(task=task_name, model=MODEL_NAME)
    rewards = []
    score = 0.01

    try:
        # Reset
        obs = _env_reset(task_name)
        dockerfile = obs.get("current_dockerfile", "")
        vuln_summary = obs.get("vulnerability_summary", "")

        # ── Step 1: Analyze ──────────────────────────────────────────────
        analyze_input = (
            f"Dockerfile:\n```\n{dockerfile}\n```\n\n"
            f"Security Scan Report:\n{vuln_summary}\n"
        )
        analyze_raw = _llm_call(ANALYZE_PROMPT, analyze_input, temperature=0.2)
        analyze_json = _parse_llm_json(analyze_raw)

        analyze_action = json.dumps(analyze_json)
        obs = _env_step({"patched_dockerfile": analyze_action}, task_name)

        analyze_reward = obs.get("reward", 0.01)
        rewards.append(analyze_reward)

        issues_str = ", ".join(analyze_json.get("identified_issues", [])[:3])
        log_step(
            step=1,
            action_summary=f"analyze({len(analyze_json.get('identified_issues', []))} issues: {issues_str[:80]})",
            reward=analyze_reward,
            done=obs.get("done", False),
            error=obs.get("last_action_error"),
        )

        # ── Step 2: Patch ────────────────────────────────────────────────
        vuln_summary_with_feedback = obs.get("vulnerability_summary", vuln_summary)
        patch_input = (
            f"Dockerfile:\n```\n{dockerfile}\n```\n\n"
            f"Security Report + Analysis Feedback:\n{vuln_summary_with_feedback}\n"
        )
        patched_dockerfile = _llm_call(PATCH_PROMPT, patch_input, temperature=0.3)

        if not patched_dockerfile or "FROM" not in patched_dockerfile.upper():
            patched_dockerfile = dockerfile

        obs = _env_step({"patched_dockerfile": patched_dockerfile}, task_name)

        patch_reward = obs.get("reward", 0.01)
        rewards.append(patch_reward)
        score = obs.get("score", 0.01)

        vulns_remaining = obs.get("current_vuln_count", "?")
        log_step(
            step=2,
            action_summary=f"patch({vulns_remaining} vulns remaining)",
            reward=patch_reward,
            done=obs.get("done", True),
            error=obs.get("last_action_error"),
        )

    except Exception as exc:
        _debug(f"Episode failed: {exc}")
        score = 0.01

    success = score > 0.5
    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        _debug("WARNING: No API key set (HF_TOKEN / OPENAI_API_KEY / API_KEY)")

    _debug(f"Model: {MODEL_NAME}")
    _debug(f"API: {API_BASE_URL}")
    _debug(f"Env: {ENV_BASE_URL}")

    for attempt in range(10):
        if _env_health():
            _debug("Environment is healthy")
            break
        _debug(f"Waiting for environment... (attempt {attempt + 1})")
        time.sleep(3)
    else:
        _debug("WARNING: Environment health check failed, proceeding anyway")

    scores = {}
    for task in TASKS:
        scores[task] = run_episode(task)

    _debug("--- Summary ---")
    for task, sc in scores.items():
        _debug(f"  {task}: {sc:.4f}")
    avg = sum(scores.values()) / len(scores)
    _debug(f"  Average: {avg:.4f}")


if __name__ == "__main__":
    main()
