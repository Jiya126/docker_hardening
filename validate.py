#!/usr/bin/env python3
"""
Pre-submission validation script.

Connects to the running environment server and verifies:
  1. Health endpoint works
  2. Spec endpoint returns valid metadata
  3. Reset returns proper observation
  4. 2-step episode flow works (analyze → patch)
  5. Scores have variance (different actions → different rewards)
  6. Randomization works (different seeds → different Dockerfiles)
"""

import json
import sys
import time

import requests

BASE = "http://localhost:7860"
TASKS = ["patch_easy", "patch_medium", "patch_hard"]


def _ok(msg: str):
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")
    return False


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_health() -> bool:
    _section("1. Health Check")
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        if r.status_code != 200:
            return _fail(f"GET /health returned {r.status_code}")
        data = r.json()
        if data.get("status") != "healthy":
            return _fail(f"Health status: {data.get('status')}")
        _ok(f"Healthy, {data.get('num_tasks', 0)} tasks loaded")
        return True
    except Exception as e:
        return _fail(f"Health check failed: {e}")


def check_spec() -> bool:
    _section("2. Spec Endpoint")
    try:
        r = requests.get(f"{BASE}/spec", timeout=10)
        if r.status_code != 200:
            return _fail(f"GET /spec returned {r.status_code}")
        spec = r.json()
        required = ["env_name", "reward_range", "episode_steps", "action_schema", "tasks"]
        missing = [k for k in required if k not in spec]
        if missing:
            return _fail(f"Missing spec fields: {missing}")

        rr = spec["reward_range"]
        if rr[0] < 0 or rr[1] > 1:
            return _fail(f"Reward range {rr} not in [0, 1]")

        _ok(f"Spec valid: {spec['env_name']}, reward_range={rr}, steps={spec['episode_steps']}")
        return True
    except Exception as e:
        return _fail(f"Spec check failed: {e}")


def check_reset_step() -> bool:
    _section("3. Reset + 2-Step Episode Flow")
    all_ok = True

    for task in TASKS:
        try:
            r = requests.post(f"{BASE}/reset", json={"task_name": task}, timeout=30)
            if r.status_code != 200:
                _fail(f"Reset {task}: status {r.status_code}")
                all_ok = False
                continue
            obs = r.json()

            if not obs.get("current_dockerfile"):
                _fail(f"Reset {task}: empty Dockerfile")
                all_ok = False
                continue
            if obs.get("done", True):
                _fail(f"Reset {task}: done=True on reset")
                all_ok = False
                continue
            _ok(f"Reset {task}: {obs.get('initial_vuln_count', 0)} vulns, step={obs.get('step_number', 0)}")

            # Step 1: Analyze
            analyze_action = {
                "patched_dockerfile": json.dumps({
                    "identified_issues": ["running as root", "secrets in ENV", "pipe-to-shell"],
                    "identified_categories": ["best_practice", "secret_exposure", "supply_chain"],
                })
            }
            r = requests.post(
                f"{BASE}/step",
                json={"action": analyze_action, "task_name": task},
                timeout=30,
            )
            if r.status_code != 200:
                _fail(f"Analyze {task}: status {r.status_code}")
                all_ok = False
                continue
            obs = r.json()
            if obs.get("done", True):
                _fail(f"Analyze {task}: done=True after analysis (should be False)")
                all_ok = False
                continue
            analyze_reward = obs.get("reward", 0)
            _ok(f"Analyze {task}: reward={analyze_reward:.3f}, done=False")

            # Step 2: Patch
            patch_action = {
                "patched_dockerfile": (
                    "FROM python:3.12-slim\n"
                    "RUN apt-get update && apt-get upgrade -y "
                    "&& apt-get install -y --no-install-recommends curl "
                    "&& rm -rf /var/lib/apt/lists/*\n"
                    "WORKDIR /app\n"
                    "COPY requirements.txt .\n"
                    "RUN pip install --no-cache-dir -r requirements.txt\n"
                    "COPY . .\n"
                    "EXPOSE 8080\n"
                    'HEALTHCHECK CMD python -c "import sys"\n'
                    "RUN useradd -r appuser\n"
                    "USER appuser\n"
                    'CMD ["python", "app.py"]\n'
                )
            }
            r = requests.post(
                f"{BASE}/step",
                json={"action": patch_action, "task_name": task},
                timeout=30,
            )
            if r.status_code != 200:
                _fail(f"Patch {task}: status {r.status_code}")
                all_ok = False
                continue
            obs = r.json()
            if not obs.get("done", False):
                _fail(f"Patch {task}: done=False after patch (should be True)")
                all_ok = False
                continue
            patch_score = obs.get("score", 0)
            _ok(f"Patch {task}: score={patch_score:.3f}, done=True")

        except Exception as e:
            _fail(f"{task} flow failed: {e}")
            all_ok = False

    return all_ok


def check_score_variance() -> bool:
    _section("4. Score Variance (empty vs. good action)")
    try:
        r = requests.post(f"{BASE}/reset", json={"task_name": "patch_easy"}, timeout=30)
        obs = r.json()

        # Empty analysis
        r = requests.post(
            f"{BASE}/step",
            json={"action": {"patched_dockerfile": "{}"}, "task_name": "patch_easy"},
            timeout=30,
        )
        r = requests.post(
            f"{BASE}/step",
            json={"action": {"patched_dockerfile": obs.get("current_dockerfile", "")}, "task_name": "patch_easy"},
            timeout=30,
        )
        empty_score = r.json().get("score", 0)

        # Good patch
        r = requests.post(f"{BASE}/reset", json={"task_name": "patch_easy"}, timeout=30)
        r = requests.post(
            f"{BASE}/step",
            json={"action": {"patched_dockerfile": json.dumps({
                "identified_issues": ["running as root", "no healthcheck", "pipe-to-shell", "no apt cache cleanup"],
                "identified_categories": ["best_practice", "supply_chain"],
            })}, "task_name": "patch_easy"},
            timeout=30,
        )
        r = requests.post(
            f"{BASE}/step",
            json={"action": {"patched_dockerfile": (
                "FROM python:3.12-slim\n"
                "RUN apt-get update && apt-get upgrade -y "
                "&& apt-get install -y --no-install-recommends curl "
                "&& rm -rf /var/lib/apt/lists/*\n"
                "WORKDIR /app\n"
                "COPY requirements.txt .\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY . .\n"
                "EXPOSE 8080\n"
                'HEALTHCHECK CMD python -c "import sys"\n'
                "RUN useradd -r appuser\n"
                "USER appuser\n"
                'CMD ["python", "app.py"]\n'
            )}, "task_name": "patch_easy"},
            timeout=30,
        )
        good_score = r.json().get("score", 0)

        diff = good_score - empty_score
        if diff > 0.05:
            _ok(f"Score variance OK: empty={empty_score:.3f}, good={good_score:.3f}, diff={diff:.3f}")
            return True
        else:
            return _fail(f"Insufficient variance: empty={empty_score:.3f}, good={good_score:.3f}")

    except Exception as e:
        return _fail(f"Variance check failed: {e}")


def check_randomization() -> bool:
    _section("5. Randomization (different Dockerfiles per reset)")
    try:
        dockerfiles = set()
        for i in range(5):
            r = requests.post(f"{BASE}/reset", json={"task_name": "patch_easy"}, timeout=30)
            obs = r.json()
            df = obs.get("current_dockerfile", "")
            dockerfiles.add(df)

        if len(dockerfiles) >= 2:
            _ok(f"Randomization works: {len(dockerfiles)} unique Dockerfiles in 5 resets")
            return True
        else:
            return _fail(f"Only {len(dockerfiles)} unique Dockerfile(s) in 5 resets — not random enough")

    except Exception as e:
        return _fail(f"Randomization check failed: {e}")


def check_state() -> bool:
    _section("6. State Endpoint")
    try:
        r = requests.post(f"{BASE}/reset", json={"task_name": "patch_easy"}, timeout=30)
        r = requests.get(f"{BASE}/state?task_name=patch_easy", timeout=10)
        if r.status_code != 200:
            return _fail(f"GET /state returned {r.status_code}")
        state = r.json()
        if "episode_id" not in state:
            return _fail("Missing episode_id in state")
        if "active_issues" not in state:
            return _fail("Missing active_issues in state")
        _ok(f"State OK: episode={state['episode_id'][:8]}..., "
            f"{len(state.get('active_issues', []))} active issues")
        return True
    except Exception as e:
        return _fail(f"State check failed: {e}")


def main():
    print("=" * 60)
    print("  SCA-Gym Pre-Submission Validator")
    print(f"  Target: {BASE}")
    print("=" * 60)

    for attempt in range(10):
        try:
            r = requests.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"  Waiting for server... ({attempt + 1}/10)")
        time.sleep(3)
    else:
        print("  [FATAL] Server not responding after 30s")
        sys.exit(1)

    results = {
        "health": check_health(),
        "spec": check_spec(),
        "episode_flow": check_reset_step(),
        "score_variance": check_score_variance(),
        "randomization": check_randomization(),
        "state": check_state(),
    }

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} — {name}")
    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  Ready for submission!")
    else:
        print("\n  Fix failing checks before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
