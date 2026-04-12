"""
Dynamic Dockerfile generators — randomized per episode.

Each reset() picks a random subset of security issues from a pool,
then composes a broken Dockerfile containing those issues.
This prevents memorization and forces genuine security reasoning.
"""

import random
import re
from typing import Dict, Set, Tuple

# ── Issue pools per difficulty ────────────────────────────────────────────────

EASY_POOL = [
    "pipe_to_shell",
    "no_user",
    "no_healthcheck",
    "apt_no_cleanup",
    "pip_no_cache",
    "no_install_recommends",
    "consecutive_runs",
    "exposed_cache_port",
    "old_base_image",
]

MEDIUM_POOL = EASY_POOL + [
    "secret_env_password",
    "secret_env_token",
    "add_instead_of_copy",
    "healthcheck_cmd_true",
    "exposed_db_port",
]

HARD_POOL = MEDIUM_POOL + [
    "add_from_url",
    "get_docker_com",
    "cred_in_url",
    "base64_secret",
    "copy_order_bad",
    "chmod_777",
    "multiple_secrets",
]

ISSUE_POOLS = {
    "patch_easy": EASY_POOL,
    "patch_medium": MEDIUM_POOL,
    "patch_hard": HARD_POOL,
}

ISSUE_RANGES = {
    "patch_easy": (4, 6),
    "patch_medium": (7, 10),
    "patch_hard": (10, 14),
}

# ── Issue → category mapping (for analysis scoring) ──────────────────────────

ISSUE_CATEGORY_MAP = {
    "pipe_to_shell": "supply_chain",
    "no_user": "best_practice",
    "no_healthcheck": "best_practice",
    "apt_no_cleanup": "best_practice",
    "pip_no_cache": "best_practice",
    "no_install_recommends": "best_practice",
    "consecutive_runs": "best_practice",
    "exposed_cache_port": "network_security",
    "old_base_image": "vulnerability",
    "secret_env_password": "secret_exposure",
    "secret_env_token": "secret_exposure",
    "add_instead_of_copy": "best_practice",
    "healthcheck_cmd_true": "best_practice",
    "exposed_db_port": "network_security",
    "add_from_url": "supply_chain",
    "get_docker_com": "supply_chain",
    "cred_in_url": "secret_exposure",
    "base64_secret": "secret_exposure",
    "copy_order_bad": "best_practice",
    "chmod_777": "permissions",
    "multiple_secrets": "secret_exposure",
}

ALL_CATEGORIES = sorted(set(ISSUE_CATEGORY_MAP.values()))

# ── Randomizable values ──────────────────────────────────────────────────────

_FAKE_PASSWORDS = [
    "admin123", "s3cret_passw0rd", "supersecret", "P@ssw0rd!",
    "changeme", "db_pass_2024", "root_secret", "mypassword123",
]
_FAKE_TOKENS = [
    "sk-1234567890abcdef", "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "token_abc123xyz789", "api_key_s3cr3t_v4lu3",
    "Bearer_ey1234567890", "pat-xxxx-yyyy-zzzz",
]
_FAKE_URLS = [
    "https://install.example.com/setup.sh",
    "https://setup.example.io/init.sh",
    "https://get.example.org/bootstrap.sh",
    "https://raw.githubusercontent.com/example/setup/main/init.sh",
]
_FAKE_CRED_URLS = [
    "https://deploy:ghp_s3cretToken123@git.example.com/org/repo.git",
    "https://admin:P@ssw0rd@registry.example.com/images/app.git",
    "https://ci:token_abc123@gitlab.example.io/team/service.git",
]
_FAKE_BASE64_SECRETS = [
    "ZGVwbG95OnMzY3JldFBhc3N3b3JkMTIz",
    "YWRtaW46c3VwZXJzZWNyZXQxMjM0NTY=",
    "dG9rZW46YXBpX2tleV9zM2NyM3RfdjRsdTM=",
]


def select_issues(task: str, rng: random.Random) -> Set[str]:
    """Randomly select which issues are active for this episode."""
    pool = ISSUE_POOLS[task]
    lo, hi = ISSUE_RANGES[task]
    count = rng.randint(lo, min(hi, len(pool)))
    return set(rng.sample(pool, count))


def generate_dockerfile(
    task: str, active_issues: Set[str], rng: random.Random,
) -> Tuple[str, str, Dict]:
    """
    Compose a broken Dockerfile containing the specified issues.
    Returns (dockerfile_text, base_image_tag, metadata).
    """
    difficulty = {"patch_easy": 1, "patch_medium": 2, "patch_hard": 3}[task]

    base_image = _pick_base_image(active_issues, difficulty, rng)
    lines = [f"FROM {base_image}"]

    # ENV/ARG section — secrets
    lines.extend(_build_secret_lines(active_issues, rng))

    # APT section
    lines.extend(_build_apt_lines(active_issues, rng))

    # Supply-chain downloads
    lines.extend(_build_download_lines(active_issues, rng))

    # Workdir
    lines.append("WORKDIR /app")

    # COPY / pip section
    lines.extend(_build_copy_pip_lines(active_issues, rng))

    # Permissions
    if "chmod_777" in active_issues:
        lines.append("RUN chmod 777 /app/tmp")

    # EXPOSE
    lines.extend(_build_expose_lines(active_issues, rng))

    # HEALTHCHECK
    lines.extend(_build_healthcheck_lines(active_issues))

    # CMD
    lines.append('CMD ["python", "app.py"]')

    dockerfile = "\n".join(lines) + "\n"

    metadata = {
        "difficulty": difficulty,
        "num_issues": len(active_issues),
        "active_issues": sorted(active_issues),
        "categories": sorted({ISSUE_CATEGORY_MAP.get(i, "other") for i in active_issues}),
    }

    return dockerfile, base_image, metadata


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_base_image(active_issues: Set[str], difficulty: int, rng: random.Random) -> str:
    if difficulty >= 3:
        return rng.choice(["python:3.6-slim", "python:3.7-slim", "python:3.8-slim"])
    if "old_base_image" in active_issues:
        return rng.choice(["python:3.9-slim", "python:3.10-slim"])
    return rng.choice(["python:3.11-slim", "python:3.11-slim", "python:3.10-slim"])


def _build_secret_lines(active_issues: Set[str], rng: random.Random) -> list:
    lines = []
    if "secret_env_password" in active_issues:
        lines.append(f"ENV DATABASE_PASSWORD={rng.choice(_FAKE_PASSWORDS)}")
    if "secret_env_token" in active_issues:
        lines.append(f"ENV API_TOKEN={rng.choice(_FAKE_TOKENS)}")
    if "multiple_secrets" in active_issues:
        lines.append(f"ENV FLASK_SECRET_KEY={rng.choice(_FAKE_PASSWORDS)}")
        lines.append(f"ARG DEPLOY_TOKEN={rng.choice(_FAKE_TOKENS)}")
    if "base64_secret" in active_issues:
        lines.append(f"ENV AUTH_TOKEN={rng.choice(_FAKE_BASE64_SECRETS)}")
    if "cred_in_url" in active_issues:
        lines.append(f"ARG REPO_URL={rng.choice(_FAKE_CRED_URLS)}")
    return lines


def _build_apt_lines(active_issues: Set[str], rng: random.Random) -> list:
    lines = []
    pkgs = ["curl", "wget"]
    if rng.random() < 0.5:
        pkgs.append("git")

    recommends = "" if "no_install_recommends" in active_issues else " --no-install-recommends"
    apt_line = f"RUN apt-get update && apt-get install -y{recommends} {' '.join(pkgs)}"

    if "apt_no_cleanup" not in active_issues:
        apt_line += " && rm -rf /var/lib/apt/lists/*"

    lines.append(apt_line)

    if "consecutive_runs" in active_issues:
        lines.append(f"RUN {rng.choice(['echo deps installed', 'apt-get clean', 'ls /usr/bin'])}")

    return lines


def _build_download_lines(active_issues: Set[str], rng: random.Random) -> list:
    lines = []
    if "pipe_to_shell" in active_issues:
        url = rng.choice(_FAKE_URLS)
        lines.append(f"RUN curl -sSL {url} | bash")
    if "get_docker_com" in active_issues:
        lines.append("RUN curl -sSL https://get.docker.com | sh")
    if "add_from_url" in active_issues:
        lines.append("ADD https://example.com/install.sh /tmp/install.sh")
        lines.append("RUN chmod +x /tmp/install.sh && sh /tmp/install.sh")
    if "cred_in_url" in active_issues:
        lines.append("RUN git clone ${REPO_URL} /src")
    return lines


def _build_copy_pip_lines(active_issues: Set[str], rng: random.Random) -> list:
    lines = []

    if "copy_order_bad" in active_issues:
        lines.append("COPY . .")
        if "add_instead_of_copy" in active_issues:
            lines.append("ADD requirements.txt /app/requirements.txt")
        else:
            lines.append("COPY requirements.txt .")
    else:
        if "add_instead_of_copy" in active_issues:
            lines.append("ADD requirements.txt /app/requirements.txt")
        else:
            lines.append("COPY requirements.txt .")

    cache_flag = "" if "pip_no_cache" in active_issues else " --no-cache-dir"
    lines.append(f"RUN pip install{cache_flag} -r requirements.txt")

    if "copy_order_bad" not in active_issues:
        lines.append("COPY . .")

    return lines


def _build_expose_lines(active_issues: Set[str], rng: random.Random) -> list:
    lines = ["EXPOSE 8080"]
    if "exposed_cache_port" in active_issues:
        lines.append("EXPOSE 6379")
    if "exposed_db_port" in active_issues:
        lines.append(f"EXPOSE {rng.choice(['5432', '3306'])}")
    return lines


def _build_healthcheck_lines(active_issues: Set[str]) -> list:
    if "no_healthcheck" in active_issues:
        return []
    if "healthcheck_cmd_true" in active_issues:
        return ["HEALTHCHECK CMD true"]
    return ['HEALTHCHECK CMD python -c "import sys"']
