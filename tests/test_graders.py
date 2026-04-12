"""Tests for grading logic, randomization, and 2-step episode flow."""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.generators import select_issues, generate_dockerfile, ISSUE_POOLS
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from graders.analysis_grader import AnalysisGrader
from tools.scanner import select_vuln_subset
from server.docker_hardening_environment import DockerHardeningEnvironment
from models import DockerHardeningAction


# ── Randomization tests ──────────────────────────────────────────────────────

def test_different_seeds_produce_different_issues():
    """Different random seeds should produce different active issue sets."""
    seen = set()
    for seed in range(20):
        rng = random.Random(seed)
        issues = select_issues("patch_easy", rng)
        seen.add(frozenset(issues))
    assert len(seen) >= 3, f"Only {len(seen)} unique issue sets in 20 seeds"


def test_different_seeds_produce_different_dockerfiles():
    """Different seeds should produce different Dockerfiles."""
    dockerfiles = set()
    for seed in range(20):
        rng = random.Random(seed)
        issues = select_issues("patch_easy", rng)
        df, _, _ = generate_dockerfile("patch_easy", issues, rng)
        dockerfiles.add(df)
    assert len(dockerfiles) >= 3, f"Only {len(dockerfiles)} unique Dockerfiles in 20 seeds"


def test_generated_dockerfile_is_valid():
    """Every generated Dockerfile should have FROM and CMD."""
    for task in ["patch_easy", "patch_medium", "patch_hard"]:
        for seed in range(10):
            rng = random.Random(seed)
            issues = select_issues(task, rng)
            df, tag, meta = generate_dockerfile(task, issues, rng)
            assert "FROM " in df, f"Missing FROM in {task} seed={seed}"
            assert "CMD " in df, f"Missing CMD in {task} seed={seed}"
            assert len(issues) == meta["num_issues"]


def test_vuln_subset_varies():
    """CVE pools should vary across seeds."""
    pools = set()
    for seed in range(20):
        rng = random.Random(seed)
        subset = select_vuln_subset(1, rng)
        key = tuple(sorted((s, p) for s, p, _, _ in subset))
        pools.add(key)
    assert len(pools) >= 2, f"Only {len(pools)} unique CVE pools in 20 seeds"


# ── Grader tests ─────────────────────────────────────────────────────────────

def test_easy_grader_good_patch():
    """A good patch should score higher than a no-op."""
    rng = random.Random(42)
    issues = select_issues("patch_easy", rng)
    df, tag, _ = generate_dockerfile("patch_easy", issues, rng)
    vuln_pool = select_vuln_subset(1, rng)

    grader = EasyGrader()

    noop_score, _ = grader.score(df, df, tag, issues, vuln_pool)

    good_patch = (
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
    good_score, breakdown = grader.score(df, good_patch, tag, issues, vuln_pool)

    assert good_score > noop_score, (
        f"Good patch ({good_score:.3f}) should score higher than noop ({noop_score:.3f})"
    )
    assert 0.01 <= good_score <= 0.99
    assert 0.01 <= noop_score <= 0.99


def test_medium_grader_penalizes_secrets():
    """Medium grader should penalize remaining secrets."""
    grader = MediumGrader()
    rng = random.Random(99)
    issues = select_issues("patch_medium", rng)
    df, tag, _ = generate_dockerfile("patch_medium", issues, rng)
    vuln_pool = select_vuln_subset(2, rng)

    with_secrets = df.replace("FROM python:", "FROM python:3.12-slim\n# still has secrets\nFROM python:")
    score, breakdown = grader.score(df, df, tag, issues, vuln_pool)
    assert 0.01 <= score <= 0.99


def test_hard_grader_clamps_score():
    """Hard grader scores should always be in [0.01, 0.99]."""
    grader = HardGrader()
    rng = random.Random(7)
    issues = select_issues("patch_hard", rng)
    df, tag, _ = generate_dockerfile("patch_hard", issues, rng)
    vuln_pool = select_vuln_subset(3, rng)

    score, _ = grader.score(df, df, tag, issues, vuln_pool)
    assert 0.01 <= score <= 0.99

    good_patch = (
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
    score2, _ = grader.score(df, good_patch, tag, issues, vuln_pool)
    assert 0.01 <= score2 <= 0.99


# ── Analysis grader tests ────────────────────────────────────────────────────

def test_analysis_perfect_score():
    """Perfect issue identification should score high."""
    grader = AnalysisGrader()
    active = {"pipe_to_shell", "no_user", "no_healthcheck", "apt_no_cleanup"}

    claims = {
        "identified_issues": [
            "curl pipe to shell pattern detected",
            "running as root — no USER instruction",
            "no healthcheck defined",
            "apt cache not cleaned",
        ],
        "identified_categories": ["supply_chain", "best_practice"],
    }
    score, feedback = grader.score(claims, active)
    assert score > 0.5, f"Perfect analysis should score > 0.5, got {score:.3f}"
    assert len(feedback["issues_confirmed"]) >= 3


def test_analysis_empty_claims():
    """Empty analysis should score very low."""
    grader = AnalysisGrader()
    active = {"pipe_to_shell", "no_user", "no_healthcheck"}
    claims = {"identified_issues": [], "identified_categories": []}
    score, feedback = grader.score(claims, active)
    assert score <= 0.1, f"Empty analysis should score < 0.1, got {score:.3f}"


def test_analysis_false_positives_penalized():
    """Claims that don't match any real issue should not help."""
    grader = AnalysisGrader()
    active = {"no_user"}
    claims = {
        "identified_issues": [
            "SQL injection vulnerability",
            "XSS in template rendering",
            "running as root",
        ],
        "identified_categories": ["best_practice", "vulnerability", "network_security"],
    }
    score, feedback = grader.score(claims, active)
    assert len(feedback["false_positives"]) >= 2


def test_analysis_negation_rejected():
    """Claims that dismiss an issue should not count."""
    grader = AnalysisGrader()
    active = {"pipe_to_shell"}
    claims = {
        "identified_issues": ["curl pipe to shell is not an issue in this case"],
        "identified_categories": [],
    }
    score, feedback = grader.score(claims, active)
    assert "pipe_to_shell" not in feedback["issues_confirmed"]


# ── Full 2-step environment tests ────────────────────────────────────────────

def test_env_2step_flow():
    """Full 2-step episode: analyze → patch."""
    os.environ["SCA_GYM_TASK"] = "patch_easy"
    env = DockerHardeningEnvironment()
    obs = env.reset()

    assert not obs.done
    assert obs.step_number == 0
    assert obs.current_dockerfile
    assert obs.vulnerability_summary

    # Step 1: Analyze
    analysis = json.dumps({
        "identified_issues": ["running as root", "pipe-to-shell pattern"],
        "identified_categories": ["best_practice", "supply_chain"],
    })
    action = DockerHardeningAction(patched_dockerfile=analysis)
    obs = env.step(action)

    assert not obs.done, "Should NOT be done after analysis step"
    assert obs.step_number == 1
    assert obs.reward > 0
    assert "Analysis Feedback" in obs.vulnerability_summary

    # Step 2: Patch
    good_patch = (
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
    action = DockerHardeningAction(patched_dockerfile=good_patch)
    obs = env.step(action)

    assert obs.done, "Should be done after patch step"
    assert 0.01 <= obs.score <= 0.99
    assert 0.01 <= obs.reward <= 0.99


def test_env_randomization():
    """Multiple resets should produce different Dockerfiles."""
    os.environ["SCA_GYM_TASK"] = "patch_easy"
    env = DockerHardeningEnvironment()

    dockerfiles = set()
    for _ in range(10):
        obs = env.reset()
        dockerfiles.add(obs.current_dockerfile)

    assert len(dockerfiles) >= 2, (
        f"Only {len(dockerfiles)} unique Dockerfile(s) in 10 resets"
    )


def test_env_skip_analysis():
    """Agent can skip analysis and go directly to patch."""
    os.environ["SCA_GYM_TASK"] = "patch_easy"
    env = DockerHardeningEnvironment()
    obs = env.reset()

    good_patch = (
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
    action = DockerHardeningAction(patched_dockerfile=good_patch)
    obs = env.step(action)

    assert obs.done
    assert 0.01 <= obs.score <= 0.99


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
