"""
Unit tests — run fully offline (mock mode, no Docker, no API calls).

Usage:
    cd docker_hardening && python -m pytest tests/ -v
"""

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    DockerHardeningAction, DockerHardeningObservation, DockerHardeningState,
    Severity, VulnReport, Vulnerability, PatchAttempt, PatchStrategy,
    TerminationReason,
)
from tools.scanner import (
    scan_mock, run_scan, check_best_practices, detect_antipatterns,
)
from tools.docker_manager import DockerBuildManager
from server.docker_hardening_environment import DockerHardeningEnvironment, TASKS


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────

SAMPLE_DOCKERFILE = """\
FROM python:3.9-slim
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
"""


def make_report(critical=0, high=0, medium=0, low=0, image="test:latest") -> VulnReport:
    vulns = []
    counter = 1000

    def add(sev, count):
        nonlocal counter
        for _ in range(count):
            vulns.append(Vulnerability(
                cve_id=f"CVE-2024-{counter}", package_name="pkg",
                installed_version="1.0", fixed_version="2.0",
                severity=sev, description="test",
            ))
            counter += 1

    add(Severity.CRITICAL, critical)
    add(Severity.HIGH, high)
    add(Severity.MEDIUM, medium)
    add(Severity.LOW, low)

    return VulnReport(
        image_tag=image, scan_tool="mock",
        scanned_at=datetime.datetime.utcnow().isoformat(),
        vulnerabilities=vulns,
    )


def make_env(task="patch_easy") -> DockerHardeningEnvironment:
    os.environ["SCA_GYM_TASK"] = task
    return DockerHardeningEnvironment()


# ───────────────────────────────────────────────────────────────────────────
# Models
# ───────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_vuln_report_counts(self):
        r = make_report(critical=2, high=3, medium=1, low=4)
        assert r.critical_count == 2
        assert r.high_count == 3
        assert r.total_count == 10

    def test_vuln_report_summary(self):
        r = make_report(critical=1, high=2)
        s = r.summary()
        assert s["CRITICAL"] == 1
        assert s["HIGH"] == 2
        assert s["MEDIUM"] == 0

    def test_hardening_action_field(self):
        a = DockerHardeningAction(patched_dockerfile="FROM python:3.12-slim\n")
        assert a.patched_dockerfile == "FROM python:3.12-slim\n"

    def test_observation_defaults(self):
        obs = DockerHardeningObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.task_name == ""
        assert obs.score == 0.0

    def test_state_model(self):
        s = DockerHardeningState(
            episode_id="ep-1", step_count=3,
            task_name="patch_easy", cumulative_reward=0.5, cycle_count=2,
        )
        assert s.task_name == "patch_easy"
        assert s.cycle_count == 2


# ───────────────────────────────────────────────────────────────────────────
# Scanner (mock)
# ───────────────────────────────────────────────────────────────────────────

class TestScanner:
    def test_mock_difficulty_1(self):
        r = scan_mock("test:latest", difficulty=1)
        assert r.scan_tool == "mock"
        assert r.critical_count == 0
        assert r.total_count > 0

    def test_mock_difficulty_3(self):
        r = scan_mock("test:latest", difficulty=3)
        assert r.critical_count > 0
        assert r.high_count > 0

    def test_mock_deterministic(self):
        r1 = scan_mock("same-image:v1", difficulty=2)
        r2 = scan_mock("same-image:v1", difficulty=2)
        assert r1.total_count == r2.total_count

    def test_run_scan_mock(self):
        r = run_scan("test:latest", scanner="mock", difficulty=2)
        assert r.scan_tool == "mock"

    def test_mock_dockerfile_upgrade_reduces_vulns(self):
        baseline = scan_mock("img:v1", difficulty=3)
        patched_df = "FROM python:3.12-slim\nRUN apt-get update && apt-get upgrade -y\nCOPY . .\n"
        after = scan_mock("img:v2", difficulty=3, current_dockerfile=patched_df)
        assert after.total_count < baseline.total_count
        assert after.critical_count < baseline.critical_count

    def test_mock_dockerfile_no_change_same_vulns(self):
        baseline = scan_mock("img:v1", difficulty=2)
        old_df = "FROM python:3.9-slim\nRUN apt-get update && apt-get install -y curl wget\n"
        after = scan_mock("img:v1", difficulty=2, current_dockerfile=old_df)
        assert after.total_count == baseline.total_count

    def test_mock_log4j_removed_on_python_image(self):
        patched_df = "FROM python:3.12-slim\nCOPY . .\n"
        r = scan_mock("img:v1", difficulty=3, current_dockerfile=patched_df)
        pkg_names = [v.package_name for v in r.vulnerabilities]
        assert "log4j" not in pkg_names

    def test_run_scan_unknown(self):
        with pytest.raises(Exception, match="Unknown scanner"):
            run_scan("test:latest", scanner="nonexistent")

    @pytest.mark.parametrize("difficulty", [4, 5, 6, 7])
    def test_mock_higher_difficulties(self, difficulty):
        r = scan_mock("test:latest", difficulty=difficulty)
        assert r.total_count > 0

    def test_no_fix_vulns_persist(self):
        patched_df = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "RUN pip install --no-cache-dir --upgrade cryptography>=42.0.0 "
            "flask>=2.3.3 setuptools>=70.0.0 numpy>=1.26.4 pyyaml>=6.0.1 scipy>=1.12.0\n"
            "COPY . .\n"
        )
        r = scan_mock("img:v1", difficulty=5, current_dockerfile=patched_df)
        no_fix_pkgs = [v.package_name for v in r.vulnerabilities
                       if v.fixed_version is None
                       and not v.cve_id.startswith("REGRESSION")]
        assert len(no_fix_pkgs) > 0, "No-fix vulns should persist regardless of patching"

    def test_alpine_regression_detected(self):
        patched_df = (
            "FROM python:3.12-alpine\n"
            "RUN pip install --no-cache-dir numpy>=1.26.4\n"
            "COPY . .\n"
        )
        r = scan_mock("img:v1", difficulty=5, current_dockerfile=patched_df)
        regression_ids = [v.cve_id for v in r.vulnerabilities]
        assert "REGRESSION-003" in regression_ids

    def test_conflict_detected_crypto_libssl(self):
        patched_df = (
            "FROM python:3.9-slim\n"
            "RUN apt-get update && apt-get install -y libssl1.1\n"
            "RUN pip install --no-cache-dir --upgrade cryptography>=42.0.0\n"
            "COPY . .\n"
        )
        r = scan_mock("img:v1", difficulty=5, current_dockerfile=patched_df)
        conflict_ids = [v.cve_id for v in r.vulnerabilities]
        assert "CONFLICT-001" in conflict_ids

    def test_decoy_package_not_removable(self):
        patched_df = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "RUN pip install --no-cache-dir --upgrade cryptography>=42.0.0\n"
            "COPY . .\n"
        )
        r = scan_mock("img:v1", difficulty=7, current_dockerfile=patched_df)
        pkg_names = [v.package_name for v in r.vulnerabilities
                     if not v.cve_id.startswith(("REGRESSION", "CONFLICT", "BUILD-FAIL"))]
        assert "myapp-crypto" in pkg_names, "Decoy package should persist"


# ───────────────────────────────────────────────────────────────────────────
# Best-practice checks (new checks)
# ───────────────────────────────────────────────────────────────────────────

class TestBestPractices:
    def test_layer_efficiency_pass(self):
        df = "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y curl\nCOPY . .\n"
        bp = check_best_practices(df)
        assert bp["layer_efficiency"] is True

    def test_layer_efficiency_fail(self):
        df = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update\n"
            "RUN apt-get install -y curl\n"
            "RUN pip install flask\n"
            "COPY . .\n"
        )
        bp = check_best_practices(df)
        assert bp["layer_efficiency"] is False

    def test_copy_order_pass(self):
        df = "FROM python:3.12-slim\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\n"
        bp = check_best_practices(df)
        assert bp["copy_order"] is True

    def test_copy_order_fail(self):
        df = "FROM python:3.12-slim\nCOPY . .\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n"
        bp = check_best_practices(df)
        assert bp["copy_order"] is False

    def test_minimal_packages_pass(self):
        df = "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y --no-install-recommends curl\nCOPY . .\n"
        bp = check_best_practices(df)
        assert bp["minimal_packages"] is True

    def test_minimal_packages_fail(self):
        df = "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y curl\nCOPY . .\n"
        bp = check_best_practices(df)
        assert bp["minimal_packages"] is False


# ───────────────────────────────────────────────────────────────────────────
# Anti-pattern detection (new patterns)
# ───────────────────────────────────────────────────────────────────────────

class TestAntipatterns:
    def test_detect_cred_in_url(self):
        df = 'FROM python:3.12-slim\nARG REPO=https://user:pass@git.example.com/repo\nCOPY . .\n'
        ap = detect_antipatterns(df)
        assert any("Credentials embedded in URL" in w for w in ap)

    def test_detect_base64_secret(self):
        df = 'FROM python:3.12-slim\nENV TOKEN=ZGVwbG95OnMzY3JldFBhc3N3b3JkMTIz\nCOPY . .\n'
        ap = detect_antipatterns(df)
        assert any("base64" in w.lower() for w in ap)

    def test_detect_docker_in_docker(self):
        df = 'FROM python:3.12-slim\nRUN curl -sSL https://get.docker.com | sh\nCOPY . .\n'
        ap = detect_antipatterns(df)
        assert any("Docker-in-Docker" in w for w in ap)

    def test_detect_chmod_777(self):
        df = 'FROM python:3.12-slim\nRUN chmod 777 /app/tmp\nCOPY . .\n'
        ap = detect_antipatterns(df)
        assert any("chmod 777" in w for w in ap)

    def test_detect_copy_order_violation(self):
        df = "FROM python:3.12-slim\nCOPY . .\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n"
        ap = detect_antipatterns(df)
        assert any("COPY . . before" in w for w in ap)

    def test_no_false_positive_clean_dockerfile(self):
        df = (
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "USER 1000\n"
            'CMD ["python", "app.py"]\n'
        )
        ap = detect_antipatterns(df)
        assert len(ap) == 0, f"Clean Dockerfile should have no anti-patterns, got: {ap}"


# ───────────────────────────────────────────────────────────────────────────
# Docker build manager (mock mode)
# ───────────────────────────────────────────────────────────────────────────

class TestDockerBuildManager:
    def test_initialise_and_apply_patch(self):
        mgr = DockerBuildManager(
            base_image_tag="python:3.9-slim", episode_id="test-ep-001", use_mock=True,
        )
        mgr.initialise(SAMPLE_DOCKERFILE)
        assert mgr.current_dockerfile == SAMPLE_DOCKERFILE

        patched = SAMPLE_DOCKERFILE.replace("python:3.9-slim", "python:3.11-slim")
        new_tag, log = mgr.apply_patch(patched)
        assert "cycle-1" in new_tag
        assert "[MOCK BUILD]" in log
        assert mgr.current_dockerfile == patched
        mgr.cleanup()

    def test_cycle_counter_increments(self):
        mgr = DockerBuildManager("img:v1", "ep-002", use_mock=True)
        mgr.initialise(SAMPLE_DOCKERFILE)
        mgr.apply_patch(SAMPLE_DOCKERFILE + "\n# v1")
        tag, _ = mgr.apply_patch(SAMPLE_DOCKERFILE + "\n# v2")
        assert "cycle-2" in tag
        mgr.cleanup()


# ───────────────────────────────────────────────────────────────────────────
# Full environment (offline) — existing tasks
# ───────────────────────────────────────────────────────────────────────────

class TestDockerHardeningEnvironment:
    def test_reset_returns_observation(self):
        env = make_env("patch_easy")
        obs = env.reset()
        assert isinstance(obs, DockerHardeningObservation)
        assert obs.initial_vuln_count > 0
        assert obs.task_name == "patch_easy"
        assert not obs.done
        assert obs.vulnerability_summary != ""

    def test_reset_patch_hard(self):
        env = make_env("patch_hard")
        obs = env.reset()
        assert obs.task_name == "patch_hard"
        assert obs.initial_vuln_count > 0

    def test_step_with_patched_dockerfile(self):
        env = make_env("patch_easy")
        env.reset()
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "WORKDIR /app\nCOPY . .\n"
            'CMD ["python", "app.py"]\n'
        )
        action = DockerHardeningAction(patched_dockerfile=patched)
        obs = env.step(action)
        assert isinstance(obs, DockerHardeningObservation)
        assert obs.reward is not None

    def test_step_reduces_vulns(self):
        env = make_env("patch_medium")
        obs_reset = env.reset()
        initial = obs_reset.initial_vuln_count
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "WORKDIR /app\nCOPY . .\n"
            'CMD ["python", "app.py"]\n'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert obs.current_vuln_count < initial

    def test_step_empty_dockerfile_error(self):
        env = make_env("patch_easy")
        env.reset()
        obs = env.step(DockerHardeningAction(patched_dockerfile=""))
        assert obs.last_action_error is not None
        assert obs.reward <= 0.01

    def test_step_no_change_penalized(self):
        env = make_env("patch_easy")
        obs_reset = env.reset()
        original = obs_reset.current_dockerfile
        obs = env.step(DockerHardeningAction(patched_dockerfile=original))
        assert obs.reward <= 0.01

    def test_episode_terminates_within_max_steps(self):
        env = make_env("patch_easy")
        env.reset()
        done = False
        steps = 0
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "COPY . .\n"
            'CMD ["python", "app.py"]\n'
        )
        while not done and steps < 10:
            obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
            done = obs.done
            steps += 1
        assert done

    def test_score_in_range(self):
        env = make_env("patch_medium")
        env.reset()
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "COPY . .\n"
            'CMD ["python", "app.py"]\n'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert 0.0 <= obs.score <= 1.0

    def test_reward_in_range(self):
        env = make_env("patch_hard")
        env.reset()
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "COPY . .\n"
            'CMD ["python", "app.py"]\n'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_state_returns_state_model(self):
        env = make_env("patch_easy")
        env.reset()
        s = env.state
        assert isinstance(s, DockerHardeningState)
        assert s.task_name == "patch_easy"

    def test_state_tracks_steps(self):
        env = make_env("patch_easy")
        env.reset()
        patched = "FROM python:3.12-slim\nCOPY . .\n" 'CMD ["python", "app.py"]\n'
        env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert env.state.step_count == 1
        assert env.state.cycle_count == 1

    def test_markdown_fences_stripped(self):
        env = make_env("patch_easy")
        env.reset()
        fenced = (
            '```dockerfile\n'
            'FROM python:3.12-slim\n'
            'RUN apt-get update && apt-get upgrade -y\n'
            'COPY . .\n'
            'CMD ["python", "app.py"]\n'
            '```'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=fenced))
        assert obs.last_action_error is None

    def test_termination_max_cycles(self):
        env = make_env("patch_hard")
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            obs = env.step(DockerHardeningAction(
                patched_dockerfile="FROM python:3.9-slim\nCOPY . .\n"
            ))
            done = obs.done
            steps += 1
        assert done
        assert obs.termination_reason is not None


# ───────────────────────────────────────────────────────────────────────────
# Grader scores
# ───────────────────────────────────────────────────────────────────────────

class TestGraderScores:
    @pytest.mark.parametrize("task", ["patch_easy", "patch_medium", "patch_hard"])
    def test_perfect_patch_high_score(self, task):
        env = make_env(task)
        env.reset()
        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y\n"
            "WORKDIR /app\nCOPY . .\n"
            "USER 1000\n"
            'CMD ["python", "app.py"]\n'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert 0.0 <= obs.score <= 1.0
        assert obs.score > 0.0
        assert obs.reward >= 0.0

    @pytest.mark.parametrize("task", ["patch_easy", "patch_medium", "patch_hard"])
    def test_no_change_low_score(self, task):
        env = make_env(task)
        obs_reset = env.reset()
        original = obs_reset.current_dockerfile
        obs = env.step(DockerHardeningAction(patched_dockerfile=original))
        assert obs.score <= 0.02
        assert obs.reward <= 0.02


# ───────────────────────────────────────────────────────────────────────────
# New tasks (difficulty 4-7)
# ───────────────────────────────────────────────────────────────────────────

class TestRandomization:
    def test_different_resets_produce_different_dockerfiles(self):
        """Each reset should generate a different Dockerfile."""
        env = make_env("patch_easy")
        dockerfiles = set()
        for _ in range(10):
            obs = env.reset()
            dockerfiles.add(obs.current_dockerfile)
        assert len(dockerfiles) >= 2, \
            f"Only {len(dockerfiles)} unique Dockerfiles in 10 resets"

    def test_all_tasks_produce_valid_dockerfiles(self):
        for task in TASKS:
            env = make_env(task)
            obs = env.reset()
            assert "FROM " in obs.current_dockerfile
            assert obs.initial_vuln_count > 0
            assert not obs.done


# ───────────────────────────────────────────────────────────────────────────
# Mode toggle
# ───────────────────────────────────────────────────────────────────────────

class TestVulnSummary:
    def test_vulnerability_summary_has_scan_report(self):
        env = make_env("patch_medium")
        obs = env.reset()
        assert "Security Scan Report" in obs.vulnerability_summary

    def test_vulnerability_summary_has_best_practices(self):
        env = make_env("patch_medium")
        obs = env.reset()
        assert "Best Practices Checklist" in obs.vulnerability_summary


# ───────────────────────────────────────────────────────────────────────────
# Regression penalties
# ───────────────────────────────────────────────────────────────────────────

class TestRegressionPenalty:
    def test_latest_tag_causes_regression(self):
        env = make_env("patch_easy")
        env.reset()
        patched = "FROM python:latest\nCOPY . .\n" 'CMD ["python", "app.py"]\n'
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert "REGRESSION" in obs.vulnerability_summary


# ───────────────────────────────────────────────────────────────────────────
# All tasks defined in TASKS dict
# ───────────────────────────────────────────────────────────────────────────

class TestAllTasksAccessible:
    @pytest.mark.parametrize("task", list(TASKS.keys()))
    def test_task_defined(self, task):
        assert "difficulty" in TASKS[task]
        assert "max_steps" in TASKS[task]

    @pytest.mark.parametrize("task", list(TASKS.keys()))
    def test_task_reset_and_step(self, task):
        env = make_env(task)
        obs = env.reset()
        assert obs.initial_vuln_count > 0

        patched = (
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*\n"
            "WORKDIR /app\nCOPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\nUSER 1000\n"
            'CMD ["python", "app.py"]\n'
        )
        obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
        assert obs.reward is not None
