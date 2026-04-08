# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core environment logic — task definitions, scoring, and the step/reset loop."""

import os
import re
import sys
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        DockerHardeningAction, DockerHardeningObservation,
        DockerHardeningState, VulnReport, PatchAttempt,
        PatchStrategy, TerminationReason, Severity,
    )
    from ..tools.scanner import (
        scan_mock, check_best_practices, detect_antipatterns,
        validate_base_image_tag,
    )
    from ..tools.docker_manager import DockerBuildManager, DockerBuildError
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import (
        DockerHardeningAction, DockerHardeningObservation,
        DockerHardeningState, VulnReport, PatchAttempt,
        PatchStrategy, TerminationReason, Severity,
    )
    from tools.scanner import (
        scan_mock, check_best_practices, detect_antipatterns,
        validate_base_image_tag,
    )
    from tools.docker_manager import DockerBuildManager, DockerBuildError


def _get_mode() -> str:
    mode = os.environ.get("SCA_GYM_MODE", "eval").strip().lower()
    return mode if mode in ("eval", "train") else "eval"


TASKS = {
    # ── Hackathon tasks (difficulty 1-3) ──────────────────────────────────
    "patch_easy": {
        "difficulty": 1,
        "image_tag": "python:3.11-slim",
        "max_steps": 4,
        "dockerfile": (
            "FROM python:3.11-slim\n"
            "RUN apt-get update && apt-get install -y curl wget\n"
            "RUN curl -sSL https://install.example.com/setup.sh | bash\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8080\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    "patch_medium": {
        "difficulty": 2,
        "image_tag": "python:3.9-slim",
        "max_steps": 4,
        "dockerfile": (
            "FROM python:3.9-slim\n"
            "ARG DB_PASSWORD=admin123\n"
            "ENV APP_SECRET_KEY=my-secret-key-do-not-share\n"
            "RUN apt-get update && apt-get install -y curl wget git\n"
            "RUN curl -sSL https://raw.githubusercontent.com/example/setup/main/init.sh | sh\n"
            "WORKDIR /app\n"
            "ADD requirements.txt /app/requirements.txt\n"
            "RUN pip install -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8080\n"
            "EXPOSE 6379\n"
            "HEALTHCHECK CMD true\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    "patch_hard": {
        "difficulty": 3,
        "image_tag": "python:3.6-slim",
        "max_steps": 3,
        "dockerfile": (
            "FROM python:3.6-slim\n"
            "ENV DATABASE_PASSWORD=s3cret_passw0rd\n"
            "ENV API_TOKEN=sk-1234567890abcdef\n"
            "ENV FLASK_SECRET_KEY=super-secret-key-123\n"
            "ARG DEPLOY_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "RUN apt-get update && apt-get install -y \\\n"
            "    libssl1.0 \\\n"
            "    curl \\\n"
            "    wget \\\n"
            "    gcc \\\n"
            "    python3-dev \\\n"
            "    && curl -sSL https://get.docker.com | sh\n"
            "ADD https://example.com/install.sh /tmp/install.sh\n"
            "RUN chmod +x /tmp/install.sh && sh /tmp/install.sh\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 5432\n"
            "EXPOSE 6379\n"
            "EXPOSE 3306\n"
            "HEALTHCHECK CMD true\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    # ── Training tasks (difficulty 4-7) ───────────────────────────────────
    "patch_multistage": {
        "difficulty": 4,
        "image_tag": "python:3.9-slim",
        "max_steps": 5,
        "dockerfile": (
            "FROM python:3.9-slim AS builder\n"
            "RUN apt-get update && apt-get install -y gcc python3-dev libffi-dev\n"
            "WORKDIR /build\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "\n"
            "FROM python:3.9-slim\n"
            "RUN apt-get update && apt-get install -y curl\n"
            "WORKDIR /app\n"
            "COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages\n"
            "COPY --from=builder /usr/local/bin /usr/local/bin\n"
            "COPY . .\n"
            "EXPOSE 8080\n"
            "HEALTHCHECK CMD true\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    "patch_conflict": {
        "difficulty": 5,
        "image_tag": "python:3.9-slim",
        "max_steps": 5,
        "dockerfile": (
            "FROM python:3.9-slim\n"
            "RUN apt-get update && apt-get install -y \\\n"
            "    libssl1.1 \\\n"
            "    curl \\\n"
            "    wget \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8080\n"
            "EXPOSE 6379\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    "patch_subtle": {
        "difficulty": 6,
        "image_tag": "python:3.9-slim",
        "max_steps": 4,
        "dockerfile": (
            "FROM python:3.9-slim\n"
            "ARG REPO_URL=https://deploy:ghp_s3cretToken123@git.example.com/org/repo.git\n"
            "ENV AUTH_TOKEN=ZGVwbG95OnMzY3JldFBhc3N3b3JkMTIz\n"
            "RUN apt-get update && apt-get install -y curl wget git\n"
            "RUN git clone ${REPO_URL} /src\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "COPY requirements.txt .\n"
            "RUN pip install -r requirements.txt\n"
            "RUN curl -sSL https://install.example.com/agent.sh | bash\n"
            "EXPOSE 8080\n"
            "HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
    "patch_adversarial": {
        "difficulty": 7,
        "image_tag": "python:3.9-slim",
        "max_steps": 3,
        "dockerfile": (
            "FROM python:3.9-slim\n"
            "ENV APP_NAME=myservice\n"
            "RUN apt-get update && apt-get install -y \\\n"
            "    gcc \\\n"
            "    python3-dev \\\n"
            "    curl \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "USER root\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "RUN chmod 777 /app/tmp\n"
            "EXPOSE 8080\n"
            "HEALTHCHECK CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8080/health')\"\n"
            'CMD ["python", "app.py"]\n'
        ),
    },
}

DEFAULT_TASK = "patch_easy"

_W_VULN = 0.50
_W_BP   = 0.30
_W_AP   = 0.20

_W_VULN_HARD = 0.40
_W_BP_HARD   = 0.30
_W_AP_HARD   = 0.30

_SEVERITY_WEIGHT = {
    Severity.CRITICAL:   4.0,
    Severity.HIGH:       3.0,
    Severity.MEDIUM:     2.0,
    Severity.LOW:        1.0,
    Severity.NEGLIGIBLE: 0.5,
    Severity.UNKNOWN:    1.0,
}

_STEP_COST        = 0.02
_NOOP_PENALTY     = 0.05
_BUILD_FAIL_COST  = 0.03
_REGRESSION_COST  = 0.10
_EFFICIENCY_BONUS = {1: 0.10, 2: 0.05}

_BP_LABELS = {
    "non_root_user":    "Non-root USER instruction",
    "healthcheck":      "HEALTHCHECK instruction defined",
    "no_secrets_in_env": "No secrets in ENV/ARG instructions",
    "apt_cache_cleanup": "APT cache cleaned (rm -rf /var/lib/apt/lists/*)",
    "pip_no_cache":     "pip --no-cache-dir flag used",
    "copy_over_add":    "COPY used instead of ADD",
    "modern_base_image": "Modern base image (Python >= 3.12)",
    "layer_efficiency": "Consecutive RUN commands merged (max 2)",
    "copy_order":       "COPY requirements.txt before COPY . . (cache efficiency)",
    "minimal_packages": "apt-get install uses --no-install-recommends",
}

_REGRESSION_PREFIXES = ("REGRESSION-", "CONFLICT-", "BUILD-FAIL-")


_PIP_PACKAGES = {
    "setuptools", "requests", "pyyaml", "urllib3", "certifi",
    "pillow", "cryptography", "flask", "jinja2", "werkzeug",
    "numpy", "scipy", "pandas",
}


def _format_vuln_summary(
    report: VulnReport, bp_results: dict,
    ap_warnings: List[str], difficulty: int,
) -> str:
    mode = _get_mode()
    lines = [
        "=== Security Scan Report ===",
        f"Vulnerabilities found: {report.total_count}",
        f"  CRITICAL: {report.critical_count}  |  HIGH: {report.high_count}",
        "",
    ]

    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"]
    pip_vulns = []

    for v in sorted(
        report.vulnerabilities,
        key=lambda x: severity_order.index(x.severity.value)
        if x.severity.value in severity_order else 99,
    ):
        is_pip = v.package_name.lower() in _PIP_PACKAGES
        is_regression = any(v.cve_id.startswith(p) for p in _REGRESSION_PREFIXES)

        if is_regression:
            lines.append(f"  [REGRESSION] {v.cve_id} | {v.package_name}")
            lines.append(f"    {v.description}")
            continue

        if mode == "eval":
            if is_pip and difficulty >= 2:
                fix = f" -> fix: pip install --upgrade {v.package_name}>={v.fixed_version}"
                pip_vulns.append(v)
            elif v.fixed_version:
                fix = f" -> fix: {v.fixed_version}"
            else:
                fix = " (no fix available)"
        else:
            if is_pip:
                pip_vulns.append(v)
                fix = " (Python dependency — requires explicit pip upgrade)"
            elif v.fixed_version is None:
                fix = " (no fix available)"
            elif difficulty >= 4:
                fix = " (system package — upgrade base or apt-get upgrade)"
            else:
                fix = f" -> fix: {v.fixed_version}"

        lines.append(f"  [{v.severity.value}] {v.cve_id} | {v.package_name} {v.installed_version}{fix}")
        lines.append(f"    {v.description}")

    if pip_vulns and mode == "eval":
        pip_cmd_parts = [f"{v.package_name}>={v.fixed_version}" for v in pip_vulns if v.fixed_version]
        if pip_cmd_parts:
            lines += [
                "",
                "=== Python Dependency Fix (add this RUN line to your Dockerfile) ===",
                f"  RUN pip install --no-cache-dir --upgrade {' '.join(pip_cmd_parts)}",
            ]

    lines += ["", "=== Best Practices Checklist ==="]
    for key, label in _BP_LABELS.items():
        if key not in bp_results:
            continue
        status = "PASS" if bp_results.get(key, False) else "FAIL"
        lines.append(f"  [{status}] {label}")

    if ap_warnings:
        lines += ["", "=== Security Anti-Patterns Detected ==="]
        for w in ap_warnings:
            lines.append(f"  [WARN] {w}")

    return "\n".join(lines)


def _format_bp_list(bp_results: dict) -> List[str]:
    return [
        f"[{'PASS' if bp_results.get(key, False) else 'FAIL'}] {label}"
        for key, label in _BP_LABELS.items()
        if key in bp_results
    ]


def _diff_summary(old_df: str, new_df: str) -> str:
    old_lines = old_df.strip().splitlines()
    new_lines = new_df.strip().splitlines()
    added = [l for l in new_lines if l not in old_lines]
    removed = [l for l in old_lines if l not in new_lines]
    parts = []
    if removed:
        parts.append("Removed: " + "; ".join(removed[:5]))
    if added:
        parts.append("Added: " + "; ".join(added[:5]))
    return " | ".join(parts) if parts else "No structural changes detected"


def _get_weights(difficulty: int):
    if difficulty >= 4:
        return _W_VULN_HARD, _W_BP_HARD, _W_AP_HARD
    return _W_VULN, _W_BP, _W_AP


def _compute_security_score(vuln_ratio_fixed: float, bp_results: dict, ap_count: int) -> float:
    bp_count = sum(1 for v in bp_results.values() if v)
    bp_total = len(bp_results) or 1
    return min(100.0,
               50.0 * vuln_ratio_fixed
               + 30.0 * (bp_count / bp_total)
               + 20.0 * max(0.0, 1.0 - ap_count * 0.2))


def _weighted_vuln_score(vulns: List) -> float:
    return sum(_SEVERITY_WEIGHT.get(v.severity, 1.0) for v in vulns)


def _count_regressions(vulns: List) -> int:
    return sum(1 for v in vulns if any(v.cve_id.startswith(p) for p in _REGRESSION_PREFIXES))


def _compute_improvement_score(
    initial_vulns: List, current_vulns: List,
    initial_bp: dict, current_bp: dict,
    initial_ap_count: int, current_ap_count: int,
    difficulty: int = 1,
) -> float:
    w_vuln, w_bp, w_ap = _get_weights(difficulty)

    initial_weight = _weighted_vuln_score(initial_vulns)
    current_weight = _weighted_vuln_score(current_vulns)
    vuln_improvement = max(0.0, (initial_weight - current_weight) / max(initial_weight, 1.0))

    init_bp_sat = sum(1 for v in initial_bp.values() if v)
    curr_bp_sat = sum(1 for v in current_bp.values() if v)
    improvable_bp = len(current_bp) - init_bp_sat
    bp_improvement = (
        max(0.0, (curr_bp_sat - init_bp_sat) / max(improvable_bp, 1))
        if improvable_bp > 0 else 0.0
    )

    ap_improvement = (
        max(0.0, (initial_ap_count - current_ap_count) / max(initial_ap_count, 1))
        if initial_ap_count > 0 else 0.0
    )

    return min(1.0, max(0.0,
        w_vuln * vuln_improvement + w_bp * bp_improvement + w_ap * ap_improvement
    ))


def _validate_dockerfile(content: str) -> Optional[str]:
    text = content.strip()
    if not text:
        return "Empty Dockerfile submitted"
    if "FROM" not in text.upper():
        return "Dockerfile must contain a FROM instruction"

    from_match = re.search(r"^\s*FROM\s+(\S+)", text, re.MULTILINE | re.IGNORECASE)
    if from_match:
        image_ref = from_match.group(1)
        if " " in image_ref:
            image_ref = image_ref.split()[0]
        tag_error = validate_base_image_tag(image_ref)
        if tag_error:
            return f"Invalid base image: {tag_error}"

    return None


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        blocks = re.findall(
            r"```(?:dockerfile|docker|Dockerfile)?\s*\n(.*?)```",
            stripped, re.DOTALL,
        )
        if blocks:
            return blocks[0].strip()
    return stripped


class DockerHardeningEnvironment(Environment):
    """OpenEnv environment — agent submits patched Dockerfiles, gets scored."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_name: str = ""
        self._difficulty: int = 1
        self._max_steps: int = 5
        self._image_tag: str = ""
        self._original_dockerfile: str = ""

        self._state = DockerHardeningState(episode_id=str(uuid4()), step_count=0)
        self._build_mgr: Optional[DockerBuildManager] = None
        self._initial_report: Optional[VulnReport] = None
        self._last_report: Optional[VulnReport] = None
        self._cycle_count: int = 0
        self._cumulative_reward: float = 0.0
        self._consecutive_failures: int = 0

        self._prev_bp: dict = {}
        self._prev_ap: List[str] = []
        self._initial_bp: dict = {}
        self._initial_ap: List[str] = []

    def reset(self) -> DockerHardeningObservation:
        if self._build_mgr:
            self._build_mgr.cleanup()

        task_name = os.environ.get("SCA_GYM_TASK", DEFAULT_TASK)
        if task_name not in TASKS:
            task_name = DEFAULT_TASK
        task = TASKS[task_name]

        self._task_name = task_name
        self._difficulty = task["difficulty"]
        self._max_steps = task["max_steps"]
        self._image_tag = task["image_tag"]
        self._original_dockerfile = task["dockerfile"]

        episode_id = str(uuid4())
        self._state = DockerHardeningState(
            episode_id=episode_id, step_count=0, task_name=task_name,
        )
        self._cycle_count = 0
        self._cumulative_reward = 0.0
        self._consecutive_failures = 0

        self._build_mgr = DockerBuildManager(
            base_image_tag=self._image_tag, episode_id=episode_id, use_mock=True,
        )
        self._build_mgr.initialise(self._original_dockerfile)

        initial_report = scan_mock(self._image_tag, difficulty=self._difficulty)
        self._initial_report = initial_report
        self._last_report = initial_report

        bp = check_best_practices(self._original_dockerfile)
        ap = detect_antipatterns(self._original_dockerfile)
        self._initial_bp = bp
        self._initial_ap = ap
        self._prev_bp = bp
        self._prev_ap = ap

        vuln_summary = _format_vuln_summary(initial_report, bp, ap, self._difficulty)
        sec_score = _compute_security_score(0.0, bp, len(ap))

        return DockerHardeningObservation(
            current_dockerfile=self._original_dockerfile,
            vulnerability_summary=vuln_summary,
            task_name=task_name,
            initial_vuln_count=initial_report.total_count,
            current_vuln_count=initial_report.total_count,
            step_number=0, max_steps=self._max_steps,
            score=0.0, done=False, reward=0.0,
            step_summary=(
                f"Task '{task_name}' started. "
                f"Fix {initial_report.total_count} vulnerabilities in {self._image_tag}."
            ),
            security_score=round(sec_score, 1),
            best_practices=_format_bp_list(bp),
            antipattern_warnings=ap,
        )

    def step(self, action: DockerHardeningAction) -> DockerHardeningObservation:
        self._state.step_count += 1
        self._cycle_count += 1
        step_num = self._state.step_count
        patched_df = action.patched_dockerfile

        error = _validate_dockerfile(patched_df)
        if error:
            self._consecutive_failures += 1
            done, reason = self._check_termination()
            return self._make_obs(
                reward=-_BUILD_FAIL_COST, done=done, reason=reason,
                step_summary=f"Invalid Dockerfile: {error}", error=error,
            )

        patched_df = _strip_markdown_fences(patched_df)

        if patched_df.strip() == self._build_mgr.current_dockerfile.strip():
            self._consecutive_failures += 1
            done, reason = self._check_termination()
            return self._make_obs(
                reward=-_NOOP_PENALTY, done=done, reason=reason,
                step_summary="Dockerfile unchanged — no improvement (repeat penalty applied).",
            )

        try:
            new_tag, _ = self._build_mgr.apply_patch(patched_df)
        except DockerBuildError as e:
            self._consecutive_failures += 1
            done, reason = self._check_termination()
            return self._make_obs(
                reward=-_BUILD_FAIL_COST, done=done, reason=reason,
                step_summary=f"Build failed: {str(e)[:200]}",
                error=str(e)[:200],
            )

        new_report = scan_mock(new_tag, difficulty=self._difficulty, current_dockerfile=patched_df)
        report_before = self._last_report
        self._last_report = new_report
        self._consecutive_failures = 0

        new_bp = check_best_practices(patched_df)
        new_ap = detect_antipatterns(patched_df)

        initial_vulns = self._initial_report.vulnerabilities if self._initial_report else []
        prev_vulns = report_before.vulnerabilities if report_before else initial_vulns
        initial_count = len(initial_vulns)

        prev_score = _compute_improvement_score(
            initial_vulns, prev_vulns,
            self._initial_bp, self._prev_bp,
            len(self._initial_ap), len(self._prev_ap),
            self._difficulty,
        )
        curr_score = _compute_improvement_score(
            initial_vulns, new_report.vulnerabilities,
            self._initial_bp, new_bp,
            len(self._initial_ap), len(new_ap),
            self._difficulty,
        )
        step_reward = (curr_score - prev_score) - _STEP_COST

        regression_count = _count_regressions(new_report.vulnerabilities)
        if regression_count > 0:
            step_reward -= _REGRESSION_COST * regression_count

        if self._difficulty < 5 and new_report.total_count == 0 and step_num in _EFFICIENCY_BONUS:
            step_reward += _EFFICIENCY_BONUS[step_num]

        step_reward = min(step_reward, 1.0)
        self._cumulative_reward += max(0.0, step_reward)
        self._prev_bp = new_bp
        self._prev_ap = new_ap

        score = curr_score
        total_fixed = max(0, initial_count - new_report.total_count)
        vuln_improvement = max(0.0, total_fixed / max(initial_count, 1))
        sec_score = _compute_security_score(vuln_improvement, new_bp, len(new_ap))

        done, reason = self._check_termination()

        vulns_before = report_before.total_count if report_before else initial_count
        curr_bp_sat = sum(1 for v in new_bp.values() if v)

        diff_info = _diff_summary(self._build_mgr.current_dockerfile, patched_df)
        regression_info = ""
        if regression_count > 0:
            reg_names = [v.cve_id for v in new_report.vulnerabilities
                         if any(v.cve_id.startswith(p) for p in _REGRESSION_PREFIXES)]
            regression_info = f" REGRESSIONS: {', '.join(reg_names)}."

        step_summary = (
            f"Step {step_num}: {vulns_before} -> {new_report.total_count} vulns. "
            f"Best practices: {curr_bp_sat}/{len(new_bp)}. "
            f"Anti-patterns: {len(new_ap)}. "
            f"Reward: {step_reward:.3f}, Score: {score:.3f}"
            f"{regression_info}"
        )

        return self._make_obs(
            reward=step_reward, done=done, reason=reason,
            step_summary=step_summary, report=new_report,
            score=score, bp_results=new_bp, ap_warnings=new_ap,
            sec_score=sec_score,
        )

    @property
    def state(self) -> DockerHardeningState:
        return DockerHardeningState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_name=self._task_name,
            cumulative_reward=self._cumulative_reward,
            cycle_count=self._cycle_count,
        )

    def _make_obs(
        self,
        reward: float, done: bool, reason: Optional[TerminationReason],
        step_summary: str,
        report: Optional[VulnReport] = None,
        score: Optional[float] = None,
        error: Optional[str] = None,
        bp_results: Optional[dict] = None,
        ap_warnings: Optional[List[str]] = None,
        sec_score: Optional[float] = None,
    ) -> DockerHardeningObservation:
        rpt = report or self._last_report
        initial_vulns = self._initial_report.vulnerabilities if self._initial_report else []
        current_vulns = rpt.vulnerabilities if rpt else initial_vulns
        current_count = len(current_vulns)

        bp = bp_results or self._prev_bp
        ap = ap_warnings if ap_warnings is not None else self._prev_ap

        if score is None:
            score = _compute_improvement_score(
                initial_vulns, current_vulns,
                self._initial_bp, bp,
                len(self._initial_ap), len(ap),
                self._difficulty,
            )

        initial_count = len(initial_vulns)
        if sec_score is None:
            vuln_ratio = max(0.0, (initial_count - current_count) / max(initial_count, 1))
            sec_score = _compute_security_score(vuln_ratio, bp, len(ap))

        vuln_summary = _format_vuln_summary(rpt, bp, ap, self._difficulty) if rpt else ""

        return DockerHardeningObservation(
            current_dockerfile=self._build_mgr.current_dockerfile if self._build_mgr else "",
            vulnerability_summary=vuln_summary,
            task_name=self._task_name,
            initial_vuln_count=initial_count,
            current_vuln_count=current_count,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            score=min(score, 1.0),
            step_summary=step_summary,
            last_action_error=error,
            termination_reason=reason,
            done=done,
            reward=round(reward, 4),
            security_score=round(sec_score, 1),
            best_practices=_format_bp_list(bp),
            antipattern_warnings=ap,
        )

    def _check_termination(self):
        if self._last_report and self._last_report.total_count == 0:
            return True, TerminationReason.ALL_VULNS_FIXED
        if self._cycle_count >= self._max_steps:
            return True, TerminationReason.MAX_CYCLES_REACHED
        if self._consecutive_failures >= 3:
            return True, TerminationReason.PATCH_FAILED_TOO_MANY_TIMES
        return False, None
