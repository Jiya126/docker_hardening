"""
Core environment logic — 2-step episodes with randomized Dockerfiles.

Step 1 (Analyze): Agent identifies security issues → receives actionable feedback.
Step 2 (Patch):   Agent submits a patched Dockerfile → graded on actual improvement.

Each reset() generates a DIFFERENT broken Dockerfile from a randomized pool,
preventing memorization and forcing genuine security reasoning.
"""

import os
import random
import re
import sys
from typing import Dict, List, Optional, Set, Tuple
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
        validate_base_image_tag, select_vuln_subset,
    )
    from ..tools.docker_manager import DockerBuildManager, DockerBuildError
    from ..tasks.generators import (
        select_issues, generate_dockerfile, ISSUE_CATEGORY_MAP,
    )
    from ..graders import GRADER_MAP
    from ..graders.analysis_grader import AnalysisGrader
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import (
        DockerHardeningAction, DockerHardeningObservation,
        DockerHardeningState, VulnReport, PatchAttempt,
        PatchStrategy, TerminationReason, Severity,
    )
    from tools.scanner import (
        scan_mock, check_best_practices, detect_antipatterns,
        validate_base_image_tag, select_vuln_subset,
    )
    from tools.docker_manager import DockerBuildManager, DockerBuildError
    from tasks.generators import (
        select_issues, generate_dockerfile, ISSUE_CATEGORY_MAP,
    )
    from graders import GRADER_MAP
    from graders.analysis_grader import AnalysisGrader


TASKS = {
    "patch_easy":   {"difficulty": 1, "max_steps": 2},
    "patch_medium": {"difficulty": 2, "max_steps": 2},
    "patch_hard":   {"difficulty": 3, "max_steps": 2},
}

DEFAULT_TASK = "patch_easy"

_SEVERITY_WEIGHT = {
    Severity.CRITICAL: 4.0, Severity.HIGH: 3.0, Severity.MEDIUM: 2.0,
    Severity.LOW: 1.0, Severity.NEGLIGIBLE: 0.5, Severity.UNKNOWN: 1.0,
}

_BP_LABELS = {
    "non_root_user":     "Non-root USER instruction",
    "healthcheck":       "HEALTHCHECK instruction defined",
    "no_secrets_in_env": "No secrets in ENV/ARG instructions",
    "apt_cache_cleanup": "APT cache cleaned (rm -rf /var/lib/apt/lists/*)",
    "pip_no_cache":      "pip --no-cache-dir flag used",
    "copy_over_add":     "COPY used instead of ADD",
    "modern_base_image": "Modern base image (Python >= 3.12)",
    "layer_efficiency":  "Consecutive RUN commands merged (max 2)",
    "copy_order":        "COPY requirements.txt before COPY . . (cache efficiency)",
    "minimal_packages":  "apt-get install uses --no-install-recommends",
}


def _format_vuln_summary(
    report: VulnReport, bp_results: dict,
    ap_warnings: List[str], difficulty: int,
) -> str:
    lines = [
        "=== Security Scan Report ===",
        f"Vulnerabilities found: {report.total_count}",
        f"  CRITICAL: {report.critical_count}  |  HIGH: {report.high_count}",
        "",
    ]

    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"]
    pip_pkgs = {
        "setuptools", "requests", "pyyaml", "urllib3", "certifi",
        "pillow", "cryptography", "flask", "jinja2", "werkzeug",
        "numpy", "scipy", "pandas",
    }

    pip_vulns = []
    for v in sorted(
        report.vulnerabilities,
        key=lambda x: severity_order.index(x.severity.value)
        if x.severity.value in severity_order else 99,
    ):
        is_pip = v.package_name.lower() in pip_pkgs
        is_regression = any(
            v.cve_id.startswith(p) for p in ("REGRESSION-", "CONFLICT-", "BUILD-FAIL-")
        )

        if is_regression:
            lines.append(f"  [REGRESSION] {v.cve_id} | {v.package_name}")
            lines.append(f"    {v.description}")
            continue

        if is_pip and difficulty >= 2:
            fix = f" -> fix: pip install --upgrade {v.package_name}>={v.fixed_version}"
            pip_vulns.append(v)
        elif v.fixed_version:
            fix = f" -> fix: {v.fixed_version}"
        else:
            fix = " (no fix available)"

        lines.append(
            f"  [{v.severity.value}] {v.cve_id} | "
            f"{v.package_name} {v.installed_version}{fix}"
        )
        lines.append(f"    {v.description}")

    if pip_vulns:
        pip_cmd_parts = [
            f"{v.package_name}>={v.fixed_version}"
            for v in pip_vulns if v.fixed_version
        ]
        if pip_cmd_parts:
            lines += [
                "",
                "=== Python Dependency Fix ===",
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


def _format_analysis_feedback(feedback: dict) -> str:
    """Format analysis grader feedback into human-readable text for the agent."""
    lines = ["", "=== Analysis Feedback ==="]

    if feedback.get("issues_confirmed"):
        lines.append("Issues correctly identified:")
        for i in feedback["issues_confirmed"]:
            lines.append(f"  [CONFIRMED] {i}")

    if feedback.get("issues_missed"):
        lines.append("Issues you missed:")
        for i in feedback["issues_missed"]:
            cat = ISSUE_CATEGORY_MAP.get(i, "other")
            lines.append(f"  [MISSED] Category: {cat} — look more carefully")

    if feedback.get("false_positives"):
        lines.append("False positives (not actual issues):")
        for fp in feedback["false_positives"]:
            lines.append(f"  [REJECTED] {fp}")

    if feedback.get("categories_confirmed"):
        lines.append(f"Categories correct: {', '.join(feedback['categories_confirmed'])}")
    if feedback.get("categories_missed"):
        lines.append(f"Categories missed: {', '.join(feedback['categories_missed'])}")

    lines.append(
        f"Detection accuracy: issue={feedback.get('issue_detection_score', 0):.3f}, "
        f"category={feedback.get('category_detection_score', 0):.3f}"
    )
    return "\n".join(lines)


def _compute_security_score(vuln_ratio_fixed: float, bp_results: dict, ap_count: int) -> float:
    bp_count = sum(1 for v in bp_results.values() if v)
    bp_total = len(bp_results) or 1
    return min(
        100.0,
        50.0 * vuln_ratio_fixed
        + 30.0 * (bp_count / bp_total)
        + 20.0 * max(0.0, 1.0 - ap_count * 0.2),
    )


def _weighted_vuln_score(vulns: List) -> float:
    return sum(_SEVERITY_WEIGHT.get(v.severity, 1.0) for v in vulns)


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
    """
    OpenEnv environment for Docker security hardening.

    2-step episode design:
      Step 1 — Analyze: Agent identifies which security issues are present.
               Returns analysis score + actionable feedback.
      Step 2 — Patch:   Agent submits a patched Dockerfile.
               Returns final improvement score.

    Each reset() generates a unique broken Dockerfile from a randomized
    pool of security issues, preventing memorization.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_name: str = ""
        self._difficulty: int = 1
        self._max_steps: int = 2
        self._image_tag: str = ""
        self._original_dockerfile: str = ""

        self._state = DockerHardeningState(episode_id=str(uuid4()), step_count=0)
        self._build_mgr: Optional[DockerBuildManager] = None
        self._initial_report: Optional[VulnReport] = None
        self._last_report: Optional[VulnReport] = None
        self._cumulative_reward: float = 0.0

        self._prev_bp: dict = {}
        self._prev_ap: List[str] = []
        self._initial_bp: dict = {}
        self._initial_ap: List[str] = []

        self._episode_seed: int = 0
        self._active_issues: Set[str] = set()
        self._active_vuln_pool: list = []
        self._analysis_completed: bool = False
        self._analysis_feedback: str = ""
        self._analysis_reward: float = 0.0

        self._analysis_grader = AnalysisGrader()

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

        episode_id = str(uuid4())
        self._episode_seed = random.randint(0, 2**31)
        rng = random.Random(self._episode_seed)

        self._active_issues = select_issues(task_name, rng)
        dockerfile, image_tag, metadata = generate_dockerfile(
            task_name, self._active_issues, rng,
        )

        self._image_tag = image_tag
        self._original_dockerfile = dockerfile

        self._state = DockerHardeningState(
            episode_id=episode_id, step_count=0, task_name=task_name,
        )
        self._cumulative_reward = 0.0
        self._analysis_completed = False
        self._analysis_feedback = ""
        self._analysis_reward = 0.0

        self._build_mgr = DockerBuildManager(
            base_image_tag=image_tag, episode_id=episode_id, use_mock=True,
        )
        self._build_mgr.initialise(dockerfile)

        self._active_vuln_pool = select_vuln_subset(self._difficulty, rng)

        initial_report = scan_mock(
            image_tag, difficulty=self._difficulty,
            vuln_pool=self._active_vuln_pool,
        )
        self._initial_report = initial_report
        self._last_report = initial_report

        bp = check_best_practices(dockerfile)
        ap = detect_antipatterns(dockerfile)
        self._initial_bp = bp
        self._initial_ap = ap
        self._prev_bp = bp
        self._prev_ap = ap

        vuln_summary = _format_vuln_summary(initial_report, bp, ap, self._difficulty)
        sec_score = _compute_security_score(0.0, bp, len(ap))

        return DockerHardeningObservation(
            current_dockerfile=dockerfile,
            vulnerability_summary=vuln_summary,
            task_name=task_name,
            initial_vuln_count=initial_report.total_count,
            current_vuln_count=initial_report.total_count,
            step_number=0,
            max_steps=self._max_steps,
            score=0.01,
            done=False,
            reward=0.0,
            step_summary=(
                f"Task '{task_name}' started (seed={self._episode_seed}). "
                f"Step 1: Identify security issues. "
                f"Step 2: Submit patched Dockerfile. "
                f"Found {initial_report.total_count} vulnerabilities in {image_tag}."
            ),
            security_score=round(sec_score, 1),
            best_practices=_format_bp_list(bp),
            antipattern_warnings=ap,
        )

    def step(self, action: DockerHardeningAction) -> DockerHardeningObservation:
        self._state.step_count += 1
        step_num = self._state.step_count

        is_analysis = self._detect_analysis(action)

        if is_analysis and not self._analysis_completed:
            return self._handle_analysis(action, step_num)
        else:
            return self._handle_patch(action, step_num)

    def _detect_analysis(self, action: DockerHardeningAction) -> bool:
        """Auto-detect whether the action is an analysis or a patch."""
        content = action.patched_dockerfile.strip()

        if content.startswith("{") or content.startswith("["):
            return True

        analysis_keywords = [
            "identified_issues", "identified_categories",
            "suspicious", "analysis", "findings",
        ]
        content_lower = content.lower()
        if any(kw in content_lower for kw in analysis_keywords):
            if "FROM " not in content.upper():
                return True

        return False

    def _handle_analysis(
        self, action: DockerHardeningAction, step_num: int,
    ) -> DockerHardeningObservation:
        """Step 1: Score the agent's issue identification, return feedback."""
        import json

        raw = action.patched_dockerfile.strip()
        try:
            agent_claims = json.loads(raw)
        except json.JSONDecodeError:
            agent_claims = {
                "identified_issues": [raw],
                "identified_categories": [],
            }

        if isinstance(agent_claims, list):
            agent_claims = {
                "identified_issues": agent_claims,
                "identified_categories": [],
            }

        analysis_reward, feedback = self._analysis_grader.score(
            agent_claims, self._active_issues,
        )

        self._analysis_completed = True
        self._analysis_reward = analysis_reward
        feedback_text = _format_analysis_feedback(feedback)
        self._analysis_feedback = feedback_text
        self._cumulative_reward += analysis_reward

        vuln_summary = _format_vuln_summary(
            self._last_report, self._prev_bp, self._prev_ap, self._difficulty,
        )
        vuln_summary += feedback_text

        return DockerHardeningObservation(
            current_dockerfile=self._original_dockerfile,
            vulnerability_summary=vuln_summary,
            task_name=self._task_name,
            initial_vuln_count=self._initial_report.total_count if self._initial_report else 0,
            current_vuln_count=self._last_report.total_count if self._last_report else 0,
            step_number=step_num,
            max_steps=self._max_steps,
            score=max(0.01, min(0.99, analysis_reward)),
            done=False,
            reward=round(analysis_reward, 4),
            step_summary=(
                f"Analysis step: identified {len(feedback.get('issues_confirmed', []))} "
                f"of {len(self._active_issues)} issues correctly. "
                f"Analysis score: {analysis_reward:.3f}. "
                f"Now submit your patched Dockerfile."
            ),
            security_score=round(
                _compute_security_score(0.0, self._prev_bp, len(self._prev_ap)), 1,
            ),
            best_practices=_format_bp_list(self._prev_bp),
            antipattern_warnings=self._prev_ap,
        )

    def _handle_patch(
        self, action: DockerHardeningAction, step_num: int,
    ) -> DockerHardeningObservation:
        """Step 2: Score the patched Dockerfile."""
        patched_df = action.patched_dockerfile

        error = _validate_dockerfile(patched_df)
        if error:
            return self._make_obs(
                reward=0.01, done=True,
                reason=TerminationReason.PATCH_FAILED_TOO_MANY_TIMES,
                step_summary=f"Invalid Dockerfile: {error}", error=error,
            )

        patched_df = _strip_markdown_fences(patched_df)

        if patched_df.strip() == self._build_mgr.current_dockerfile.strip():
            return self._make_obs(
                reward=0.01, done=True,
                reason=TerminationReason.MAX_CYCLES_REACHED,
                step_summary="Dockerfile unchanged — no improvement.",
            )

        try:
            new_tag, _ = self._build_mgr.apply_patch(patched_df)
        except DockerBuildError as e:
            return self._make_obs(
                reward=0.01, done=True,
                reason=TerminationReason.PATCH_FAILED_TOO_MANY_TIMES,
                step_summary=f"Build failed: {str(e)[:200]}",
                error=str(e)[:200],
            )

        grader_cls = GRADER_MAP.get(self._task_name)
        if grader_cls:
            grader = grader_cls()
            patch_score, breakdown = grader.score(
                initial_dockerfile=self._original_dockerfile,
                patched_dockerfile=patched_df,
                image_tag=self._image_tag,
                active_issues=self._active_issues,
                vuln_pool=self._active_vuln_pool,
            )
        else:
            patch_score = 0.01
            breakdown = {}

        new_report = scan_mock(
            new_tag, difficulty=self._difficulty,
            current_dockerfile=patched_df,
            vuln_pool=self._active_vuln_pool,
        )
        self._last_report = new_report

        new_bp = check_best_practices(patched_df)
        new_ap = detect_antipatterns(patched_df)
        self._prev_bp = new_bp
        self._prev_ap = new_ap

        if self._analysis_completed:
            final_score = 0.3 * self._analysis_reward + 0.7 * patch_score
        else:
            final_score = patch_score

        final_score = max(0.01, min(0.99, final_score))
        self._cumulative_reward += patch_score

        initial_count = self._initial_report.total_count if self._initial_report else 0
        curr_bp_sat = sum(1 for v in new_bp.values() if v)

        step_summary = (
            f"Patch step: {initial_count} -> {new_report.total_count} vulns. "
            f"Best practices: {curr_bp_sat}/{len(new_bp)}. "
            f"Anti-patterns: {len(new_ap)}. "
            f"Patch score: {patch_score:.3f}"
        )
        if self._analysis_completed:
            step_summary += f", Analysis score: {self._analysis_reward:.3f}"
        step_summary += f", Final score: {final_score:.3f}"

        return self._make_obs(
            reward=patch_score,
            done=True,
            reason=TerminationReason.ALL_VULNS_FIXED
            if new_report.total_count == 0
            else TerminationReason.MAX_CYCLES_REACHED,
            step_summary=step_summary,
            report=new_report,
            score=final_score,
            bp_results=new_bp,
            ap_warnings=new_ap,
        )

    @property
    def state(self) -> DockerHardeningState:
        return DockerHardeningState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_name=self._task_name,
            cumulative_reward=self._cumulative_reward,
            cycle_count=self._state.step_count,
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
    ) -> DockerHardeningObservation:
        rpt = report or self._last_report
        initial_vulns = self._initial_report.vulnerabilities if self._initial_report else []
        current_vulns = rpt.vulnerabilities if rpt else initial_vulns
        current_count = len(current_vulns)

        bp = bp_results or self._prev_bp
        ap = ap_warnings if ap_warnings is not None else self._prev_ap

        if score is None:
            score = 0.01

        initial_count = len(initial_vulns)
        vuln_ratio = max(0.0, (initial_count - current_count) / max(initial_count, 1))
        sec_score = _compute_security_score(vuln_ratio, bp, len(ap))

        vuln_summary = _format_vuln_summary(rpt, bp, ap, self._difficulty) if rpt else ""

        if self._analysis_feedback and not done:
            vuln_summary += self._analysis_feedback

        return DockerHardeningObservation(
            current_dockerfile=self._build_mgr.current_dockerfile if self._build_mgr else "",
            vulnerability_summary=vuln_summary,
            task_name=self._task_name,
            initial_vuln_count=initial_count,
            current_vuln_count=current_count,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            score=max(0.01, min(0.99, score)),
            step_summary=step_summary,
            last_action_error=error,
            termination_reason=reason,
            done=done,
            reward=round(max(0.01, min(0.99, reward)), 4),
            security_score=round(sec_score, 1),
            best_practices=_format_bp_list(bp),
            antipattern_warnings=ap,
        )
