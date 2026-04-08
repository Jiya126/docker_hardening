"""Legacy shaped-reward helpers (severity-weighted, cycle-penalised)."""

from typing import Dict, Optional

from models import VulnReport, PatchAttempt, Severity


SEVERITY_WEIGHTS: Dict[str, float] = {
    Severity.CRITICAL.value: 10.0,
    Severity.HIGH.value:      5.0,
    Severity.MEDIUM.value:    2.0,
    Severity.LOW.value:       0.5,
    Severity.NEGLIGIBLE.value: 0.1,
    Severity.UNKNOWN.value:    0.2,
}

CYCLE_PENALTY:        float = -0.5
CLEAN_CRITICAL_BONUS: float = 20.0
CLEAN_HIGH_BONUS:     float = 10.0
FAILED_PATCH_PENALTY: float = -2.0
BEST_PRACTICE_BONUS:  float = 3.0


def compute_step_reward(
    report_before: Optional[VulnReport],
    report_after: VulnReport,
    patch_attempt: Optional[PatchAttempt],
    cycle_number: int,
) -> float:
    reward = 0.0

    if patch_attempt is not None and not patch_attempt.success:
        return FAILED_PATCH_PENALTY

    if report_before is not None:
        before_s = report_before.summary()
        after_s = report_after.summary()
        for sev, weight in SEVERITY_WEIGHTS.items():
            delta = before_s.get(sev, 0) - after_s.get(sev, 0)
            reward += delta * weight

    if report_after.critical_count == 0 and (
        report_before is None or report_before.critical_count > 0
    ):
        reward += CLEAN_CRITICAL_BONUS

    if report_after.high_count == 0 and (
        report_before is None or report_before.high_count > 0
    ):
        reward += CLEAN_HIGH_BONUS

    if cycle_number > 1:
        reward += CYCLE_PENALTY * (cycle_number - 1)

    if patch_attempt is not None and patch_attempt.patch_strategy is not None:
        diff = patch_attempt.dockerfile_diff.lower()
        if "user " in diff or "useradd" in diff or "non-root" in diff:
            reward += BEST_PRACTICE_BONUS
        if "scratch" in diff or "distroless" in diff or "alpine" in diff:
            reward += BEST_PRACTICE_BONUS

    return round(reward, 4)


def compute_terminal_reward(
    initial_report: VulnReport,
    final_report: VulnReport,
    total_cycles: int,
    max_cycles: int,
) -> float:
    reward = 0.0

    if final_report.critical_count == 0 and final_report.high_count == 0:
        efficiency = 1.0 - (total_cycles / max_cycles)
        reward += 50.0 * (1 + efficiency)
    else:
        initial_w = sum(initial_report.summary().get(s, 0) * w
                        for s, w in SEVERITY_WEIGHTS.items())
        final_w = sum(final_report.summary().get(s, 0) * w
                      for s, w in SEVERITY_WEIGHTS.items())
        if initial_w > 0:
            reward += 20.0 * ((initial_w - final_w) / initial_w)

    if total_cycles >= max_cycles:
        reward -= 5.0

    return round(reward, 4)


def build_severity_delta(
    before: Optional[VulnReport],
    after: VulnReport,
) -> Dict[str, int]:
    if before is None:
        return {}
    before_s, after_s = before.summary(), after.summary()
    return {sev: after_s.get(sev, 0) - before_s.get(sev, 0) for sev in SEVERITY_WEIGHTS}


def compute_normalized_reward(
    initial_report: VulnReport,
    current_report: VulnReport,
) -> float:
    initial = initial_report.total_count
    if initial == 0:
        return 1.0
    fixed = initial - current_report.total_count
    return max(0.0, min(1.0, fixed / initial))
