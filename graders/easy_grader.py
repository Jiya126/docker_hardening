"""Easy task grader — scores Dockerfile patches for difficulty 1."""

import sys
import os
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.scanner import (
    scan_mock, check_best_practices, detect_antipatterns,
)


class EasyGrader:
    """
    Easy-level grader.
    Weights: 50% vulnerability reduction, 30% best practices, 20% anti-patterns.
    """

    W_VULN = 0.50
    W_BP = 0.30
    W_AP = 0.20

    def score(
        self,
        initial_dockerfile: str,
        patched_dockerfile: str,
        image_tag: str,
        active_issues: Set[str],
        vuln_pool: list = None,
    ) -> Tuple[float, Dict]:
        """
        Score a patched Dockerfile.
        Returns (score_0_to_1, breakdown_dict).
        """
        initial_report = scan_mock(image_tag, difficulty=1, vuln_pool=vuln_pool)
        patched_report = scan_mock(
            image_tag, difficulty=1,
            current_dockerfile=patched_dockerfile,
            vuln_pool=vuln_pool,
        )

        init_bp = check_best_practices(initial_dockerfile)
        patch_bp = check_best_practices(patched_dockerfile)

        init_ap = detect_antipatterns(initial_dockerfile)
        patch_ap = detect_antipatterns(patched_dockerfile)

        vuln_score = self._vuln_improvement(
            initial_report.vulnerabilities,
            patched_report.vulnerabilities,
        )
        bp_score = self._bp_improvement(init_bp, patch_bp)
        ap_score = self._ap_improvement(len(init_ap), len(patch_ap))

        raw = (self.W_VULN * vuln_score
               + self.W_BP * bp_score
               + self.W_AP * ap_score)

        fp_count = self._false_positive_count(patched_report.vulnerabilities)
        if fp_count > 0:
            raw -= 0.05 * fp_count

        score = max(0.01, min(0.99, raw))

        breakdown = {
            "vuln_improvement": round(vuln_score, 3),
            "bp_improvement": round(bp_score, 3),
            "ap_improvement": round(ap_score, 3),
            "false_positives": fp_count,
            "vulns_initial": len(initial_report.vulnerabilities),
            "vulns_remaining": len(patched_report.vulnerabilities),
            "bp_satisfied": sum(1 for v in patch_bp.values() if v),
            "bp_total": len(patch_bp),
            "ap_remaining": len(patch_ap),
        }

        return score, breakdown

    @staticmethod
    def _vuln_improvement(initial_vulns: list, current_vulns: list) -> float:
        WEIGHT = {
            "CRITICAL": 4.0, "HIGH": 3.0, "MEDIUM": 2.0,
            "LOW": 1.0, "NEGLIGIBLE": 0.5, "UNKNOWN": 1.0,
        }
        init_w = sum(WEIGHT.get(v.severity.value, 1.0) for v in initial_vulns)
        curr_w = sum(WEIGHT.get(v.severity.value, 1.0) for v in current_vulns)
        return max(0.0, (init_w - curr_w) / max(init_w, 1.0))

    @staticmethod
    def _bp_improvement(init_bp: dict, patch_bp: dict) -> float:
        init_sat = sum(1 for v in init_bp.values() if v)
        patch_sat = sum(1 for v in patch_bp.values() if v)
        improvable = len(patch_bp) - init_sat
        if improvable <= 0:
            return 0.0
        return max(0.0, (patch_sat - init_sat) / improvable)

    @staticmethod
    def _ap_improvement(init_count: int, patch_count: int) -> float:
        if init_count <= 0:
            return 0.0
        return max(0.0, (init_count - patch_count) / init_count)

    @staticmethod
    def _false_positive_count(vulns: list) -> int:
        return sum(
            1 for v in vulns
            if any(v.cve_id.startswith(p) for p in ("REGRESSION-", "CONFLICT-", "BUILD-FAIL-"))
        )
