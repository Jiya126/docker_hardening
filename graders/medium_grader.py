"""Medium task grader — scores Dockerfile patches for difficulty 2."""

from typing import Dict, Set, Tuple

from graders.easy_grader import EasyGrader


class MediumGrader(EasyGrader):
    """
    Medium-level grader.
    Same structure as Easy but stricter weights and secret-leak penalty.
    """

    W_VULN = 0.40
    W_BP = 0.35
    W_AP = 0.25

    DIFFICULTY = 2

    def score(
        self,
        initial_dockerfile: str,
        patched_dockerfile: str,
        image_tag: str,
        active_issues: Set[str],
        vuln_pool: list = None,
    ) -> Tuple[float, Dict]:
        from tools.scanner import (
            scan_mock, check_best_practices, detect_antipatterns,
        )

        initial_report = scan_mock(image_tag, difficulty=self.DIFFICULTY, vuln_pool=vuln_pool)
        patched_report = scan_mock(
            image_tag, difficulty=self.DIFFICULTY,
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

        if not patch_bp.get("no_secrets_in_env", True):
            raw -= 0.10

        score = max(0.01, min(0.99, raw))

        breakdown = {
            "vuln_improvement": round(vuln_score, 3),
            "bp_improvement": round(bp_score, 3),
            "ap_improvement": round(ap_score, 3),
            "false_positives": fp_count,
            "secrets_remaining": not patch_bp.get("no_secrets_in_env", True),
            "vulns_initial": len(initial_report.vulnerabilities),
            "vulns_remaining": len(patched_report.vulnerabilities),
            "bp_satisfied": sum(1 for v in patch_bp.values() if v),
            "bp_total": len(patch_bp),
            "ap_remaining": len(patch_ap),
        }

        return score, breakdown
