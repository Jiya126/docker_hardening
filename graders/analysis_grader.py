"""
Analysis grader — scores the agent's ability to IDENTIFY issues before fixing.

Computes a weighted detection accuracy between agent's claimed issues
and ground truth using precision-recall on both issues and categories.
Returns (score, feedback_dict) so the env can produce actionable feedback.
"""

import re
from typing import Dict, List, Set, Tuple

from tasks.generators import ISSUE_CATEGORY_MAP, ALL_CATEGORIES

ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "pipe_to_shell":       ["curl|bash", "pipe to shell", "pipe-to-shell", "curl.*|.*sh", "supply chain", "remote script"],
    "no_user":             ["non-root", "user instruction", "running as root", "root user", "USER"],
    "no_healthcheck":      ["healthcheck", "health check", "health_check"],
    "apt_no_cleanup":      ["apt.*cache", "apt.*clean", "/var/lib/apt", "layer bloat", "apt-get clean"],
    "pip_no_cache":        ["--no-cache-dir", "pip cache", "pip.*cache"],
    "no_install_recommends": ["--no-install-recommends", "install recommends", "minimal packages"],
    "consecutive_runs":    ["consecutive run", "merge run", "layer", "multiple run"],
    "exposed_cache_port":  ["6379", "redis", "cache port"],
    "old_base_image":      ["old base", "outdated base", "python 3.9", "python 3.10", "upgrade base", "modern base"],
    "secret_env_password": ["password", "secret.*env", "env.*secret", "DATABASE_PASSWORD", "credentials"],
    "secret_env_token":    ["token", "api.*key", "API_TOKEN", "secret.*token"],
    "add_instead_of_copy": ["ADD.*instead", "use copy", "ADD instruction"],
    "healthcheck_cmd_true": ["healthcheck.*true", "CMD true", "useless healthcheck", "noop healthcheck"],
    "exposed_db_port":     ["5432", "3306", "database port", "postgres", "mysql"],
    "add_from_url":        ["ADD.*http", "ADD.*url", "add from url"],
    "get_docker_com":      ["docker-in-docker", "get.docker.com", "docker.*install"],
    "cred_in_url":         ["credential.*url", "user:pass", "password.*url", "embedded credential"],
    "base64_secret":       ["base64", "encoded secret", "AUTH_TOKEN"],
    "copy_order_bad":      ["copy order", "cache invalidat", "COPY . . before", "layer cache"],
    "chmod_777":           ["chmod 777", "overly permissive", "777 permission"],
    "multiple_secrets":    ["multiple secret", "flask.*secret", "deploy.*token"],
}


class AnalysisGrader:
    """Scores the analysis step (issue identification) and generates feedback."""

    def score(
        self,
        agent_claims: Dict,
        active_issues: Set[str],
    ) -> Tuple[float, Dict]:
        """
        Compare agent's identified issues against ground truth.

        agent_claims should contain:
          - identified_issues: list[str] — free-text descriptions
          - identified_categories: list[str] — category names

        Returns (reward, feedback_dict).
        """
        claimed_texts = agent_claims.get("identified_issues", [])
        claimed_cats = set(agent_claims.get("identified_categories", []))

        matched, unmatched = self._match_issues(claimed_texts, active_issues)

        true_cats = {ISSUE_CATEGORY_MAP.get(i, "other") for i in active_issues}
        cat_tp = len(claimed_cats & true_cats)
        cat_fp = len(claimed_cats - true_cats)
        cat_fn = len(true_cats - claimed_cats)

        issue_precision = len(matched) / max(len(matched) + len(unmatched), 1)
        issue_recall = len(matched) / max(len(active_issues), 1)
        issue_accuracy = (
            2 * issue_precision * issue_recall / max(issue_precision + issue_recall, 1e-9)
        )

        cat_precision = cat_tp / max(cat_tp + cat_fp, 1)
        cat_recall = cat_tp / max(cat_tp + cat_fn, 1)
        cat_accuracy = 2 * cat_precision * cat_recall / max(cat_precision + cat_recall, 1e-9)

        raw_score = 0.6 * issue_accuracy + 0.4 * cat_accuracy
        reward = max(0.01, min(0.99, raw_score))

        feedback = {
            "issues_confirmed": sorted(matched),
            "issues_missed": sorted(active_issues - matched),
            "false_positives": unmatched,
            "categories_confirmed": sorted(true_cats & claimed_cats),
            "categories_missed": sorted(true_cats - claimed_cats),
            "categories_rejected": sorted(claimed_cats - true_cats),
            "issue_detection_score": round(issue_accuracy, 3),
            "category_detection_score": round(cat_accuracy, 3),
        }

        return reward, feedback

    def _match_issues(
        self, claimed_texts: List[str], active_issues: Set[str],
    ) -> Tuple[Set[str], List[str]]:
        """Match free-text claims against known issues via keyword lookup."""
        matched = set()
        unmatched = []

        for text in claimed_texts:
            text_lower = text.lower()
            found = False
            for issue_id in active_issues:
                if issue_id in matched:
                    continue
                keywords = ISSUE_KEYWORDS.get(issue_id, [])
                for kw in keywords:
                    if re.search(kw, text_lower):
                        if not self._is_negated(text_lower, kw):
                            matched.add(issue_id)
                            found = True
                            break
                if found:
                    break
            if not found:
                unmatched.append(text)

        return matched, unmatched

    @staticmethod
    def _is_negated(text: str, keyword: str) -> bool:
        """Reject claims where the agent dismisses the issue."""
        negation_terms = [
            "not an issue", "not present", "not found", "does not",
            "no evidence", "is fine", "is ok", "not applicable",
            "doesn't apply", "not relevant",
        ]
        for neg in negation_terms:
            if neg in text:
                return True
        return False
