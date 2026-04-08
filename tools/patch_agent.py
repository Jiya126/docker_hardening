"""LLM-based patch agent (Claude / GPT) with rule-based mock fallback."""

import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import VulnReport, PatchAttempt, PatchStrategy, Severity


SYSTEM_PROMPT = """\
You are a Docker security hardening expert.
Your job is to analyse vulnerability reports for Docker images and produce
minimal, correct Dockerfile patches to eliminate as many vulnerabilities as
possible, prioritising CRITICAL and HIGH severity issues.

You MUST respond with a single JSON object with these keys:
{
  "strategy": "<one of: upgrade_package | replace_base_image | remove_package | add_security_config | multi_step>",
  "reasoning": "<brief explanation of your approach>",
  "patched_dockerfile": "<full content of the patched Dockerfile>",
  "changes_summary": "<one-line description of what changed>"
}

Rules:
- Do NOT add comments explaining your changes inside the Dockerfile.
- Prefer upgrading the base image tag or using a more minimal base (alpine, distroless).
- Always use a pinned version for base images, never 'latest'.
- Add 'USER nonroot' or equivalent where feasible.
- If a package has no fix available, remove it if non-essential.
- Do NOT invent or add image digests (@sha256:...) unless the original already has one.
- Do NOT edit, append to, or rewrite requirements.txt or any other file from inside the Dockerfile.
- Do NOT add pip install --require-hashes unless the original already uses it.
- Preserve existing build-context assumptions; don't add new COPY/ADD source paths.
- The patched Dockerfile must be syntactically valid.
- Output ONLY the JSON object, no markdown fences, no preamble.

apt-get rules:
- Do NOT pin exact Debian package versions (e.g. curl=7.88.1-10).
- Use unversioned apt-get install -y --no-install-recommends <package> or apt-get upgrade -y.
- To fix system CVEs, prefer upgrading the base image tag.
- Do NOT apt-get remove packages from a different ecosystem (e.g. log4j is a Java JAR).
- Do NOT install new language runtimes just to address a reported vulnerability.\
"""

SUPPORTED_PROVIDERS = {"claude", "gpt", "openai"}
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "gpt":    "gpt-4.1",
}
API_KEY_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "gpt":    "OPENAI_API_KEY",
}


def normalise_provider(provider: str) -> str:
    normalised = provider.lower().strip()
    if normalised == "openai":
        normalised = "gpt"
    if normalised not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported patch agent provider '{provider}'. Supported: {supported}")
    return normalised


def _build_user_prompt(
    dockerfile: str,
    report: VulnReport,
    patch_history: list,
    cycle: int,
) -> str:
    vuln_lines = []
    for v in sorted(report.vulnerabilities,
                    key=lambda x: list(Severity).index(x.severity)):
        fix = f" -> fix: {v.fixed_version}" if v.fixed_version else " (no fix available)"
        vuln_lines.append(
            f"  [{v.severity.value}] {v.cve_id} | {v.package_name} {v.installed_version}{fix}"
        )

    vuln_text = "\n".join(vuln_lines[:40])
    if len(report.vulnerabilities) > 40:
        vuln_text += f"\n  ... and {len(report.vulnerabilities) - 40} more"

    history_text = ""
    if patch_history:
        history_lines = []
        for p in patch_history[-3:]:
            status = "OK" if p.success else "FAIL"
            history_lines.append(
                f"  Cycle {p.cycle} [{status}] {p.patch_strategy.value}: "
                f"{p.dockerfile_diff[:200]}"
            )
        history_text = ("\nPrevious patch attempts (avoid repeating failures):\n"
                        + "\n".join(history_lines))

    return f"""\
=== CYCLE {cycle} ===

Current Dockerfile:
```
{dockerfile}
```

Vulnerability Report ({report.total_count} total):
  CRITICAL: {report.critical_count} | HIGH: {report.high_count}
{vuln_text}
{history_text}

Produce a patched Dockerfile that addresses the highest-severity issues.

Important constraints:
- Only modify the Dockerfile itself.
- Assume requirements files may contain editable installs or constraints; do not append to or rewrite them.
- If you change the base image tag, use a normal tag only. Do not add a new digest pin.
"""


def _sanitize_patched_dockerfile(patched: str, original: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    sanitized = patched

    if "@sha256:" not in original:
        sanitized = re.sub(r"(@sha256:[0-9a-fA-F]{64})", "", sanitized)

    lower = sanitized.lower()
    forbidden = (
        ">> requirements.txt", "> requirements.txt",
        "tee requirements.txt", "tee -a requirements.txt", "sed -i",
    )
    if "requirements.txt" in lower and any(t in lower for t in forbidden):
        errors.append("Patched Dockerfile attempts to modify requirements.txt inside the image build")

    if "--require-hashes" not in original:
        sanitized = sanitized.replace("--require-hashes ", "").replace(" --require-hashes", "")

    return sanitized, errors


class LLMPatchAgent:
    """Calls an LLM to generate patches; falls back to rule-based mock."""

    def __init__(
        self,
        provider: str = "claude",
        model: str | None = None,
        max_tokens: int = 2048,
        use_mock: bool = False,
    ):
        self.provider = normalise_provider(provider)
        self.model = model or DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self.api_key_env_var = API_KEY_ENV_VARS[self.provider]
        self.use_mock = use_mock or not os.environ.get(self.api_key_env_var)

    def patch(
        self,
        dockerfile: str,
        report: VulnReport,
        patch_history: list,
        cycle: int,
    ) -> PatchAttempt:
        if self.use_mock:
            return self._mock_patch(dockerfile, report, cycle)

        user_prompt = _build_user_prompt(dockerfile, report, patch_history, cycle)
        try:
            raw = self._generate_response(user_prompt)
        except Exception as e:
            return PatchAttempt(
                cycle=cycle, patch_strategy=PatchStrategy.UPGRADE_PACKAGE,
                dockerfile_diff="", patched_dockerfile=None, success=False,
                provider=self.provider, model=self.model, error_message=str(e),
            )

        return self._parse_response(raw, dockerfile, cycle)

    def _generate_response(self, user_prompt: str) -> str:
        if self.provider == "claude":
            return self._generate_with_claude(user_prompt)
        if self.provider == "gpt":
            return self._generate_with_gpt(user_prompt)
        raise ValueError(f"Unhandled provider '{self.provider}'")

    def _generate_with_claude(self, user_prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()

    def _generate_with_gpt(self, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty completion")
        return content.strip()

    def _parse_response(
        self, raw: str, original_dockerfile: str, cycle: int,
    ) -> PatchAttempt:
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return PatchAttempt(
                cycle=cycle, patch_strategy=PatchStrategy.UPGRADE_PACKAGE,
                dockerfile_diff="", patched_dockerfile=None, success=False,
                provider=self.provider, model=self.model,
                error_message=f"JSON parse error: {e}\nRaw: {raw[:300]}",
            )

        patched = data.get("patched_dockerfile", "")
        if not patched:
            return PatchAttempt(
                cycle=cycle, patch_strategy=PatchStrategy.UPGRADE_PACKAGE,
                dockerfile_diff="", patched_dockerfile=None, success=False,
                provider=self.provider, model=self.model,
                error_message="No patched_dockerfile in response",
            )

        patched, validation_errors = _sanitize_patched_dockerfile(patched, original_dockerfile)
        if validation_errors:
            return PatchAttempt(
                cycle=cycle, patch_strategy=PatchStrategy.UPGRADE_PACKAGE,
                dockerfile_diff="", patched_dockerfile=patched, success=False,
                provider=self.provider, model=self.model,
                error_message="; ".join(validation_errors),
                agent_reasoning=data.get("reasoning"),
            )

        diff = "\n".join(difflib.unified_diff(
            original_dockerfile.splitlines(), patched.splitlines(),
            fromfile="Dockerfile.before", tofile="Dockerfile.after", lineterm="",
        ))

        strategy_map = {s.value: s for s in PatchStrategy}
        strategy = strategy_map.get(data.get("strategy", ""), PatchStrategy.UPGRADE_PACKAGE)

        return PatchAttempt(
            cycle=cycle, patch_strategy=strategy,
            dockerfile_diff=diff, patched_dockerfile=patched, success=True,
            provider=self.provider, model=self.model,
            agent_reasoning=data.get("reasoning"),
        )

    def get_patched_dockerfile(self, attempt: PatchAttempt, original: str) -> str:
        if not attempt.dockerfile_diff:
            return original
        try:
            result = subprocess.run(
                ["patch", "-p0", "--output=-"],
                input="\n".join([original, attempt.dockerfile_diff]),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return original

    def _mock_patch(
        self, dockerfile: str, report: VulnReport, cycle: int,
    ) -> PatchAttempt:
        lines = dockerfile.splitlines()
        patched_lines = list(lines)

        strategy = PatchStrategy.UPGRADE_PACKAGE
        reasoning = "Mock patcher: upgrading packages with known fixes"
        changed = False

        if report.critical_count > 0:
            for i, line in enumerate(patched_lines):
                if line.strip().upper().startswith("FROM"):
                    parts = line.split()
                    if len(parts) >= 2 and ":" in parts[1]:
                        base, tag = parts[1].rsplit(":", 1)
                        new_tag = _bump_mock_tag(tag)
                        patched_lines[i] = f"FROM {base}:{new_tag}"
                        strategy = PatchStrategy.REPLACE_BASE_IMAGE
                        reasoning = f"Mock: bumped base image tag {tag} -> {new_tag}"
                        changed = True
                        break

        has_user = any("USER " in l.upper() for l in patched_lines)
        if not has_user:
            insert_at = len(patched_lines)
            for i in range(len(patched_lines) - 1, -1, -1):
                if patched_lines[i].strip().upper().startswith(("CMD", "ENTRYPOINT")):
                    insert_at = i
                    break
            patched_lines.insert(insert_at, "RUN addgroup -S appgroup && adduser -S appuser -G appgroup")
            patched_lines.insert(insert_at + 1, "USER appuser")
            strategy = PatchStrategy.ADD_SECURITY_CONFIG
            reasoning += " + added non-root user"
            changed = True

        patched = "\n".join(patched_lines)
        diff = "\n".join(difflib.unified_diff(
            dockerfile.splitlines(), patched.splitlines(),
            fromfile="Dockerfile.before", tofile="Dockerfile.after", lineterm="",
        ))

        return PatchAttempt(
            cycle=cycle, patch_strategy=strategy,
            dockerfile_diff=diff, patched_dockerfile=patched,
            success=changed, provider=self.provider, model=self.model,
            error_message=None if changed else "Mock patcher: nothing to change",
            agent_reasoning=reasoning,
        )


def _bump_mock_tag(tag: str) -> str:
    m = re.match(r"(\d+)\.(\d+)", tag)
    if m:
        return f"{int(m.group(1))}.{int(m.group(2)) + 1}"
    return tag + "-fixed"


class ClaudeCodePatchAgent(LLMPatchAgent):

    def __init__(self, model: str | None = None, max_tokens: int = 2048, use_mock: bool = False):
        super().__init__(provider="claude", model=model, max_tokens=max_tokens, use_mock=use_mock)
