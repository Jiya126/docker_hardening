"""Vulnerability scanning — mock (default), trivy, and grype backends."""

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Vulnerability, VulnReport, Severity


# ───────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────

class ScannerError(Exception):
    pass


# ───────────────────────────────────────────────────────────────────────────
# Trivy
# ───────────────────────────────────────────────────────────────────────────

_SEVERITY_MAP = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "negligible": Severity.NEGLIGIBLE,
}


def _normalise_severity(raw: str) -> Severity:
    return _SEVERITY_MAP.get(raw.lower(), Severity.UNKNOWN)


def scan_with_trivy(image_tag: str, timeout: int = 120) -> VulnReport:
    if not shutil.which("trivy"):
        raise ScannerError("trivy not found in PATH")

    cmd = ["trivy", "image", "--format", "json", "--quiet", "--no-progress", image_tag]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ScannerError(f"trivy timed out after {timeout}s scanning {image_tag}")

    if result.returncode not in (0, 1):
        raise ScannerError(f"trivy failed: {result.stderr}")

    data = json.loads(result.stdout)
    vulns: List[Vulnerability] = []
    for item in data.get("Results", []):
        for v in item.get("Vulnerabilities") or []:
            vulns.append(Vulnerability(
                cve_id=v.get("VulnerabilityID", "UNKNOWN"),
                package_name=v.get("PkgName", ""),
                installed_version=v.get("InstalledVersion", ""),
                fixed_version=v.get("FixedVersion"),
                severity=_normalise_severity(v.get("Severity", "UNKNOWN")),
                description=v.get("Description", "")[:500],
                layer=(v.get("Layer", {}) or {}).get("DiffID"),
                score=v.get("CVSS", {}).get("nvd", {}).get("V3Score", 0.0),
                cvss_vector=v.get("CVSS", {}).get("nvd", {}).get("V3Vector"),
            ))

    return VulnReport(
        image_tag=image_tag, scan_tool="trivy",
        scanned_at=datetime.datetime.utcnow().isoformat(),
        vulnerabilities=vulns,
        os_info=data.get("Metadata", {}).get("OS", {}).get("Family"),
        total_packages=len(data.get("Results", [])),
    )


# ───────────────────────────────────────────────────────────────────────────
# Grype
# ───────────────────────────────────────────────────────────────────────────

def scan_with_grype(image_tag: str, timeout: int = 120) -> VulnReport:
    if not shutil.which("grype"):
        raise ScannerError("grype not found in PATH")

    cmd = ["grype", image_tag, "-o", "json", "--quiet"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ScannerError(f"grype timed out after {timeout}s")

    if result.returncode not in (0, 1):
        raise ScannerError(f"grype failed: {result.stderr}")

    data = json.loads(result.stdout)
    vulns: List[Vulnerability] = []
    for match in data.get("matches", []):
        vd = match.get("vulnerability", {})
        art = match.get("artifact", {})
        vulns.append(Vulnerability(
            cve_id=vd.get("id", "UNKNOWN"),
            package_name=art.get("name", ""),
            installed_version=art.get("version", ""),
            fixed_version=(vd.get("fix", {}) or {}).get("versions", [None])[0],
            severity=_normalise_severity(vd.get("severity", "UNKNOWN")),
            description=vd.get("description", "")[:500],
            score=(vd.get("cvss", [{}])[0].get("metrics", {}).get("baseScore", 0.0)
                   if vd.get("cvss") else 0.0),
        ))

    return VulnReport(
        image_tag=image_tag, scan_tool="grype",
        scanned_at=datetime.datetime.utcnow().isoformat(),
        vulnerabilities=vulns,
    )


# ───────────────────────────────────────────────────────────────────────────
# Mock scanner — pre-computed CVE data
# ───────────────────────────────────────────────────────────────────────────

_CVE_DESCRIPTIONS = {
    "libssl1.1":      "Buffer overread in libssl allows remote info disclosure via crafted TLS handshake",
    "curl":           "HTTP/2 stream cancellation can cause use-after-free in curl < 7.86",
    "openssl":        "OpenSSL X.509 certificate verification buffer overflow (CVE-2022-3602)",
    "libpython3.9":   "Memory corruption in Python 3.9 ctypes foreign function interface",
    "libc6":          "glibc printf buffer overflow allows local privilege escalation",
    "tzdata":         "Outdated timezone data causes incorrect time calculations",
    "log4j":          "Apache Log4j2 JNDI injection allows remote code execution (Log4Shell)",
    "setuptools":     "Setuptools package_index regex denial-of-service vulnerability",
    "requests":       "Requests library leaks Proxy-Authorization header to redirected hosts",
    "pyyaml":         "PyYAML load() allows arbitrary code execution via crafted YAML documents",
    "pillow":         "Pillow decompression bomb denial-of-service via crafted image files",
    "cryptography":   "cryptography RSA PKCS#1 v1.5 decryption timing side-channel (Bleichenbacher)",
    "certifi":        "certifi ships outdated Mozilla CA bundle allowing MitM with revoked certificates",
    "urllib3":        "urllib3 HTTP redirect handling allows SSRF via crafted Location header",
    "flask":          "Flask debugger PIN bypass allows remote code execution in debug mode",
    "jinja2":         "Jinja2 sandbox escape allows arbitrary code execution via crafted templates",
    "werkzeug":       "Werkzeug multipart parser DoS via crafted Content-Type boundary",
    "numpy":          "numpy buffer overflow in array reshape allows heap corruption",
    "scipy":          "scipy Fortran interface memory corruption via crafted input arrays",
    "myapp-crypto":   "Internal cryptographic utility has potential timing side-channel",
    "libxml2":        "libxml2 use-after-free in xmlXPathCompOpEval allows code execution",
    "libexpat":       "Expat XML parser integer overflow in storeRawNames",
    "pandas":         "pandas eval() allows arbitrary code execution via crafted expressions",
}

_MOCK_TEMPLATES = {
    1: [
        (Severity.MEDIUM, "curl",      "7.64.0",    "7.86.0"),
        (Severity.MEDIUM, "urllib3",    "1.26.5",    "2.0.7"),
        (Severity.LOW,    "libssl1.1",  "1.1.1f",    "1.1.1n"),
        (Severity.LOW,    "tzdata",     "2020a",     "2023c"),
    ],
    2: [
        (Severity.HIGH,   "libpython3.9", "3.9.0",    "3.9.16"),
        (Severity.HIGH,   "openssl",      "1.1.1",    "3.0.7"),
        (Severity.MEDIUM, "setuptools",   "65.5.0",   "70.0.0"),
        (Severity.MEDIUM, "certifi",      "2023.7.22","2024.2.2"),
        (Severity.MEDIUM, "flask",        "2.2.0",    "2.3.3"),
        (Severity.MEDIUM, "jinja2",       "3.1.0",    "3.1.3"),
        (Severity.MEDIUM, "werkzeug",     "2.2.0",    "3.0.1"),
        (Severity.MEDIUM, "libc6",        "2.31",     "2.36"),
        (Severity.LOW,    "tzdata",       "2020a",    "2023c"),
    ],
    3: [
        (Severity.CRITICAL, "log4j",        "2.14.1",    "2.17.1"),
        (Severity.CRITICAL, "openssl",      "1.0.2k",    "3.0.7"),
        (Severity.HIGH,     "libpython3.9", "3.9.0",     "3.9.16"),
        (Severity.HIGH,     "curl",         "7.64.0",    "7.86.0"),
        (Severity.HIGH,     "requests",     "2.25.0",    "2.32.0"),
        (Severity.HIGH,     "cryptography", "38.0.0",    "42.0.0"),
        (Severity.HIGH,     "flask",        "2.2.0",     "2.3.3"),
        (Severity.MEDIUM,   "libc6",        "2.31",      "2.36"),
        (Severity.MEDIUM,   "pyyaml",       "5.4",       "6.0.1"),
        (Severity.MEDIUM,   "pillow",       "9.0.0",     "10.3.0"),
        (Severity.MEDIUM,   "certifi",      "2023.7.22", "2024.2.2"),
        (Severity.MEDIUM,   "jinja2",       "3.1.0",     "3.1.3"),
        (Severity.MEDIUM,   "werkzeug",     "2.2.0",     "3.0.1"),
        (Severity.LOW,      "tzdata",       "2020a",     "2023c"),
    ],
    # ── Multi-stage (difficulty 4) ────────────────────────────────────────
    # Build deps (gcc) must stay in builder stage; pip packages need C compiler
    4: [
        (Severity.HIGH,     "openssl",      "1.1.1",     "3.0.7"),
        (Severity.HIGH,     "cryptography", "38.0.0",    "42.0.0"),
        (Severity.HIGH,     "pillow",       "9.0.0",     "10.3.0"),
        (Severity.MEDIUM,   "setuptools",   "65.5.0",    "70.0.0"),
        (Severity.MEDIUM,   "numpy",        "1.24.0",    "1.26.4"),
        (Severity.MEDIUM,   "libc6",        "2.31",      "2.36"),
        (Severity.MEDIUM,   "certifi",      "2023.7.22", "2024.2.2"),
        (Severity.LOW,      "libexpat",     "2.4.1",     "2.6.0"),
        (Severity.LOW,      "tzdata",       "2020a",     "2023c"),
    ],
    # ── Conflicting requirements (difficulty 5) ───────────────────────────
    # cryptography>=42 needs openssl>=3.0, but libssl1.1 is pinned for legacy binary.
    # numpy C-extension needs glibc (alpine breaks it).
    # Some CVEs have no fix.
    5: [
        (Severity.CRITICAL, "openssl",      "1.0.2k",    "3.0.7"),
        (Severity.HIGH,     "cryptography", "38.0.0",    "42.0.0"),
        (Severity.HIGH,     "numpy",        "1.24.0",    "1.26.4"),
        (Severity.HIGH,     "flask",        "2.2.0",     "2.3.3"),
        (Severity.MEDIUM,   "libxml2",      "2.9.10",    None),
        (Severity.MEDIUM,   "setuptools",   "65.5.0",    "70.0.0"),
        (Severity.MEDIUM,   "pyyaml",       "5.4",       "6.0.1"),
        (Severity.MEDIUM,   "libc6",        "2.31",      None),
        (Severity.LOW,      "libexpat",     "2.4.1",     "2.6.0"),
        (Severity.LOW,      "tzdata",       "2020a",     "2023c"),
        (Severity.NEGLIGIBLE, "scipy",      "1.10.0",    "1.12.0"),
    ],
    # ── Subtle anti-patterns (difficulty 6) ───────────────────────────────
    # Secrets hidden in URLs and base64, cache-busting COPY order, curl-dependent healthcheck
    6: [
        (Severity.HIGH,     "openssl",      "1.1.1",     "3.0.7"),
        (Severity.HIGH,     "requests",     "2.25.0",    "2.32.0"),
        (Severity.HIGH,     "cryptography", "38.0.0",    "42.0.0"),
        (Severity.MEDIUM,   "flask",        "2.2.0",     "2.3.3"),
        (Severity.MEDIUM,   "certifi",      "2023.7.22", "2024.2.2"),
        (Severity.MEDIUM,   "jinja2",       "3.1.0",     "3.1.3"),
        (Severity.MEDIUM,   "pillow",       "9.0.0",     "10.3.0"),
        (Severity.MEDIUM,   "libc6",        "2.31",      "2.36"),
        (Severity.LOW,      "tzdata",       "2020a",     "2023c"),
    ],
    # ── Adversarial traps (difficulty 7) ──────────────────────────────────
    # Decoy package (myapp-crypto != cryptography), needed app port, USER root mid-build
    7: [
        (Severity.CRITICAL, "openssl",      "1.0.2k",    "3.0.7"),
        (Severity.HIGH,     "myapp-crypto", "1.2.0",     None),
        (Severity.HIGH,     "cryptography", "38.0.0",    "42.0.0"),
        (Severity.HIGH,     "requests",     "2.25.0",    "2.32.0"),
        (Severity.HIGH,     "flask",        "2.2.0",     "2.3.3"),
        (Severity.MEDIUM,   "numpy",        "1.24.0",    "1.26.4"),
        (Severity.MEDIUM,   "pandas",       "2.0.0",     "2.2.0"),
        (Severity.MEDIUM,   "setuptools",   "65.5.0",    "70.0.0"),
        (Severity.MEDIUM,   "libc6",        "2.31",      "2.36"),
        (Severity.MEDIUM,   "pyyaml",       "5.4",       "6.0.1"),
        (Severity.LOW,      "libexpat",     "2.4.1",     "2.6.0"),
        (Severity.LOW,      "tzdata",       "2020a",     "2023c"),
        (Severity.NEGLIGIBLE, "libxml2",    "2.9.10",    None),
    ],
}

_SYSTEM_PKGS = {"openssl", "curl", "libc6", "libpython3.9", "libssl1.1", "tzdata", "libxml2", "libexpat"}
_PIP_PKGS = {
    "setuptools", "requests", "pyyaml", "urllib3", "certifi",
    "pillow", "cryptography", "flask", "jinja2", "werkzeug",
    "numpy", "scipy", "pandas",
}
_C_EXTENSION_PKGS = {"numpy", "scipy", "pandas", "pillow", "cryptography"}


# ───────────────────────────────────────────────────────────────────────────
# Static base image tag validation (replaces Docker Hub registry API)
# ───────────────────────────────────────────────────────────────────────────

VALID_BASE_IMAGES: dict[str, set[str]] = {
    "python": {
        "latest",
        "3.13", "3.13-slim", "3.13-alpine", "3.13-bookworm", "3.13-slim-bookworm",
        "3.12", "3.12-slim", "3.12-alpine", "3.12-bookworm", "3.12-slim-bookworm",
        "3.11", "3.11-slim", "3.11-alpine", "3.11-bookworm", "3.11-slim-bookworm",
        "3.11-bullseye", "3.11-slim-bullseye",
        "3.10", "3.10-slim", "3.10-alpine", "3.10-bookworm", "3.10-slim-bookworm",
        "3.10-bullseye", "3.10-slim-bullseye",
        "3.9", "3.9-slim", "3.9-alpine", "3.9-bookworm", "3.9-slim-bookworm",
        "3.9-bullseye", "3.9-slim-bullseye",
        "3.8", "3.8-slim", "3.8-alpine",
        "3.7", "3.7-slim", "3.7-alpine",
        "3.6", "3.6-slim", "3.6-alpine",
    },
    "node": {
        "22", "22-slim", "22-alpine", "22-bookworm", "22-slim-bookworm",
        "20", "20-slim", "20-alpine", "20-bookworm", "20-slim-bookworm",
        "18", "18-slim", "18-alpine", "18-bookworm", "18-slim-bookworm",
        "18-bullseye", "18-slim-bullseye", "18-buster",
        "16", "16-slim", "16-alpine", "16-bullseye",
        "14", "14-slim", "14-alpine", "14-bullseye", "14-buster",
    },
    "alpine": {"3.20", "3.19", "3.18", "3.17", "3.16", "3.15", "3.14"},
    "ubuntu": {"24.04", "22.04", "20.04", "18.04", "noble", "jammy", "focal", "bionic"},
    "debian": {
        "bookworm", "bookworm-slim", "bullseye", "bullseye-slim",
        "buster", "buster-slim", "trixie", "trixie-slim",
    },
    "nginx": {
        "1.27", "1.27-alpine", "1.26", "1.26-alpine",
        "1.25", "1.25-alpine", "1.24", "1.24-alpine",
        "stable", "stable-alpine",
    },
    "golang": {
        "1.22", "1.22-alpine", "1.22-bookworm",
        "1.21", "1.21-alpine", "1.21-bookworm",
    },
    "gcr.io/distroless/python3": {"latest", "nonroot", "debug"},
    "gcr.io/distroless/base":    {"latest", "nonroot", "debug"},
}

_ALWAYS_VALID_IMAGES = {"scratch"}


def validate_base_image_tag(image_ref: str) -> str | None:
    """Returns None if valid, or an error message if the tag is unknown."""
    if image_ref in _ALWAYS_VALID_IMAGES:
        return None

    image, tag = (image_ref.rsplit(":", 1) if ":" in image_ref
                  else (image_ref, "latest"))

    if "@" in tag:
        tag = tag.split("@")[0]

    if image not in VALID_BASE_IMAGES:
        return None

    valid_tags = VALID_BASE_IMAGES[image]
    if tag in valid_tags:
        return None

    suggestions = sorted(valid_tags)[:5]
    return (
        f"Unknown tag '{tag}' for image '{image}'. "
        f"Valid tags include: {', '.join(suggestions)}"
    )


# ───────────────────────────────────────────────────────────────────────────
# Dockerfile fix simulation
# ───────────────────────────────────────────────────────────────────────────

def _strip_comments(dockerfile: str) -> str:
    return "\n".join(
        line for line in dockerfile.splitlines()
        if not line.lstrip().startswith("#")
    )


def _pip_explicitly_upgrades(pkg: str, dockerfile: str) -> bool:
    pkg_lower = pkg.lower()
    df_clean = _strip_comments(dockerfile)
    df_lower = df_clean.lower()

    for line in df_lower.splitlines():
        line_s = line.strip()
        if "pip install" not in line_s and "pip3 install" not in line_s:
            continue
        if "-r " in line_s or "-r\t" in line_s:
            continue
        if not re.search(rf"\b{re.escape(pkg_lower)}\b", line_s):
            continue
        has_upgrade = ("--upgrade" in line_s or
                       re.search(r"\s-U\b", line_s) is not None)
        has_pin = re.search(rf"\b{re.escape(pkg_lower)}\s*[><=!]", line_s) is not None
        if has_upgrade or has_pin:
            return True

    sed_pins = bool(re.search(rf"sed\b.*{re.escape(pkg_lower)}.*[><=!0-9]", df_lower))
    has_pip_r = bool(re.search(r"pip3?\s+install\b.*-r\s", df_lower))
    if sed_pins and has_pip_r:
        return True

    return False


def _simulate_dockerfile_fixes(
    entries: List[tuple],
    dockerfile: str,
) -> List[tuple]:
    df_clean = _strip_comments(dockerfile)
    df_lower = df_clean.lower()

    has_apt_upgrade = ("apt-get upgrade" in df_lower or
                       "apt-get dist-upgrade" in df_lower)

    base_is_modern = False
    from_match = re.search(r"from\s+python:(\d+)\.(\d+)", df_lower)
    if from_match and int(from_match.group(2)) >= 12:
        base_is_modern = True

    has_java = any(tok in df_lower for tok in ("jdk", "jre", "java"))

    remaining = []
    for sev, pkg, installed, fixed in entries:
        pkg_l = pkg.lower()

        if fixed is None:
            remaining.append((sev, pkg, installed, fixed))
            continue

        if base_is_modern and pkg_l in _SYSTEM_PKGS:
            continue
        if has_apt_upgrade and fixed and pkg_l in _SYSTEM_PKGS:
            continue
        if pkg_l in _PIP_PKGS and _pip_explicitly_upgrades(pkg_l, dockerfile):
            continue
        if pkg_l == "log4j" and not has_java:
            continue

        remaining.append((sev, pkg, installed, fixed))

    return remaining


# ───────────────────────────────────────────────────────────────────────────
# Best-practice & anti-pattern analysis
# ───────────────────────────────────────────────────────────────────────────

_SECRET_PATTERN = r'\b(?:ENV|ARG)\b\s+\w*(?:PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY|CREDENTIALS)\w*\s*='
_SENSITIVE_PORTS = {"5432", "3306", "27017", "6379", "1433", "11211"}
_CRED_URL_PATTERN = r'https?://[^:@\s]+:[^@\s]+@\S+'
_BASE64_SECRET_PATTERN = r'\b(?:ENV|ARG)\b\s+\w+\s*=\s*[A-Za-z0-9+/]{20,}={0,2}\s*$'


def check_best_practices(dockerfile: str) -> dict:
    df = _strip_comments(dockerfile.strip())
    df_lower = df.lower()
    lines = df.splitlines()

    last_build_idx = -1
    last_nonroot_user_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith(("RUN ", "COPY ", "ADD ")):
            last_build_idx = i
        if stripped.startswith("USER "):
            user_val = stripped.split("USER ", 1)[1].split()[0]
            if user_val not in ("ROOT", "0"):
                last_nonroot_user_idx = i
    has_user = last_nonroot_user_idx > last_build_idx >= 0

    has_healthcheck = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("HEALTHCHECK"):
            body = stripped[len("HEALTHCHECK"):].strip().upper()
            if body.startswith("NONE"):
                continue
            body_no_flags = re.sub(r"--\w+[= ]\S+\s*", "", body).strip()
            if body_no_flags.startswith("CMD"):
                cmd = body_no_flags[3:].strip()
                if cmd and cmd not in ("TRUE", "EXIT 0", "true", "/bin/true"):
                    has_healthcheck = True

    has_pip_no_cache = any(
        "--no-cache-dir" in line and ("pip install" in line or "pip3 install" in line)
        for line in df_lower.splitlines()
    )

    consecutive_runs = 0
    max_consecutive_runs = 0
    for line in lines:
        if line.strip().upper().startswith("RUN "):
            consecutive_runs += 1
            max_consecutive_runs = max(max_consecutive_runs, consecutive_runs)
        else:
            consecutive_runs = 0
    layer_efficient = max_consecutive_runs <= 2

    req_copy_idx = -1
    bulk_copy_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"COPY\s+requirements\.txt\b", stripped, re.IGNORECASE):
            req_copy_idx = i
        if re.match(r"COPY\s+\.\s+\.", stripped, re.IGNORECASE):
            bulk_copy_idx = i
    copy_order_ok = (req_copy_idx < bulk_copy_idx) if (req_copy_idx >= 0 and bulk_copy_idx >= 0) else True

    has_minimal_pkgs = True
    for line in df_lower.splitlines():
        if "apt-get install" in line and "--no-install-recommends" not in line:
            has_minimal_pkgs = False
            break

    return {
        "non_root_user":      has_user,
        "healthcheck":        has_healthcheck,
        "no_secrets_in_env":  not bool(re.search(_SECRET_PATTERN, df, re.IGNORECASE)),
        "apt_cache_cleanup":  "rm -rf /var/lib/apt/lists" in df_lower,
        "pip_no_cache":       has_pip_no_cache,
        "copy_over_add":      not bool(re.search(r"^\s*ADD\s", df, re.MULTILINE | re.IGNORECASE)),
        "modern_base_image":  bool(re.search(r"FROM\s+python:3\.1[2-9]", df, re.IGNORECASE)),
        "layer_efficiency":   layer_efficient,
        "copy_order":         copy_order_ok,
        "minimal_packages":   has_minimal_pkgs,
    }


def detect_antipatterns(dockerfile: str) -> List[str]:
    warnings: List[str] = []
    df = _strip_comments(dockerfile.strip())
    df_lower = df.lower()

    if re.search(_SECRET_PATTERN, df, re.IGNORECASE):
        warnings.append("Secrets exposed in ENV instructions (passwords, tokens, API keys)")

    if re.search(r"^\s*ADD\s+https?://", df, re.MULTILINE | re.IGNORECASE):
        warnings.append("ADD from URL — use curl/wget + COPY for better caching and auditability")

    if re.search(r"(?:curl|wget)\s+[^|]*\|\s*(?:sh|bash)", df_lower):
        warnings.append("Pipe-to-shell pattern (curl|sh) — supply chain risk, unverified code execution")

    if re.search(r"FROM\s+\S+:latest\b", df, re.IGNORECASE):
        warnings.append("Using :latest tag — builds are not reproducible")

    for match in re.finditer(r"EXPOSE\s+(\d+)", df, re.IGNORECASE):
        port = match.group(1)
        if port in _SENSITIVE_PORTS:
            warnings.append(f"Exposing database/cache port {port} — should not be in application image")

    if re.search(_CRED_URL_PATTERN, df):
        warnings.append("Credentials embedded in URL (user:pass@host) — use build secrets or env vars at runtime")

    if re.search(_BASE64_SECRET_PATTERN, df, re.MULTILINE):
        warnings.append("Possible base64-encoded secret in ENV/ARG — secrets should not be baked into images")

    if re.search(r"get\.docker\.com", df_lower):
        warnings.append("Installing Docker-in-Docker — should not be in application images")

    if re.search(r"chmod\s+777\b", df_lower):
        warnings.append("chmod 777 — overly permissive file permissions, use minimal permissions")

    lines = df.splitlines()
    bulk_copy_seen = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"COPY\s+\.\s+\.", stripped, re.IGNORECASE):
            bulk_copy_seen = True
        if bulk_copy_seen and re.match(r"COPY\s+requirements\.txt\b", stripped, re.IGNORECASE):
            warnings.append("COPY . . before COPY requirements.txt — invalidates Docker layer cache")
            break

    last_build_idx = -1
    last_nonroot_user_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith(("RUN ", "COPY ", "ADD ")):
            last_build_idx = i
        if stripped.startswith("USER "):
            user_val = stripped.split("USER ", 1)[1].split()[0]
            if user_val not in ("ROOT", "0"):
                last_nonroot_user_idx = i
    if not (last_nonroot_user_idx > last_build_idx >= 0):
        warnings.append("Running as root — no non-root USER instruction")

    return warnings


# ───────────────────────────────────────────────────────────────────────────
# Regression and conflict detection
# ───────────────────────────────────────────────────────────────────────────

def _detect_regressions(dockerfile: str, difficulty: int = 1) -> List[Vulnerability]:
    regressions: List[Vulnerability] = []
    df_lower = dockerfile.lower()

    from_match = re.search(r"FROM\s+(\S+)", dockerfile, re.IGNORECASE)
    if not from_match:
        return regressions

    image_ref = from_match.group(1).lower()

    if ":latest" in image_ref:
        regressions.append(Vulnerability(
            cve_id="REGRESSION-001", package_name="base-image",
            installed_version="latest", fixed_version=None,
            severity=Severity.MEDIUM, score=5.0,
            description="Using :latest tag makes builds non-reproducible and may "
                        "pull images with unknown vulnerability status",
        ))

    is_alpine = "alpine" in image_ref
    if is_alpine:
        for cext_pkg in _C_EXTENSION_PKGS:
            if re.search(rf"\b{re.escape(cext_pkg)}\b", df_lower):
                regressions.append(Vulnerability(
                    cve_id="REGRESSION-003", package_name=f"musl-{cext_pkg}",
                    installed_version="alpine", fixed_version=None,
                    severity=Severity.HIGH, score=7.0,
                    description=f"Alpine uses musl libc; {cext_pkg} has C extensions that "
                                f"require glibc. Build will fail or produce corrupt binaries.",
                ))
                break

    if difficulty >= 5:
        has_crypto_upgrade = _pip_explicitly_upgrades("cryptography", dockerfile)
        has_libssl1 = bool(re.search(r"libssl1\.", df_lower))
        if has_crypto_upgrade and has_libssl1:
            regressions.append(Vulnerability(
                cve_id="CONFLICT-001", package_name="cryptography+libssl1.1",
                installed_version="conflict", fixed_version=None,
                severity=Severity.HIGH, score=7.5,
                description="cryptography>=42.0.0 requires OpenSSL 3.x but libssl1.1 is still "
                            "installed. Remove libssl1.1 or keep cryptography<42.",
            ))

    if difficulty >= 4:
        has_gcc_in_original = "gcc" in df_lower or "build-essential" in df_lower
        needs_compilation = any(
            re.search(rf"\b{re.escape(pkg)}\b", df_lower)
            for pkg in ("numpy", "scipy", "pillow", "cryptography")
        )
        # Check if it's a multi-stage build — gcc removal is fine in final stage
        is_multistage = len(re.findall(r"^\s*FROM\s+", dockerfile, re.MULTILINE | re.IGNORECASE)) > 1
        if not has_gcc_in_original and needs_compilation and not is_multistage:
            regressions.append(Vulnerability(
                cve_id="BUILD-FAIL-001", package_name="gcc",
                installed_version="missing", fixed_version=None,
                severity=Severity.MEDIUM, score=5.0,
                description="C compiler (gcc) removed but pip packages with C extensions "
                            "require compilation. Use a multi-stage build or keep gcc.",
            ))

    return regressions


# ───────────────────────────────────────────────────────────────────────────
# Mock scanner entry point
# ───────────────────────────────────────────────────────────────────────────

_SEVERITY_SCORE = {
    Severity.CRITICAL: 9.8, Severity.HIGH: 7.5,
    Severity.MEDIUM: 5.0, Severity.LOW: 2.5,
}


def select_vuln_subset(
    difficulty: int, rng: "random.Random",
) -> list:
    """Randomly select a subset of CVEs from the template pool for this episode."""
    full_pool = list(_MOCK_TEMPLATES.get(difficulty, _MOCK_TEMPLATES[1]))
    ranges = {1: (3, 4), 2: (5, 8), 3: (8, 12)}
    lo, hi = ranges.get(difficulty, (3, len(full_pool)))
    count = rng.randint(lo, min(hi, len(full_pool)))
    return rng.sample(full_pool, count)


def scan_mock(
    image_tag: str,
    difficulty: int = 1,
    current_dockerfile: str | None = None,
    vuln_pool: list | None = None,
) -> VulnReport:
    """Return a mock vulnerability report.

    *vuln_pool*: if provided, uses this specific set of CVEs instead of the
    full template (enables per-episode randomization).

    When *current_dockerfile* is provided, simulates which CVEs are resolved
    by the Dockerfile changes and detects any regressions introduced.
    """
    import random
    rng = random.Random(hash(image_tag) % 2**32)

    entries = list(vuln_pool or _MOCK_TEMPLATES.get(difficulty, _MOCK_TEMPLATES[1]))
    if current_dockerfile:
        entries = _simulate_dockerfile_fixes(entries, current_dockerfile)

    cve_pool = [f"CVE-2024-{rng.randint(1000, 9999)}" for _ in range(30)]
    vulns = [
        Vulnerability(
            cve_id=cve_pool[i], package_name=pkg,
            installed_version=installed, fixed_version=fixed,
            severity=sev, score=_SEVERITY_SCORE.get(sev, 0.0),
            description=_CVE_DESCRIPTIONS.get(pkg, f"Vulnerability in {pkg}"),
        )
        for i, (sev, pkg, installed, fixed) in enumerate(entries)
    ]

    if current_dockerfile:
        vulns.extend(_detect_regressions(current_dockerfile, difficulty))

    return VulnReport(
        image_tag=image_tag, scan_tool="mock",
        scanned_at=datetime.datetime.utcnow().isoformat(),
        vulnerabilities=vulns, total_packages=50,
    )


# ───────────────────────────────────────────────────────────────────────────
# Unified dispatch
# ───────────────────────────────────────────────────────────────────────────

SCANNER_REGISTRY = {
    "trivy": scan_with_trivy,
    "grype": scan_with_grype,
    "mock":  scan_mock,
}


def run_scan(image_tag: str, scanner: str = "trivy", **kwargs) -> VulnReport:
    """Dispatch to the named scanner backend."""
    if scanner not in SCANNER_REGISTRY:
        raise ScannerError(f"Unknown scanner '{scanner}'. Choose from: {list(SCANNER_REGISTRY)}")
    return SCANNER_REGISTRY[scanner](image_tag, **kwargs)
