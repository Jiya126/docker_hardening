"""7-tier difficulty curriculum for RL training."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DifficultyLevel:
    level: int
    name: str
    description: str
    base_images: List[str]
    scanner: str                              # 'trivy' | 'grype' | 'mock'
    max_cycles: int
    target_vuln_count: int                    # success threshold
    expected_severity_mix: Dict[str, int]
    graduation_reward_threshold: float        # advance when avg reward >= this
    sample_dockerfiles: List[str] = field(default_factory=list)


CURRICULUM: Dict[int, DifficultyLevel] = {

    1: DifficultyLevel(
        level=1,
        name="Low Vulnerability Baseline",
        description=(
            "Recent slim images with only LOW/MEDIUM CVEs. "
            "Agent learns basic package upgrades and non-root user patterns."
        ),
        base_images=["python:3.11-slim", "node:18-alpine", "nginx:1.25-alpine"],
        scanner="mock",
        max_cycles=5,
        target_vuln_count=2,
        expected_severity_mix={"CRITICAL": 0, "HIGH": 0, "MEDIUM": 2, "LOW": 3},
        graduation_reward_threshold=30.0,
        sample_dockerfiles=[
            """\
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
""",
        ],
    ),

    2: DifficultyLevel(
        level=2,
        name="Medium Vulnerability Challenge",
        description=(
            "Older base images with multiple HIGH CVEs. "
            "Agent must upgrade base image or apply multi-package fixes."
        ),
        base_images=["python:3.9-slim", "node:14-buster", "ubuntu:20.04"],
        scanner="mock",
        max_cycles=8,
        target_vuln_count=1,
        expected_severity_mix={"CRITICAL": 0, "HIGH": 3, "MEDIUM": 4, "LOW": 5},
        graduation_reward_threshold=55.0,
        sample_dockerfiles=[
            """\
FROM python:3.9-slim
RUN apt-get update && apt-get install -y curl wget
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
""",
        ],
    ),

    3: DifficultyLevel(
        level=3,
        name="Critical Vulnerability Gauntlet",
        description=(
            "Legacy images with CRITICAL CVEs (log4j era, old openssl, etc.). "
            "Agent must perform multi-step fixes including base image replacement."
        ),
        base_images=["python:3.6-slim", "java:8", "ubuntu:18.04"],
        scanner="mock",
        max_cycles=12,
        target_vuln_count=0,
        expected_severity_mix={"CRITICAL": 2, "HIGH": 4, "MEDIUM": 5, "LOW": 6},
        graduation_reward_threshold=80.0,
        sample_dockerfiles=[
            """\
FROM python:3.6-slim
RUN apt-get update && apt-get install -y \\
    libssl1.0 \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
""",
            """\
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    openssl \\
    liblog4j2-java \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
CMD ["python3", "app.py"]
""",
        ],
    ),

    4: DifficultyLevel(
        level=4,
        name="Multi-Stage Build Mastery",
        description=(
            "Multi-stage Dockerfiles where build deps (gcc, python3-dev) belong "
            "in the builder stage only. Agent must preserve AS builder/final stage "
            "structure. Removing gcc from builder breaks C-extension pip installs."
        ),
        base_images=["python:3.9-slim"],
        scanner="mock",
        max_cycles=10,
        target_vuln_count=0,
        expected_severity_mix={"CRITICAL": 0, "HIGH": 2, "MEDIUM": 4, "LOW": 3},
        graduation_reward_threshold=70.0,
        sample_dockerfiles=[
            """\
FROM python:3.9-slim AS builder
RUN apt-get update && apt-get install -y gcc python3-dev libffi-dev
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
CMD ["python", "app.py"]
""",
        ],
    ),

    5: DifficultyLevel(
        level=5,
        name="Conflicting Requirements",
        description=(
            "Conflicting dependencies: cryptography>=42 needs openssl>=3.0 but "
            "libssl1.1 is pinned. Some CVEs have no fix. Switching to alpine "
            "breaks numpy C extensions. Tests trade-off reasoning."
        ),
        base_images=["python:3.9-slim"],
        scanner="mock",
        max_cycles=10,
        target_vuln_count=2,
        expected_severity_mix={"CRITICAL": 1, "HIGH": 3, "MEDIUM": 4, "LOW": 2},
        graduation_reward_threshold=65.0,
        sample_dockerfiles=[
            """\
FROM python:3.9-slim
RUN apt-get update && apt-get install -y \\
    libssl1.1 \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
""",
        ],
    ),

    6: DifficultyLevel(
        level=6,
        name="Subtle Anti-Pattern Detection",
        description=(
            "Secrets hidden in non-obvious ways: base64 tokens in ENV, credentials "
            "in URLs, build-arg with hardcoded values. HEALTHCHECK depends on curl "
            "but prior step removed it. COPY . . before COPY requirements.txt."
        ),
        base_images=["python:3.9-slim"],
        scanner="mock",
        max_cycles=8,
        target_vuln_count=0,
        expected_severity_mix={"CRITICAL": 0, "HIGH": 3, "MEDIUM": 4, "LOW": 1},
        graduation_reward_threshold=75.0,
        sample_dockerfiles=[
            """\
FROM python:3.9-slim
ARG REPO_URL=https://deploy:ghp_s3cretToken123@git.example.com/org/repo.git
ENV AUTH_TOKEN=ZGVwbG95OnMzY3JldFBhc3N3b3JkMTIz
RUN apt-get update && apt-get install -y curl wget git
WORKDIR /app
COPY . .
COPY requirements.txt .
RUN pip install -r requirements.txt
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "app.py"]
""",
        ],
    ),

    7: DifficultyLevel(
        level=7,
        name="Adversarial Trap Gauntlet",
        description=(
            "Decoy vulns that look fixable but aren't (myapp-crypto != cryptography). "
            "Port 8080 IS the application port. USER root needed for mid-build RUN. "
            "Conflicting requirements.txt pins. Tests whether agent avoids over-patching."
        ),
        base_images=["python:3.9-slim"],
        scanner="mock",
        max_cycles=6,
        target_vuln_count=3,
        expected_severity_mix={"CRITICAL": 1, "HIGH": 4, "MEDIUM": 4, "LOW": 2},
        graduation_reward_threshold=60.0,
        sample_dockerfiles=[
            """\
FROM python:3.9-slim
RUN apt-get update && apt-get install -y \\
    gcc \\
    python3-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
USER root
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["python", "app.py"]
""",
        ],
    ),
}


def get_level(level: int) -> DifficultyLevel:
    if level not in CURRICULUM:
        raise ValueError(f"Unknown difficulty level {level}. Choose from {list(CURRICULUM.keys())}")
    return CURRICULUM[level]


def get_sample_dockerfile(level: int, index: int = 0) -> str:
    lvl = get_level(level)
    if not lvl.sample_dockerfiles:
        return f"FROM {lvl.base_images[0]}\n"
    return lvl.sample_dockerfiles[index % len(lvl.sample_dockerfiles)]


def get_sample_image(level: int, index: int = 0) -> str:
    lvl = get_level(level)
    return lvl.base_images[index % len(lvl.base_images)]
