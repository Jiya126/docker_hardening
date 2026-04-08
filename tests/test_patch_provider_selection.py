"""
Tests for patch agent provider selection, build manager context seeding,
and Dockerfile sanitization.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import DockerHardeningAction
from server.docker_hardening_environment import DockerHardeningEnvironment
from tools.docker_manager import DockerBuildManager
from tools.patch_agent import LLMPatchAgent, normalise_provider
from tools.scanner import scan_mock


SAMPLE_DOCKERFILE = """\
FROM python:3.9-slim
WORKDIR /app
COPY . .
CMD ["python", "app.py"]
"""


def make_env(task="patch_easy") -> DockerHardeningEnvironment:
    os.environ["SCA_GYM_TASK"] = task
    return DockerHardeningEnvironment()


def test_normalise_provider_aliases_openai_to_gpt():
    assert normalise_provider("openai") == "gpt"
    assert normalise_provider(" GPT ") == "gpt"


def test_patch_agent_records_selected_gpt_provider_in_mock_mode():
    agent = LLMPatchAgent(provider="gpt", model="gpt-4.1", use_mock=True)
    report = scan_mock("test:latest", difficulty=2)
    attempt = agent.patch(SAMPLE_DOCKERFILE, report, [], cycle=1)
    assert attempt.provider == "gpt"
    assert attempt.model == "gpt-4.1"


def test_environment_reset_and_step_with_new_api():
    env = make_env("patch_medium")
    obs = env.reset()
    assert obs.task_name == "patch_medium"
    assert obs.initial_vuln_count > 0

    patched = 'FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nCMD ["python", "app.py"]\n'
    obs = env.step(DockerHardeningAction(patched_dockerfile=patched))
    assert obs.reward > 0
    assert obs.current_vuln_count < obs.initial_vuln_count


def test_environment_reports_best_practices_and_antipatterns():
    env = make_env("patch_hard")
    obs = env.reset()
    assert len(obs.best_practices) > 0
    assert len(obs.antipattern_warnings) > 0
    assert any("FAIL" in bp for bp in obs.best_practices)


def test_environment_rewards_antipattern_removal():
    env = make_env("patch_hard")
    obs = env.reset()
    initial_ap = len(obs.antipattern_warnings)
    assert initial_ap > 0

    clean_patch = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m appuser\n"
        "WORKDIR /app\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . .\n"
        'HEALTHCHECK CMD python -c "import sys"\n'
        "USER appuser\n"
        'CMD ["python", "app.py"]\n'
    )
    obs = env.step(DockerHardeningAction(patched_dockerfile=clean_patch))
    assert len(obs.antipattern_warnings) < initial_ap
    assert obs.reward > 0


def test_docker_build_manager_seeds_minimal_context():
    dockerfile = """\
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
    mgr = DockerBuildManager(
        base_image_tag="python:3.9-slim",
        episode_id="test-seeded-context", use_mock=True,
    )
    try:
        mgr.initialise(dockerfile)
        req_path = Path(mgr.build_dir, "requirements.txt")
        assert req_path.exists()
        assert len(req_path.read_text().strip()) > 0
        assert "placeholder app" in Path(mgr.build_dir, "app.py").read_text()
    finally:
        mgr.cleanup()


def test_docker_build_manager_copies_supplied_context_directory():
    with tempfile.TemporaryDirectory() as ctx:
        Path(ctx, "requirements.txt").write_text("flask==3.0.0\n")
        Path(ctx, "app.py").write_text("print('real app')\n")
        Path(ctx, "nested").mkdir()
        Path(ctx, "nested", "data.txt").write_text("hello\n")

        mgr = DockerBuildManager(
            base_image_tag="python:3.9-slim",
            episode_id="test-copy-context", context_dir=ctx, use_mock=True,
        )
        try:
            mgr.initialise(SAMPLE_DOCKERFILE)
            assert Path(mgr.build_dir, "requirements.txt").read_text() == "flask==3.0.0\n"
            assert Path(mgr.build_dir, "app.py").read_text() == "print('real app')\n"
            assert Path(mgr.build_dir, "nested", "data.txt").read_text() == "hello\n"
        finally:
            mgr.cleanup()


def test_patch_agent_strips_invented_digest_from_from_line():
    agent = LLMPatchAgent(provider="gpt", model="gpt-4.1", use_mock=True)
    raw = """
    {
      "strategy": "replace_base_image",
      "reasoning": "upgrade base image",
      "patched_dockerfile": "FROM python:3.11-slim@sha256:4e3e99baa9740e250a6f0cdef7b27cd1b732fa96aa708ced84eaca8f5d361fbb\\nWORKDIR /app\\nCOPY . .\\nCMD [\\"python\\", \\"app.py\\"]",
      "changes_summary": "updated base image"
    }
    """
    attempt = agent._parse_response(
        raw,
        'FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD ["python", "app.py"]\n',
        cycle=1,
    )
    assert attempt.success is True
    assert attempt.patched_dockerfile is not None
    assert "@sha256:" not in attempt.patched_dockerfile


def test_patch_agent_rejects_runtime_requirements_txt_mutation():
    agent = LLMPatchAgent(provider="gpt", model="gpt-4.1", use_mock=True)
    raw = """
    {
      "strategy": "upgrade_package",
      "reasoning": "upgrade python dependencies",
      "patched_dockerfile": "FROM python:3.11-slim\\nWORKDIR /app\\nCOPY requirements.txt .\\nRUN echo 'wheel==0.46.2' >> requirements.txt && pip install -r requirements.txt\\nCOPY . .\\nCMD [\\"python\\", \\"app.py\\"]",
      "changes_summary": "updated requirements"
    }
    """
    attempt = agent._parse_response(
        raw,
        'FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\n'
        'RUN pip install -r requirements.txt\nCOPY . .\nCMD ["python", "app.py"]\n',
        cycle=1,
    )
    assert attempt.success is False
    assert "modify requirements.txt" in (attempt.error_message or "")
