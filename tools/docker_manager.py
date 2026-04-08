"""Manages Dockerfile writes and image builds (real or mock) per episode."""

import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class DockerBuildError(Exception):
    pass


class DockerBuildManager:

    def __init__(
        self,
        base_image_tag: str,
        episode_id: str,
        context_dir: Optional[str] = None,
        build_timeout: int = 300,
        use_mock: bool = False,
    ):
        self.base_image_tag = base_image_tag
        self.episode_id = episode_id
        self.context_dir = context_dir
        self.build_timeout = build_timeout
        self.use_mock = use_mock

        self.build_dir = tempfile.mkdtemp(prefix=f"rl_hardening_{episode_id}_")
        self._current_tag = base_image_tag
        self._current_dockerfile: Optional[str] = None
        self._cycle = 0

    @property
    def current_tag(self) -> str:
        return self._current_tag

    @property
    def current_dockerfile(self) -> str:
        return self._current_dockerfile or ""

    def initialise(self, dockerfile_content: str) -> None:
        self._current_dockerfile = dockerfile_content
        self._prepare_build_context(dockerfile_content)
        self._write_dockerfile(dockerfile_content)

    def apply_patch(self, patched_dockerfile: str) -> Tuple[str, str]:
        self._cycle += 1
        new_tag = f"rl-hardened/{self.episode_id}:cycle-{self._cycle}"

        self._write_dockerfile(patched_dockerfile)

        if self.use_mock:
            self._current_dockerfile = patched_dockerfile
            self._current_tag = new_tag
            return new_tag, f"[MOCK BUILD] {new_tag}"

        build_log = self._docker_build(new_tag)
        self._current_dockerfile = patched_dockerfile
        self._current_tag = new_tag
        return new_tag, build_log

    def cleanup(self) -> None:
        if os.path.exists(self.build_dir):
            shutil.rmtree(self.build_dir, ignore_errors=True)
        if not self.use_mock:
            self._prune_episode_images()

    def _prepare_build_context(self, dockerfile_content: str) -> None:
        self._clear_build_dir()
        if self.context_dir:
            self._copy_context_tree(self.context_dir)
        else:
            self._seed_minimal_context(dockerfile_content)

    def _clear_build_dir(self) -> None:
        for entry in os.listdir(self.build_dir):
            path = os.path.join(self.build_dir, entry)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)

    def _copy_context_tree(self, context_dir: str) -> None:
        src_root = Path(context_dir)
        dst_root = Path(self.build_dir)
        ignore_names = {".git", ".venv", "__pycache__", ".pytest_cache"}

        for root, dirs, files in os.walk(src_root):
            dirs[:] = [d for d in dirs if d not in ignore_names]
            root_path = Path(root)
            rel_root = root_path.relative_to(src_root)
            target_root = dst_root / rel_root
            target_root.mkdir(parents=True, exist_ok=True)

            for name in files:
                if name.endswith((".pyc", ".pyo", ".pyd")):
                    continue
                src_file = root_path / name
                dst_file = target_root / name
                if src_file.resolve() == dst_root.resolve():
                    continue
                shutil.copy2(src_file, dst_file)

    def _seed_minimal_context(self, dockerfile_content: str) -> None:
        if "COPY requirements.txt" in dockerfile_content:
            self._write_context_file("requirements.txt", "requests>=2.20.0\n")
        if "COPY . ." in dockerfile_content or 'CMD ["python", "app.py"]' in dockerfile_content:
            self._write_context_file("app.py", "print('docker-hardening placeholder app')\n")

    def _write_context_file(self, relative_path: str, content: str) -> None:
        path = os.path.join(self.build_dir, relative_path)
        os.makedirs(os.path.dirname(path) or self.build_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _write_dockerfile(self, content: str) -> None:
        path = os.path.join(self.build_dir, "Dockerfile")
        with open(path, "w") as f:
            f.write(content)

    def _docker_build(self, tag: str) -> str:
        cmd = [
            "docker", "build", "--no-cache",
            "-t", tag,
            "-f", os.path.join(self.build_dir, "Dockerfile"),
            self.build_dir,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.build_timeout,
            )
        except subprocess.TimeoutExpired:
            raise DockerBuildError(f"docker build timed out after {self.build_timeout}s")

        log = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            raise DockerBuildError(f"docker build failed:\n{log[-2000:]}")
        return log

    def _prune_episode_images(self) -> None:
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
                 "--filter", f"reference=rl-hardened/{self.episode_id}:*"],
                capture_output=True, text=True, timeout=30,
            )
            tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
            if tags:
                subprocess.run(["docker", "rmi", "--force"] + tags,
                               capture_output=True, timeout=60)
        except Exception:
            pass


def load_dockerfile(path: str) -> str:
    with open(path) as f:
        return f.read()


def diff_dockerfiles(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="Dockerfile.before", tofile="Dockerfile.after", lineterm="",
    ))
