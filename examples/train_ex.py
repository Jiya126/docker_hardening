#!/usr/bin/env python3
"""
Docker Hardening — Minimal Client Example
==========================================
Shows how to connect to a running environment server and run one episode.

Usage:
    # Start the server first:
    cd docker_hardening && uvicorn server.app:app --port 8000

    # Then run this client:
    python examples/train_ex.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client import DockerHardeningEnv
from models import DockerHardeningAction


def main() -> None:
    base_url = os.environ.get("DOCKER_HARDENING_URL", "http://127.0.0.1:8000")

    with DockerHardeningEnv(base_url=base_url, message_timeout_s=300).sync() as env:
        result = env.reset()
        obs = result.observation
        print(f"Task: {obs.task_name} | Initial vulns: {obs.initial_vuln_count}")

        while not result.done:
            action = DockerHardeningAction(patched_dockerfile=obs.current_dockerfile)
            result = env.step(action)
            obs = result.observation
            print(
                f"Step {obs.step_number}: vulns={obs.current_vuln_count} "
                f"reward={result.reward:.3f} score={obs.score:.3f}"
            )


if __name__ == "__main__":
    main()
