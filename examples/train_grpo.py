#!/usr/bin/env python3
"""
Docker Hardening RL — Offline Curriculum Demo
==============================================
Demonstrates the environment loop by running a mock patcher through all
curriculum levels. No server, no LLM API keys, no GPU needed.

This is a quick sanity check that the environment, scanner, and reward
engine all work together before running actual GRPO training.

Usage:
    cd docker_hardening
    python examples/train_grpo.py                      # offline mock episodes
    python examples/train_grpo.py --difficulty 3        # just level 3
    python examples/train_grpo.py --mode curriculum     # all levels with graduation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import DockerHardeningAction
from server.docker_hardening_environment import DockerHardeningEnvironment
from curriculum import CURRICULUM, get_level


def run_offline_episode(task_name: str, verbose: bool = True) -> dict:
    """Run a full episode using the environment directly (no HTTP server)."""
    os.environ["SCA_GYM_TASK"] = task_name
    env = DockerHardeningEnvironment()
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_name} | Initial vulns: {obs.initial_vuln_count}")
        print(f"{'='*60}")

    rewards = []
    step = 0

    while not obs.done:
        step += 1
        # Use the original dockerfile as a no-change baseline (will get penalty)
        # In real training, the policy model generates the patched dockerfile
        action = DockerHardeningAction(patched_dockerfile=obs.current_dockerfile)
        obs = env.step(action)
        rewards.append(obs.reward or 0.0)

        if verbose:
            print(
                f"  Step {step}: vulns={obs.current_vuln_count}/{obs.initial_vuln_count} "
                f"reward={obs.reward:.3f} score={obs.score:.3f}"
            )
            if obs.termination_reason:
                print(f"  Terminated: {obs.termination_reason.value}")

    return {
        "task": task_name,
        "total_reward": sum(rewards),
        "score": obs.score,
        "steps": step,
        "final_vulns": obs.current_vuln_count,
        "termination": obs.termination_reason.value if obs.termination_reason else None,
    }


TASK_MAP = {1: "patch_easy", 2: "patch_medium", 3: "patch_hard"}


def run_curriculum(episodes_per_level: int = 3, verbose: bool = True):
    """Run through all curriculum levels."""
    for level, task_name in sorted(TASK_MAP.items()):
        lvl_config = CURRICULUM.get(level)
        if lvl_config:
            print(f"\n{'#'*60}")
            print(f"# LEVEL {level}: {lvl_config.name}")
            print(f"{'#'*60}")

        for ep in range(episodes_per_level):
            result = run_offline_episode(task_name, verbose=verbose)
            print(
                f"  Episode {ep+1}: score={result['score']:.2f} "
                f"reward={result['total_reward']:.2f} "
                f"vulns={result['final_vulns']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker Hardening — Offline Demo")
    parser.add_argument("--mode", choices=["offline", "curriculum"], default="offline")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "curriculum":
        run_curriculum(episodes_per_level=args.episodes)
    else:
        task_name = TASK_MAP[args.difficulty]
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep+1}/{args.episodes} ---")
            run_offline_episode(task_name)
