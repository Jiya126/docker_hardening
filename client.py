# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async OpenEnv client for the Docker Hardening environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        DockerHardeningAction,
        DockerHardeningObservation,
        DockerHardeningState,
    )
except ImportError:
    from models import (
        DockerHardeningAction,
        DockerHardeningObservation,
        DockerHardeningState,
    )


class DockerHardeningEnv(
    EnvClient[DockerHardeningAction, DockerHardeningObservation, DockerHardeningState]
):

    def _step_payload(self, action: DockerHardeningAction) -> Dict:
        return {"patched_dockerfile": action.patched_dockerfile}

    def _parse_result(self, payload: Dict) -> StepResult[DockerHardeningObservation]:
        obs_data = payload.get("observation", {})
        observation = DockerHardeningObservation(
            current_dockerfile=obs_data.get("current_dockerfile", ""),
            vulnerability_summary=obs_data.get("vulnerability_summary", ""),
            task_name=obs_data.get("task_name", ""),
            initial_vuln_count=obs_data.get("initial_vuln_count", 0),
            current_vuln_count=obs_data.get("current_vuln_count", 0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 5),
            score=obs_data.get("score", 0.0),
            step_summary=obs_data.get("step_summary", ""),
            last_action_error=obs_data.get("last_action_error"),
            termination_reason=obs_data.get("termination_reason"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DockerHardeningState:
        return DockerHardeningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            cycle_count=payload.get("cycle_count", 0),
        )
