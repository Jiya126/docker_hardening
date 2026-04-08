# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic v2 models for the Docker Hardening environment."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field
from openenv.core.env_server.types import Action, Observation, State


class Severity(str, Enum):
    CRITICAL   = "CRITICAL"
    HIGH       = "HIGH"
    MEDIUM     = "MEDIUM"
    LOW        = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"
    UNKNOWN    = "UNKNOWN"


class PatchStrategy(str, Enum):
    UPGRADE_PACKAGE    = "upgrade_package"
    REPLACE_BASE_IMAGE = "replace_base_image"
    REMOVE_PACKAGE     = "remove_package"
    ADD_SECURITY_CONFIG = "add_security_config"
    MULTI_STEP         = "multi_step"


class TerminationReason(str, Enum):
    MAX_CYCLES_REACHED          = "max_cycles_reached"
    TARGET_VULNS_REACHED        = "target_vulns_reached"
    ALL_VULNS_FIXED             = "all_vulns_fixed"
    PATCH_FAILED_TOO_MANY_TIMES = "patch_failed_too_many_times"


class Vulnerability(BaseModel):
    cve_id:            str
    package_name:      str
    installed_version: str
    fixed_version:     Optional[str] = None
    severity:          Severity      = Severity.UNKNOWN
    description:       str           = ""
    layer:             Optional[str] = None
    score:             float         = 0.0
    cvss_vector:       Optional[str] = None


class VulnReport(BaseModel):
    image_tag:       str
    scan_tool:       str
    scanned_at:      str
    vulnerabilities: List[Vulnerability] = []
    os_info:         Optional[str]       = None
    total_packages:  int                 = 0

    @computed_field
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == Severity.CRITICAL)

    @computed_field
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == Severity.HIGH)

    @computed_field
    @property
    def total_count(self) -> int:
        return len(self.vulnerabilities)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s.value: 0 for s in Severity}
        for v in self.vulnerabilities:
            counts[v.severity.value] += 1
        return counts


class PatchAttempt(BaseModel):
    cycle:              int
    patch_strategy:     PatchStrategy
    dockerfile_diff:    str
    patched_dockerfile: Optional[str] = None
    success:            bool
    provider:           str           = "agent"
    model:              Optional[str] = None
    error_message:      Optional[str] = None
    vulns_before:       int           = 0
    vulns_after:        int           = 0
    agent_reasoning:    Optional[str] = None


class DockerHardeningAction(Action):
    patched_dockerfile: str = Field(
        ...,
        description="The full patched Dockerfile content produced by the agent",
    )


class DockerHardeningObservation(Observation):
    current_dockerfile:    str                         = Field(default="",  description="Full text of the current Dockerfile")
    vulnerability_summary: str                         = Field(default="",  description="Human-readable vulnerability report for the LLM agent")
    task_name:             str                         = Field(default="",  description="Current task: patch_easy | patch_medium | patch_hard")
    initial_vuln_count:    int                         = Field(default=0,   description="Number of vulns at episode start")
    current_vuln_count:    int                         = Field(default=0,   description="Number of remaining vulnerabilities")
    step_number:           int                         = Field(default=0,   description="Current step number")
    max_steps:             int                         = Field(default=5,   description="Maximum steps allowed")
    score:                 float                       = Field(default=0.0, description="Normalized score [0.0, 1.0]")
    step_summary:          str                         = Field(default="",  description="Human-readable summary of this step's result")
    last_action_error:     Optional[str]               = Field(None,        description="Error from the last action, if any")
    termination_reason:    Optional[TerminationReason] = Field(None,        description="Why the episode ended, if done=True")
    security_score:        float                       = Field(default=0.0, description="Security audit score 0-100")
    best_practices:        List[str]                   = Field(default_factory=list, description="Satisfied best practices, e.g. '[PASS] Non-root USER'")
    antipattern_warnings:  List[str]                   = Field(default_factory=list, description="Detected Dockerfile security anti-patterns")


class DockerHardeningState(State):
    task_name:         str   = Field(default="",  description="Current task name")
    cumulative_reward: float = Field(default=0.0, description="Accumulated reward this episode")
    cycle_count:       int   = Field(default=0,   description="Number of patch cycles completed")
