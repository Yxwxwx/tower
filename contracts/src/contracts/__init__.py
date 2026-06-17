"""Tower Contracts — frozen Pydantic schemas for all agent communication.

This is an independent pip-installable package with ZERO tower dependencies.
All agent-to-agent communication uses these models.

Version: v1.0 — FROZEN. Breaking changes require contracts/v2/.
"""
from contracts.agent_task import (
    AgentTask,
    AgentResult,
    Artifact,
    ArtifactRef,
    TaskStatus,
    AgentName,
    RetryPolicy,
)
from contracts.gaussian_task import GaussianParams, GaussianResult
from contracts.pyscf_task import PySCFParams, PySCFResult
from contracts.orca_task import OrcaParams, OrcaResult
from contracts.hpc_task import HPCParams, HPCResult, JobRequest
from contracts.monitor_task import MonitorParams, MonitorResult, MonitorEvent

__all__ = [
    # Universal
    "AgentTask",
    "AgentResult",
    "Artifact",
    "ArtifactRef",
    "TaskStatus",
    "AgentName",
    "RetryPolicy",
    # Gaussian
    "GaussianParams",
    "GaussianResult",
    # PySCF
    "PySCFParams",
    "PySCFResult",
    # Orca
    "OrcaParams",
    "OrcaResult",
    # HPC
    "HPCParams",
    "HPCResult",
    "JobRequest",
    # Monitor
    "MonitorParams",
    "MonitorResult",
    "MonitorEvent",
]
