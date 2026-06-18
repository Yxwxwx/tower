"""Universal agent communication types.

FROZEN v1.0 — breaking changes require contracts/v2/.

All agents communicate exclusively through AgentTask[T] → AgentResult[T].
No side-channel communication. No raw file paths (use Artifact).
"""
from datetime import datetime
from enum import Enum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════
# Agent identity
# ═══════════════════════════════════════════════════════════════════

AgentName = Literal["supervisor", "gaussian", "pyscf", "orca", "hpc", "monitor"]


# ═══════════════════════════════════════════════════════════════════
# Status enums
# ═══════════════════════════════════════════════════════════════════


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"
    ABANDONED = "abandoned"
    NEEDS_HUMAN = "needs_human"


class ArtifactStatus(str, Enum):
    PRODUCING = "producing"
    READY = "ready"
    CONSUMED = "consumed"
    STALE = "stale"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    ABANDONED = "abandoned"


# ═══════════════════════════════════════════════════════════════════
# Artifact — produced file/data with lifecycle tracking
# ═══════════════════════════════════════════════════════════════════


class Artifact(BaseModel):
    """An agent-produced file or data object.

    Artifacts are immutable after creation. Retry → new artifact_id.
    Agents reference artifacts by artifact_id, never by raw path.
    """
    artifact_id: str = ""                     # globally unique, set by registry
    path: str                                 # filesystem path
    type: Literal["fchk", "json", "log", "slurm", "inp", "gjf", "hess"]
    description: str = ""
    content_hash: str = ""                    # sha256, set by registry on registration
    producer_agent: AgentName | None = None
    producer_task_id: str = ""
    status: ArtifactStatus = ArtifactStatus.PRODUCING
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class ArtifactRef(BaseModel):
    """Lightweight reference to an artifact — used in AgentTask.artifacts_in."""
    artifact_id: str
    type: str = ""
    description: str = ""


# ═══════════════════════════════════════════════════════════════════
# Retry policy
# ═══════════════════════════════════════════════════════════════════


class RetryPolicy(BaseModel):
    """Declared by each agent at registration."""
    max_retries: int = 2
    is_idempotent: bool = True
    requires_cleanup_before_retry: bool = False
    cleanup_steps: list[str] = Field(default_factory=list)
    backoff_s: int = 30
    escalate_after_max: bool = True


# ═══════════════════════════════════════════════════════════════════
# AgentTask / AgentResult — the universal communication protocol
# ═══════════════════════════════════════════════════════════════════

TParams = TypeVar("TParams")
TResult = TypeVar("TResult")


class AgentTask(BaseModel, Generic[TParams]):
    """Supervisor → Agent task input.

    This is the ONLY way a task is dispatched. Every agent MUST accept this.
    """
    task_id: str                              # globally unique
    trace_id: str                             # end-to-end trace identifier
    parent_run_id: str = ""                   # root run ID from supervisor
    goal: str = ""                            # natural language goal
    agent: AgentName                          # target agent
    params: TParams                           # agent-specific parameters (strongly typed)
    artifacts_in: list[ArtifactRef] = Field(default_factory=list)
    schema_version: str = "v1"                # contract version
    max_retries: int = 2
    deadline: datetime | None = None


class AgentResult(BaseModel, Generic[TResult]):
    """Agent → Supervisor task output.

    This is the ONLY way an agent reports back. Every agent MUST return this.
    """
    task_id: str
    trace_id: str
    status: TaskStatus
    agent: AgentName
    artifacts_out: list[Artifact] = Field(default_factory=list)
    data: TResult | None = None               # agent-specific structured output
    errors: list[str] = Field(default_factory=list)
    next_action: str = ""                     # suggested correction or next step
    retries_used: int = 0
    wall_time_s: float = 0.0
