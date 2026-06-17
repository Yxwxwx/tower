"""RunStateStore — the single source of truth for one run.

All agents READ from this store. Only the owning writer can WRITE.
State transitions are validated against legal transition tables.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from contracts.agent_task import TaskStatus, AgentName
from contracts.monitor_task import MonitorEvent
from tower.state.artifact_registry import ArtifactRecord
from tower.state.job_registry import JobRecord

# ═══════════════════════════════════════════════════════════════════
# State transition tables — the ONLY legal transitions
# ═══════════════════════════════════════════════════════════════════

AGENT_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING:   {TaskStatus.RUNNING},
    TaskStatus.RUNNING:   {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.NEEDS_HUMAN},
    TaskStatus.FAILED:    {TaskStatus.RETRYING, TaskStatus.ABANDONED},
    TaskStatus.RETRYING:  {TaskStatus.RUNNING},
    TaskStatus.ABANDONED: set(),           # terminal
    TaskStatus.DONE:      set(),           # terminal
    TaskStatus.NEEDS_HUMAN: set(),         # terminal
}


class IllegalStateTransition(Exception):
    """Raised when a state transition violates the transition table."""
    pass


def validate_transition(
    from_status: TaskStatus,
    to_status: TaskStatus,
    transition_map: dict,
) -> bool:
    """All state writes MUST pass this check."""
    allowed = transition_map.get(from_status, set())
    if to_status not in allowed:
        raise IllegalStateTransition(
            f"Illegal transition: {from_status.value} → {to_status.value}"
        )
    return True


# ═══════════════════════════════════════════════════════════════════
# Supervisor decision log
# ═══════════════════════════════════════════════════════════════════


class SupervisorDecision(BaseModel):
    """Record of every supervisor routing/retry/escalation decision."""
    timestamp: datetime = Field(default_factory=datetime.now)
    decision_type: Literal["dispatch", "retry", "escalate", "complete", "abort"]
    agent: AgentName | None = None
    task_id: str = ""
    reason: str = ""
    new_params: dict | None = None


# ═══════════════════════════════════════════════════════════════════
# RunState — the full state of one user request
# ═══════════════════════════════════════════════════════════════════


class RunState(BaseModel):
    """Global state for one user request.

    This is the single source of truth. All agents read from it.
    Only the designated writer may modify each field (see write ownership table).
    """
    run_id: str
    trace_id: str
    task: str                              # original user task
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Sub-task status: {"gaussian-001": DONE, "pyscf-001": RUNNING}
    agent_tasks: dict[str, TaskStatus] = Field(default_factory=dict)

    # Artifact index: artifact_id → ArtifactRecord
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)

    # Job index: job_id → JobRecord
    jobs: dict[str, JobRecord] = Field(default_factory=dict)

    # Supervisor decision log (append-only)
    decisions: list[SupervisorDecision] = Field(default_factory=list)

    # Monitor event log (append-only, monotonic IDs)
    event_log: list[MonitorEvent] = Field(default_factory=list)
    last_processed_event_id: int = 0       # event watermark

    # Agent heartbeats: agent_name → last heartbeat time
    agent_heartbeats: dict[str, datetime] = Field(default_factory=dict)

    # Execution mode
    mode: Literal["live", "replay", "dry_run"] = "live"

    # Write ownership (for runtime enforcement, not serialized to checkpoint)
    # OWNER_MAP: field_name → writer_agent_name
    # See Section 6.6 of the architecture spec.
    OWNER_MAP: dict[str, str] = {
        "status": "supervisor",
        "agent_tasks": "supervisor",
        "artifacts": "domain_agent",
        "jobs": "hpc",
        "decisions": "supervisor",
        "event_log": "monitor",
        "last_processed_event_id": "supervisor",
        "agent_heartbeats": "domain_agent",
    }


# ═══════════════════════════════════════════════════════════════════
# RunStateStore — CRUD operations with validation
# ═══════════════════════════════════════════════════════════════════


class RunStateStore:
    """In-memory store for the active run state.

    In production, this is backed by LangGraph's checkpoint (AsyncPostgresSaver).
    For MVP, we use an in-memory dict keyed by run_id.

    All write operations validate state transitions.
    """

    def __init__(self):
        self._store: dict[str, RunState] = {}

    # ── CRUD ──

    async def create(self, run_state: RunState) -> RunState:
        """Create a new run."""
        self._store[run_state.run_id] = run_state
        return run_state

    async def get(self, run_id: str) -> RunState | None:
        """Get run state by ID."""
        return self._store.get(run_id)

    async def update(self, run_state: RunState):
        """Replace the run state (after transition validation)."""
        self._store[run_state.run_id] = run_state

    # ── Agent task status ──

    async def update_agent_task(
        self, run_id: str, task_id: str,
        new_status: TaskStatus, writer: AgentName,
    ):
        """Update one agent task's status. Validates transition + ownership."""
        rs = await self._get_or_raise(run_id)
        old_status = rs.agent_tasks.get(task_id, TaskStatus.PENDING)

        # Validate transition
        validate_transition(old_status, new_status, AGENT_TASK_TRANSITIONS)

        # Validate ownership: only supervisor may change agent task status
        if writer != "supervisor":
            raise PermissionError(
                f"Only supervisor may change agent task status. "
                f"Got writer={writer}"
            )

        rs.agent_tasks[task_id] = new_status

    # ── Event log ──

    async def append_event(self, run_id: str, event: MonitorEvent, writer: AgentName):
        """Append a monitor event (monotonic ID, append-only)."""
        if writer != "monitor":
            raise PermissionError(f"Only monitor may append events. Got writer={writer}")

        rs = await self._get_or_raise(run_id)
        event.id = len(rs.event_log) + 1
        rs.event_log.append(event)

    async def read_new_events(self, run_id: str) -> list[MonitorEvent]:
        """Read events since the last processed watermark. Idempotent."""
        rs = await self._get_or_raise(run_id)
        new = [
            e for e in rs.event_log
            if e.id > rs.last_processed_event_id
        ]
        return sorted(new, key=lambda e: e.id)

    async def mark_events_processed(self, run_id: str, up_to_id: int):
        """Update the event watermark."""
        rs = await self._get_or_raise(run_id)
        rs.last_processed_event_id = max(rs.last_processed_event_id, up_to_id)

    # ── Heartbeat ──

    async def heartbeat(self, run_id: str, agent: AgentName):
        """Record an agent heartbeat."""
        rs = await self._get_or_raise(run_id)
        rs.agent_heartbeats[agent] = datetime.now()

    # ── Helpers ──

    async def _get_or_raise(self, run_id: str) -> RunState:
        rs = self._store.get(run_id)
        if rs is None:
            raise KeyError(f"RunState not found: {run_id}")
        return rs
