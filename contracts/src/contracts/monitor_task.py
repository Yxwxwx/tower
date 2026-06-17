"""Monitor agent contract — frozen v1.0."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from contracts.agent_task import AgentName


class MonitorParams(BaseModel):
    """Supervisor → MonitorAgent."""
    watchlist: dict[str, str] = Field(default_factory=dict)
    # {"12345": "gaussian", "12346": "orca"}
    poll_interval_s: int = 60
    max_watch_hours: int = 48


class MonitorEvent(BaseModel):
    """MonitorAgent writes these to RunStateStore.event_log (append-only)."""
    id: int = 0                              # monotonic, set by event_log
    job_id: str = ""
    agent: AgentName = "supervisor"
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal[
        "JOB_STARTED", "JOB_DONE", "JOB_FAILED",
        "JOB_TIMEOUT", "JOB_OOM", "AGENT_HEARTBEAT_LOST",
    ] = "JOB_STARTED"
    log_snippet: str = ""
    error_category: Literal[
        "scf_not_converged", "oom", "mpi_error",
        "timeout", "unknown", "",
    ] = ""
    suggestion: str = ""


class MonitorResult(BaseModel):
    """MonitorAgent → Supervisor."""
    events: list[MonitorEvent] = Field(default_factory=list)
    failed_jobs: list[str] = Field(default_factory=list)
    completed_jobs: list[str] = Field(default_factory=list)
    summary: str = ""
