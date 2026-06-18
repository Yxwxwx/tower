"""Monitor agent contract — v1.1.

Real Slurm job polling via sacct/squeue, log reading, and result delivery.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from contracts.agent_task import AgentName


class MonitorParams(BaseModel):
    """Supervisor → Monitor agent."""
    watchlist: dict[str, str] = Field(default_factory=dict)
    # {"232164": "pyscf"} — job_id → agent_name
    run_dir: str = ""                              # shared job directory
    log_path: str = ""                              # relative to run_dir, set by supervisor
    poll_interval_s: int = 10
    max_wait_s: int = 86400                        # 24h default


class MonitorEvent(BaseModel):
    """Monitor writes these to state.event_log."""
    id: int = 0
    job_id: str = ""
    agent: AgentName = "supervisor"
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal[
        "JOB_STARTED", "JOB_RUNNING", "JOB_DONE",
        "JOB_FAILED", "JOB_TIMEOUT", "JOB_OOM",
    ] = "JOB_STARTED"
    log_snippet: str = ""
    error_category: Literal[
        "scf_not_converged", "oom", "mpi_error",
        "timeout", "unknown", "",
    ] = ""
    suggestion: str = ""


class MonitorResult(BaseModel):
    """Monitor agent → Supervisor."""
    events: list[MonitorEvent] = Field(default_factory=list)
    completed_jobs: list[str] = Field(default_factory=list)
    failed_jobs: list[str] = Field(default_factory=list)
    log_content: str = ""                           # actual log file content
    summary: str = ""
