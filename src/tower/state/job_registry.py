"""JobRegistry — HPC job state machine.

Tracks every Slurm job through its full lifecycle:
QUEUED → RUNNING → DONE/FAILED/TIMEOUT → RETRYING/ABANDONED
"""
from datetime import datetime

from pydantic import BaseModel, Field

from contracts.agent_task import JobStatus, AgentName

# ═══════════════════════════════════════════════════════════════════
# Transition table
# ═══════════════════════════════════════════════════════════════════

JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED:    {JobStatus.RUNNING},
    JobStatus.RUNNING:   {JobStatus.DONE, JobStatus.FAILED, JobStatus.TIMEOUT},
    JobStatus.FAILED:    {JobStatus.RETRYING, JobStatus.ABANDONED},
    JobStatus.TIMEOUT:   {JobStatus.RETRYING, JobStatus.ABANDONED},
    JobStatus.RETRYING:  {JobStatus.QUEUED},
    JobStatus.ABANDONED: set(),               # terminal
    JobStatus.DONE:      set(),               # terminal
}


# ═══════════════════════════════════════════════════════════════════
# JobRecord
# ═══════════════════════════════════════════════════════════════════


class JobRecord(BaseModel):
    """One HPC job tracked in the registry."""
    job_id: str                                # Slurm job ID
    agent: AgentName                           # responsible agent
    agent_task_id: str                         # which AgentTask this job belongs to
    slurm_artifact_id: str = ""                # artifact_id of final Slurm script
    submitted_at: datetime = Field(default_factory=datetime.now)
    last_polled_at: datetime | None = None
    status: JobStatus = JobStatus.QUEUED
    node: str = ""                             # assigned node (from squeue)
    exit_code: int | None = None
    error_category: str = ""                   # monitor fills this in
    retries: int = 0
    max_retries: int = 2
    nprocs: int = 0
    mem_per_cpu_mb: int = 0
    walltime_hours: int = 0


# ═══════════════════════════════════════════════════════════════════
# JobRegistry
# ═══════════════════════════════════════════════════════════════════


class JobRegistry:
    """Central job registry — part of RunStateStore.

    HPC agent creates jobs (QUEUED). Monitor updates status (RUNNING→DONE/FAILED).
    Supervisor reads status for retry decisions.
    """

    @staticmethod
    async def create(
        run_state,                       # RunState
        job_id: str,
        agent: AgentName,
        agent_task_id: str,
        slurm_artifact_id: str = "",
        nprocs: int = 0,
        mem_per_cpu_mb: int = 0,
        walltime_hours: int = 0,
    ) -> JobRecord:
        """Register a new job (status = QUEUED)."""
        record = JobRecord(
            job_id=job_id,
            agent=agent,
            agent_task_id=agent_task_id,
            slurm_artifact_id=slurm_artifact_id,
            status=JobStatus.QUEUED,
            nprocs=nprocs,
            mem_per_cpu_mb=mem_per_cpu_mb,
            walltime_hours=walltime_hours,
        )
        run_state.jobs[job_id] = record
        return record

    @staticmethod
    async def get(run_state, job_id: str) -> JobRecord | None:
        """Get a job record."""
        return run_state.jobs.get(job_id)

    @staticmethod
    async def update_status(
        run_state,
        job_id: str,
        new_status: JobStatus,
        writer: AgentName,
        **kwargs,
    ) -> JobRecord:
        """Update job status with transition validation.

        Args:
            writer: "hpc" or "monitor" — only these may update job status.
            **kwargs: extra fields to set (node, exit_code, error_category, etc.)
        """
        record = run_state.jobs.get(job_id)
        if record is None:
            raise KeyError(f"Job not found: {job_id}")

        old_status = record.status
        from tower.state.run_store import validate_transition
        validate_transition(old_status, new_status, JOB_TRANSITIONS)

        record.status = new_status
        record.last_polled_at = datetime.now()

        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        return record

    @staticmethod
    async def get_all_active(run_state) -> list[JobRecord]:
        """Get all jobs that are not in a terminal state."""
        terminal = {JobStatus.DONE, JobStatus.ABANDONED}
        return [j for j in run_state.jobs.values() if j.status not in terminal]
