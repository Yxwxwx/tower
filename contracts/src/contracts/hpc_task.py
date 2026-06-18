"""HPC agent contract — v1.1.

Real cluster submission: analyze artifacts, generate Slurm, sbatch.
Falls back to local bash execution when no Slurm is available.
"""
from pydantic import BaseModel, Field
from contracts.agent_task import AgentName


class JobRequest(BaseModel):
    """Single agent's compute requirements — collected by Supervisor, sent to HPC."""
    agent: AgentName
    input_file_artifact_id: str = ""
    rough_slurm_artifact_id: str = ""
    run_command_artifact_id: str = ""
    resources: dict = Field(default_factory=dict)
    # {"omp_threads": 8, "memory_mb": 20000, "walltime_hours": 24,
    #  "modules": [...], "conda_env": "...", "partition_hint": "cpu32",
    #  "pythonpath": "..."}


class HPCParams(BaseModel):
    """Supervisor → HPC agent."""
    jobs: list[JobRequest] = Field(default_factory=list)
    run_command: str = ""              # extracted from domain agent's artifacts
    partition: str = "compute"
    account: str = "default"
    email_on_fail: str | None = None
    nvme_path: str = "/nvme"
    scratch_path: str = ""


class HPCResult(BaseModel):
    """HPC agent → Supervisor."""
    job_ids: dict[str, str] = Field(default_factory=dict)
    # {"pyscf": "12345"} or {"pyscf": "local-abc123"}
    job_state: str = ""                # "RUNNING" | "COMPLETED" | "FAILED"
    slurm_artifact_id: str = ""        # artifact ID of the generated Slurm script
    log_artifact_id: str = ""          # artifact ID of computation log
    tmpdir_artifact_id: str = ""       # artifact ID of TMPDIR tarball
    submitted_at: str = ""
    node_assignment: dict[str, str] = Field(default_factory=dict)
    final_slurm_artifact_ids: dict[str, str] = Field(default_factory=dict)
