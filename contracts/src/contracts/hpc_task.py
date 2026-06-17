"""HPC agent contract — frozen v1.0."""
from datetime import datetime

from pydantic import BaseModel, Field

from contracts.agent_task import AgentName


class JobRequest(BaseModel):
    """Single agent's compute requirements — collected by Supervisor, sent to HPC."""
    agent: AgentName
    input_file_artifact_id: str = ""         # artifact_id of the agent's input file
    rough_slurm_artifact_id: str = ""        # artifact_id of rough Slurm template
    mem_per_cpu_mb: int = 4000
    nprocs: int = 8
    walltime_hours: int = 24


class HPCParams(BaseModel):
    """Supervisor → HPCAgent."""
    jobs: list[JobRequest] = Field(default_factory=list)
    partition: str = "compute"
    account: str = "default"
    email_on_fail: str | None = None


class HPCResult(BaseModel):
    """HPCAgent → Supervisor."""
    job_ids: dict[str, str] = Field(default_factory=dict)
    # {"gaussian": "12345", "orca": "12346"}
    submitted_at: datetime | None = None
    node_assignment: dict[str, str] = Field(default_factory=dict)
    # {"gaussian": "node03"}
    final_slurm_artifact_ids: dict[str, str] = Field(default_factory=dict)
    # {"gaussian": "artifact-xyz"}
