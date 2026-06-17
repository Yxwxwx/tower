"""Memory OS data models — MemoryRecord, MemoryType, RunContext."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Classification of long-term memory records."""

    TASK_PATTERN = "task_pattern"  # "Gaussian opt → PySCF CASSCF → Orca NEVPT2"
    CHEMICAL_INSIGHT = "chemical_insight"  # "N2 bond length ~1.1 Å"
    FAILURE_PATTERN = "failure_pattern"  # "SCF fails when maxiter < 100"
    TOOL_HEURISTIC = "tool_heuristic"  # "HPC memory underestimates if nprocs > 32"
    USER_PREFERENCE = "user_preference"  # "prefers B3LYP/def2SVP for organics"
    SUCCESSFUL_PIPELINE = "successful_pipeline"  # full run that completed


class MemoryRecord(BaseModel):
    """One knowledge record in the long-term memory store.

    Stored in AsyncPostgresStore. In MVP: key-value lookup by type.
    Post-MVP: pgvector embedding similarity search.
    """

    memory_id: str  # unique, hash(content)
    trace_id: str | None = None  # source run (None = manually added)
    content: str  # human-readable knowledge
    memory_type: MemoryType
    source_artifacts: list[str] = Field(default_factory=list)
    confidence: float = 1.0  # 1.0 = verified by successful run
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime | None = None
    access_count: int = 0
