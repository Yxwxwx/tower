"""Working memory — intra-run context buffer.

NOT persisted. Lives only for the duration of one Run.
Rebuildable from ExecutionMemory (checkpoint) if crash recovery needed.
"""
from typing import Any

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    """Intra-run working memory.

    Holds information that flows between agents within a single run
    but does NOT need full ArtifactRecord tracking.

    NOT authoritative — RunStateStore is the single source of truth.
    """
    trace_id: str = ""
    run_id: str = ""

    # Supervisor reasoning chain (for debugging / replay)
    supervisor_plan: list[str] = Field(default_factory=list)
    supervisor_rationale: str = ""

    # Scratchpad: agents can stash intermediate results here
    # {"scf_converged": True, "active_orbital_indices": [3,4,5,6]}
    scratchpad: dict[str, Any] = Field(default_factory=dict)

    # Agent-to-agent handoff notes
    # {"gaussian→pyscf": "use orbitals 3-6 as active space"}
    handoff_notes: dict[str, str] = Field(default_factory=dict)
