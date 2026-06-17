"""Orca agent contract — frozen v1.0."""
from typing import Literal

from pydantic import BaseModel


class OrcaParams(BaseModel):
    """Supervisor → OrcaAgent."""
    orbital_info_artifact_id: str = ""       # artifact_id from PySCF
    method: Literal["NEVPT2", "DLPNO-CCSD(T)", "CCSD(T)"] = "NEVPT2"
    basis: str = "def2-TZVP"
    charge: int = 0
    spin: int = 1
    memory_mb: int = 16000
    nprocs: int = 16


class OrcaResult(BaseModel):
    """OrcaAgent → Supervisor."""
    energy: float | None = None              # NEVPT2/CC energy (Hartree)
    energy_correction: float | None = None
    converged: bool = False
    log_artifact_id: str = ""                # artifact_id of .log
    wall_time_s: float = 0.0
