"""PySCF agent contract — frozen v1.0."""
from typing import Literal

from pydantic import BaseModel, Field


class PySCFParams(BaseModel):
    """Supervisor → PySCFAgent.

    For RHF/DFT: set job_type, basis, charge, spin. Leave CAS fields at 0.
    For CASSCF: set n_active_electrons > 0 and n_active_orbitals > 0.
    """
    job_type: Literal["RHF", "UHF", "RDFT", "UDFT", "CASSCF"] = "RHF"
    fchk_artifact_id: str = ""               # artifact_id of Gaussian .fchk
    n_active_electrons: int = 0              # > 0 triggers CASSCF path
    n_active_orbitals: int = 0
    basis: str = "def2SVP"
    functional: str = ""                     # for DFT: "b3lyp", "pbe0", etc.
    charge: int = 0
    spin: int = 1
    cas_irrep: str | None = None
    max_memory_mb: int = 8000


class PySCFResult(BaseModel):
    """PySCFAgent → Supervisor."""
    scf_energy: float | None = None          # RHF/DFT energy (Hartree)
    casscf_energy: float | None = None       # CASSCF energy (Hartree)
    active_orbitals: list[int] = Field(default_factory=list)
    natural_occupations: list[float] = Field(default_factory=list)
    converged: bool = False
    orbital_info_artifact_id: str = ""       # artifact_id of active_orbitals.json
    wall_time_s: float = 0.0
