"""PySCF agent contract — frozen v1.0."""
from pydantic import BaseModel, Field


class PySCFParams(BaseModel):
    """Supervisor → PySCFAgent."""
    fchk_artifact_id: str = ""               # artifact_id of Gaussian .fchk
    n_active_electrons: int = 0
    n_active_orbitals: int = 0
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    cas_irrep: str | None = None
    max_memory_mb: int = 8000


class PySCFResult(BaseModel):
    """PySCFAgent → Supervisor."""
    active_orbitals: list[int] = Field(default_factory=list)
    casscf_energy: float | None = None       # CASSCF energy (Hartree)
    natural_occupations: list[float] = Field(default_factory=list)
    orbital_info_artifact_id: str = ""       # artifact_id of active_orbitals.json
