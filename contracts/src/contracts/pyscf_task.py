"""PySCF agent contract — frozen v1.1.

Flexible Params/Result: the LLM extracts method, basis, functional, active space
from the task description. Results use dict-based mol/energy/converged/extra so
any PySCF calculation type is representable without schema changes.
"""
from pydantic import BaseModel, Field


class PySCFParams(BaseModel):
    """Supervisor → PySCF agent.

    The agent LLM infers method, basis, functional, active space, etc.
    from task_description. No hardcoded method/basis fields.
    """
    task_description: str = ""
    fchk_artifact_id: str = ""       # upstream Gaussian checkpoint (for CASSCF etc.)
    charge: int = 0
    spin: int = 0
    additional_info: dict = Field(default_factory=dict)
    resources: dict = Field(default_factory=dict)
    # {"omp_threads": 8, "memory_mb": 20000, "walltime_hours": 24,
    #  "modules": ["anaconda3-2024.10-1"], "conda_env": "pyscf_311",
    #  "partition_hint": "cpu32", "pythonpath": "/home/..."}


class PySCFResult(BaseModel):
    """PySCF agent → Supervisor.

    All four dict fields are filled by the LLM parser at runtime.
    No calculation type is hardcoded.
    """
    mol: dict = Field(default_factory=dict)
    # {"atom": [["N",0,0,0],["N",0,0,1.098]], "basis": "cc-pVDZ", "charge": 0, "spin": 0}

    energy: dict = Field(default_factory=dict)
    # {"scf": -109.42}  or  {"scf": -109.42, "mp2_total": -109.74}
    # or {"tddft_root0": 0.15, "tddft_root1": 0.18}

    converged: dict = Field(default_factory=dict)
    # {"scf": true, "casscf": true} — empty dict for methods without convergence

    extra: dict = Field(default_factory=dict)
    # Orbital energies, dipole, frequencies, wall_time, errors, etc.
