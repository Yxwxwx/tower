"""Gaussian agent contract — frozen v1.0."""
from typing import Literal

from pydantic import BaseModel, Field


class GaussianParams(BaseModel):
    """Supervisor → GaussianAgent."""
    method: str = "B3LYP"
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    job_type: Literal["opt", "opt+freq", "sp", "irc"] = "opt"
    additional_keywords: dict[str, str] = Field(default_factory=dict)
    memory_mb: int = 4000
    nprocs: int = 8
    checkpoint_from: str | None = None       # artifact_id of upstream .fchk


class GaussianResult(BaseModel):
    """GaussianAgent → Supervisor."""
    energy: float | None = None              # final single-point energy (Hartree)
    dipole: list[float] | None = None
    n_imag_freq: int = 0                     # 0 = true minimum
    converged: bool = False
    checkpoint_path: str | None = None       # artifact_id of produced .fchk
    wall_time_s: float = 0.0
