"""HPC agent — cluster resource query, Slurm refinement, job submission."""
from agents.hpc.agent import hpc_graph, register

__all__ = ["hpc_graph", "register"]
