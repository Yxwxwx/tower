"""Gaussian agent — HF/DFT optimization and wavefunction output.

Exposes register() → AgentRegistration.
"""
from agents.gaussian.agent import gaussian_graph, register

__all__ = ["gaussian_graph", "register"]
