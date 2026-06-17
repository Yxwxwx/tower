"""Supervisor agent — central task orchestrator.

Exposes register() → AgentRegistration for the agent framework.
"""
from agents.supervisor.agent import supervisor_graph, register

__all__ = ["supervisor_graph", "register"]
