"""Tower Agent Kit — lightweight scaffold for building tower agents.

Every agent inherits from BaseAgentState and exposes register() → AgentRegistration.
This eliminates glue code duplication across agents.
"""
from tower_agent_kit.base import (
    BaseAgentState,
    AgentRegistration,
)

__all__ = ["BaseAgentState", "AgentRegistration"]
