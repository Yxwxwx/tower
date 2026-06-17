"""Base classes for all tower agents.

Every agent:
1. Inherits from BaseAgentState for its internal state.
2. Exposes register() → AgentRegistration at module level.
3. Compiles to a LangGraph CompiledStateGraph.
"""
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from contracts.agent_task import (
    AgentName,
    AgentTask,
    AgentResult,
    RetryPolicy,
    TaskStatus,
)

# ═══════════════════════════════════════════════════════════════════
# Base agent state
# ═══════════════════════════════════════════════════════════════════

TParams = TypeVar("TParams")
TResult = TypeVar("TResult")


class BaseAgentState(BaseModel, Generic[TParams, TResult]):
    """Every agent's internal state inherits from this.

    Agents MAY add private fields. External agents only see AgentResult.
    """
    # ── Universal fields (set by supervisor on dispatch) ──
    trace_id: str = ""
    task_id: str = ""
    task: AgentTask[TParams] | None = None

    # ── Agent output (returned to supervisor) ──
    status: TaskStatus = TaskStatus.PENDING
    result_data: TResult | None = None
    errors: list[str] = Field(default_factory=list)
    artifacts_out: list = Field(default_factory=list)

    # ── Execution tracking ──
    node_history: list[str] = Field(default_factory=list)
    retries_used: int = 0

    # ── Private scratchpad (not visible to supervisor) ──
    scratchpad: dict[str, Any] = Field(default_factory=dict)

    def to_agent_result(self, agent: AgentName) -> AgentResult[TResult]:
        """Convert internal state to the standard AgentResult protocol."""
        return AgentResult[TResult](
            task_id=self.task_id,
            trace_id=self.trace_id,
            status=self.status,
            agent=agent,
            artifacts_out=self.artifacts_out,
            data=self.result_data,
            errors=self.errors,
            retries_used=self.retries_used,
        )


# ═══════════════════════════════════════════════════════════════════
# Agent registration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AgentRegistration:
    """Every agent exposes register() → AgentRegistration.

    Supervisor reads this to know how to dispatch to the agent.
    """
    name: AgentName                              # "gaussian" | "pyscf" | ...
    subgraph: StateGraph                        # compiled LangGraph subgraph
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: int = 3600
    dependencies: set[AgentName] = field(default_factory=set)
    description: str = ""
