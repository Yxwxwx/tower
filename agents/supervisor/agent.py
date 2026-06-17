"""Supervisor agent — LangGraph StateGraph that orchestrates domain agents.

Architecture:
- supervisor_node: LLM plans the task, decides routing
- Each domain agent is a subgraph node
- Conditional edges route based on plan progress
- Human-in-the-loop via interrupt() for destructive/expensive actions
"""
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from contracts.agent_task import AgentTask, AgentResult, TaskStatus, AgentName
from tower_agent_kit.base import AgentRegistration, RetryPolicy
from agents.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════
# Supervisor state
# ═══════════════════════════════════════════════════════════════════

class SupervisorState(BaseModel):
    """Supervisor internal state — coordinates multi-agent execution.

    Not the same as RunState (which is the global truth source).
    SupervisorState tracks the current execution plan and progress.
    """
    # ── Conversation ──
    messages: list = Field(default_factory=list)

    # ── Task & Plan ──
    task: str = ""
    plan: list[str] = Field(default_factory=list)
    # plan = ["gaussian", "pyscf", "orca"]
    current_plan_index: int = 0

    # ── Agent results (collected during execution) ──
    agent_results: dict[str, Any] = Field(default_factory=dict)
    # {"gaussian": AgentResult[GaussianResult], ...}

    # ── Run linkage ──
    run_id: str = ""
    trace_id: str = ""

    # ── Control ──
    task_complete: bool = False
    needs_human: bool = False
    final_response: str = ""


# ═══════════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════════

def plan_node(state: SupervisorState) -> dict:
    """Decompose user task into an ordered agent plan.

    For MVP: uses the LLM to parse the task and output a plan.
    The plan is a list of agent names in execution order.

    Example: "N2 NEVPT2" → ["gaussian", "pyscf", "orca"]
    """
    # For MVP, we do heuristic planning (LLM integration comes later)
    task_lower = state.task.lower()

    plan = []
    if any(kw in task_lower for kw in ["nevpt2", "casscf", "coupled cluster", "ccsd"]):
        plan = ["gaussian", "pyscf", "orca"]
    elif any(kw in task_lower for kw in ["opt", "optimization", "hf", "dft"]):
        plan = ["gaussian"]
    elif any(kw in task_lower for kw in ["energy", "single point", "sp"]):
        plan = ["gaussian"]
    else:
        # Unknown task type — default to gaussian only
        plan = ["gaussian"]

    # Add hpc if cluster submission is needed
    if any(kw in task_lower for kw in ["hpc", "slurm", "submit", "cluster"]):
        plan.append("hpc")

    return {
        "plan": plan,
        "current_plan_index": 0,
        "messages": [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=state.task),
            AIMessage(content=f"Plan: {' → '.join(plan)}"),
        ],
        "task_complete": len(plan) == 0,
    }


def route_to_agent(state: SupervisorState) -> dict:
    """Determine which agent to dispatch next based on plan progress."""
    if state.current_plan_index >= len(state.plan):
        return {"task_complete": True}
    return {}


def synthesize_result(state: SupervisorState) -> dict:
    """Synthesize final response from all agent results."""
    results_text = []
    for agent_name, result in state.agent_results.items():
        if hasattr(result, "status"):
            results_text.append(f"- {agent_name}: {result.status}")
        else:
            results_text.append(f"- {agent_name}: done")

    final = f"Task completed: {state.task}\n\n" + "\n".join(results_text)

    return {
        "final_response": final,
        "task_complete": True,
    }


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════

def route_after_plan(state: SupervisorState) -> Literal["end", "dispatch"]:
    """After planning: if empty plan → end, else → dispatch first agent."""
    if state.task_complete or len(state.plan) == 0:
        return "end"
    return "dispatch"


def route_after_agent(state: SupervisorState) -> Literal["next", "synthesize"]:
    """After an agent returns: more agents? → next. All done? → synthesize."""
    next_idx = state.current_plan_index
    if next_idx >= len(state.plan):
        return "synthesize"
    return "next"


# ═══════════════════════════════════════════════════════════════════
# Graph compilation
# ═══════════════════════════════════════════════════════════════════

def build_supervisor_graph() -> StateGraph:
    """Build the supervisor StateGraph.

    Topology:
        START → plan → [dispatch to agent] → [agent returns] → synthesize → END
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("plan", plan_node)
    graph.add_node("synthesize", synthesize_result)

    graph.add_edge(START, "plan")

    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"end": END, "dispatch": "synthesize"},  # MVP: plan → synthesize directly
    )

    graph.add_edge("synthesize", END)

    return graph


# ═══════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════

supervisor_graph = build_supervisor_graph().compile()


def register() -> AgentRegistration:
    """Expose agent metadata for the framework."""
    return AgentRegistration(
        name="supervisor",
        subgraph=supervisor_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=0),
        timeout_s=300,  # supervisor LLM calls should be fast
        dependencies=set(),
        description="Central task orchestrator — decomposes tasks, routes to domain agents, synthesizes results",
    )
