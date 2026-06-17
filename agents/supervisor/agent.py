"""Supervisor agent — actually dispatches to sub-agents and collects results.

Topology:
    START → plan → dispatch_agent → [agent executes] → collect_result
           ↑                                          ↓
           └────────── next agent ←──── route ─────────┘
                                                      ↓
                                                  synthesize → END
"""
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from contracts.agent_task import (
    AgentTask, AgentResult, ArtifactRef, TaskStatus, AgentName,
)
from contracts.gaussian_task import GaussianParams
from contracts.pyscf_task import PySCFParams
from contracts.orca_task import OrcaParams
from tower_agent_kit.base import AgentRegistration, RetryPolicy


# ═══════════════════════════════════════════════════════════════════
# Agent subgraph registry — loaded at startup, used by dispatch node
# ═══════════════════════════════════════════════════════════════════

_agent_subgraphs: dict[str, Any] = {}


def set_agent_registry(subgraphs: dict[str, Any]):
    """Called by CLI at startup to register all agent subgraphs."""
    _agent_subgraphs.clear()
    _agent_subgraphs.update(subgraphs)


# ═══════════════════════════════════════════════════════════════════
# Supervisor state
# ═══════════════════════════════════════════════════════════════════

class SupervisorState(BaseModel):
    task: str = ""
    run_id: str = ""
    trace_id: str = ""

    # Plan: ordered list of agent names
    plan: list[str] = Field(default_factory=list)
    current_plan_index: int = 0

    # Results from each dispatched agent
    agent_results: dict[str, Any] = Field(default_factory=dict)
    # Latest artifacts (passed as artifacts_in to next agent)
    pending_artifacts: list[dict] = Field(default_factory=list)

    # Control
    task_complete: bool = False
    needs_human: bool = False
    final_response: str = ""


# ═══════════════════════════════════════════════════════════════════
# Node: plan
# ═══════════════════════════════════════════════════════════════════

def plan_node(state: SupervisorState) -> dict:
    """Decompose user task into ordered agent plan.

    MVP: heuristic keyword matching. Post-MVP: LLM-based decomposition.
    """
    task_lower = state.task.lower()

    plan = []
    if any(kw in task_lower for kw in ["nevpt2", "casscf", "coupled cluster", "ccsd"]):
        plan = ["gaussian", "pyscf", "orca"]
    elif any(kw in task_lower for kw in ["opt", "optimization", "hf", "dft"]):
        plan = ["gaussian"]
    else:
        plan = ["gaussian"]

    # HPC runs in parallel after last chemical agent
    if any(kw in task_lower for kw in ["hpc", "slurm", "submit", "cluster"]):
        plan.append("hpc")

    return {
        "plan": plan,
        "current_plan_index": 0,
        "task_complete": len(plan) == 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Node: dispatch_agent — actually invokes the sub-agent's subgraph
# ═══════════════════════════════════════════════════════════════════

def dispatch_agent(state: SupervisorState) -> dict:
    """Invoke the current agent's subgraph with the right AgentTask.

    Builds AgentTask with artifacts_in from the previous agent's output.
    Stores the AgentResult in state.agent_results.
    """
    idx = state.current_plan_index
    agent_name = state.plan[idx]

    subgraph = _agent_subgraphs.get(agent_name)
    if subgraph is None:
        return {
            "agent_results": {
                **state.agent_results,
                agent_name: AgentResult(
                    task_id=f"{state.run_id}-{agent_name}",
                    trace_id=state.trace_id,
                    status=TaskStatus.FAILED,
                    agent=agent_name,
                    errors=[f"Agent '{agent_name}' not found in registry"],
                ),
            },
            "current_plan_index": idx + 1,
        }

    # Build AgentTask with upstream artifacts
    artifacts_in = [
        ArtifactRef(artifact_id=a["artifact_id"], type=a.get("type", ""))
        for a in state.pending_artifacts
    ]

    params = _build_params(agent_name, state)
    task = AgentTask(
        task_id=f"{state.run_id}-{agent_name}",
        trace_id=state.trace_id,
        parent_run_id=state.run_id,
        goal=_goal_for(agent_name, state.task),
        agent=agent_name,
        params=params,
        artifacts_in=artifacts_in,
    )

    # === INVOKE THE AGENT ===
    agent_state = {
        "task": task,
        "task_id": task.task_id,
        "trace_id": state.trace_id,
        "status": TaskStatus.PENDING,
    }
    result_state = subgraph.invoke(agent_state)
    agent_result = result_state.get("agent_result")
    # =========================

    # Pass artifacts from this agent to the next one
    new_artifacts = []
    if agent_result and agent_result.artifacts_out:
        new_artifacts = [
            {"artifact_id": a.artifact_id, "type": a.type}
            for a in agent_result.artifacts_out
            if a.artifact_id
        ]

    return {
        "agent_results": {**state.agent_results, agent_name: agent_result},
        "current_plan_index": idx + 1,
        "pending_artifacts": new_artifacts,
        "needs_human": (
            agent_result is not None and agent_result.status == TaskStatus.NEEDS_HUMAN
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Node: synthesize
# ═══════════════════════════════════════════════════════════════════

def synthesize_result(state: SupervisorState) -> dict:
    """Build final response from all agent results."""
    lines = []
    for name in state.plan:
        result = state.agent_results.get(name)
        if result is None:
            lines.append(f"- {name}: [not executed]")
        elif hasattr(result, "status"):
            status_icon = "✓" if result.status == TaskStatus.DONE else "✕"
            lines.append(f"- {name}: {status_icon} {result.status.value}")
            if hasattr(result, "data") and result.data:
                lines.append(f"  {_summarize_data(result.data)}")
        else:
            lines.append(f"- {name}: done")

    final = f"Task: {state.task}\n\n" + "\n".join(lines)
    return {"final_response": final, "task_complete": True}


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════

def route_after_plan(state: SupervisorState) -> Literal["dispatch", "end"]:
    if state.task_complete or len(state.plan) == 0:
        return "end"
    return "dispatch"


def route_after_dispatch(state: SupervisorState) -> Literal["dispatch", "synthesize"]:
    """After collecting one agent's result: more? → dispatch next. Done? → synthesize."""
    if state.needs_human:
        return "synthesize"
    if state.current_plan_index >= len(state.plan):
        return "synthesize"
    return "dispatch"


# ═══════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════

def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(SupervisorState)

    graph.add_node("plan", plan_node)
    graph.add_node("dispatch_agent", dispatch_agent)
    graph.add_node("synthesize", synthesize_result)

    graph.add_edge(START, "plan")

    graph.add_conditional_edges(
        "plan", route_after_plan,
        {"dispatch": "dispatch_agent", "end": END},
    )

    graph.add_conditional_edges(
        "dispatch_agent", route_after_dispatch,
        {"dispatch": "dispatch_agent", "synthesize": "synthesize"},
    )

    graph.add_edge("synthesize", END)

    return graph


supervisor_graph = build_supervisor_graph().compile()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _build_params(agent_name: str, state: SupervisorState) -> Any:
    """Build agent-specific params. MVP: defaults. Domain devs customize."""
    if agent_name == "gaussian":
        return GaussianParams(method="B3LYP", basis="def2SVP")
    elif agent_name == "pyscf":
        return PySCFParams(n_active_electrons=6, n_active_orbitals=6)
    elif agent_name == "orca":
        return OrcaParams()
    return {}


def _goal_for(agent_name: str, task: str) -> str:
    goals = {
        "gaussian": f"HF optimization for: {task}",
        "pyscf": f"Orbital selection + CASSCF for: {task}",
        "orca": f"NEVPT2 computation for: {task}",
        "hpc": f"Slurm generation + job submission for: {task}",
        "monitor": f"Monitor jobs for: {task}",
    }
    return goals.get(agent_name, task)


def _summarize_data(data) -> str:
    """Brief summary of agent result data."""
    if hasattr(data, "energy") and data.energy is not None:
        return f"E = {data.energy:.6f} Ha"
    if hasattr(data, "casscf_energy") and data.casscf_energy is not None:
        return f"E(CASSCF) = {data.casscf_energy:.6f} Ha"
    return ""


# ═══════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════

def register() -> AgentRegistration:
    return AgentRegistration(
        name="supervisor",
        subgraph=supervisor_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=0),
        timeout_s=300,
        dependencies=set(),
        description="Central task orchestrator — plans, dispatches agents, collects results, synthesizes",
    )
