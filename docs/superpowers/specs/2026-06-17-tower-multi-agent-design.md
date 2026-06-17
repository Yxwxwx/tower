# Tower Multi-Agent System — Architecture Design Spec

**Date:** 2026-06-17
**Status:** Approved
**Author:** Sun Xinyu

## Table of Contents

1. [Three-Layer Architecture](#1-three-layer-architecture)
2. [Agent Contract Schema](#2-agent-contract-schema)
3. [Supergraph — LangGraph Topology](#3-supergraph--langgraph-topology)
4. [Agent Responsibility Boundaries & I/O Contracts](#4-agent-responsibility-boundaries--io-contracts)
5. [File Structure — Independent Package Layout](#5-file-structure--independent-package-layout)
6. [Execution Model & State Consistency](#6-execution-model--state-consistency)
7. [Execution Semantics & Fault Tolerance](#7-execution-semantics--fault-tolerance)

---

## 1. Three-Layer Architecture

### 1.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR AGENT                            │
│  只做：任务分解 → 路由 → 结果合成 → 重试/升级/人工确认           │
│  不做：输入文件编写、参数选择、Slurm 生成、日志解析              │
└────┬──────────┬──────────┬──────────┬───────────────────────────┘
     │          │          │          │
┌────▼──┐ ┌────▼──┐ ┌────▼──┐  ┌────▼──────────┐
│Gaussian│ │PySCF  │ │ Orca  │  │  HPC Agent    │  ← Infrastructure
│Agent  │ │Agent  │ │Agent  │  │  ·Resource-aware│
│       │ │       │ │       │  │  ·Slurm gen    │
│       │ │       │ │       │  │  ·Job submit   │
└───┬───┘ └──┬────┘ └───┬───┘  └──────┬────────┘
    │        │          │              │
    └────────┼──────────┼──────────────┘
             │          │
     ┌───────▼──────────▼───────┐
     │     Monitor Agent        │  ← Infrastructure (event-driven)
     │  ·Queue polling          │
     │  ·Log parsing            │
     │  ·Error classification   │
     │  ·Feedback to agent      │
     └──────────────────────────┘
             │
    ┌────────▼──────────────────────────────┐
    │       SHARED MCP TOOL LAYER           │
    │  infra-mcp: filesystem + template     │
    │            + slurm-gen                │
    │  hpc-mcp:   queue-status + log-parser │
    └───────────────────────────────────────┘
```

### 1.2 Layer Definitions

| Layer | Components | Responsibility |
|-------|-----------|---------------|
| **Supervisor** | `supervisor agent` | Task decomposition, routing, result synthesis, retry/escalation decisions |
| **Domain Agents** | `gaussian`, `pyscf`, `orca` | Software-specific input generation, computation workflow, result interpretation |
| **Infrastructure Agents** | `hpc`, `monitor` | Resource awareness, submission, monitoring, log parsing, failure feedback. Run in parallel with domain agents where possible. |
| **Shared MCP Layer** | `infra-mcp`, `hpc-mcp` | Standardized tool exposure: filesystem, template rendering, Slurm generation, queue queries, log parsing |

### 1.3 Example Workflow: "N2 NEVPT2"

```
Supervisor receives task
    │
    ├─ (1) → GaussianAgent:  "HF optimization, output converged wavefunction fchk"
    │        Returns: {status: DONE, artifacts: ["N2_opt.fchk"]}
    │
    ├─ (2) → PySCFAgent:     "Read fchk, select active orbitals, CASSCF"
    │        Returns: {status: DONE, artifacts: ["active_orbitals.json"]}
    │
    ├─ (3) → OrcaAgent:      "Read CASSCF result, generate NEVPT2 input, compute"
    │                          Parallel →
    ├─ (4) → HPCAgent:       "Collect all agent needs, generate Slurm, query cluster, submit"
    │
    └─ (5) → MonitorAgent:   "Watch all submitted jobs, on failure → feedback to agent"
```

### 1.4 Key Design Decisions

- **Supervisor never writes input files**: It only routes, sequences, and synthesizes.
- **HPC and Monitor are orchestration-layer parallelism**: They don't participate in chemical computation.
- **Monitor is event-driven**: Independent of the main handoff chain. Writes events to shared store; supervisor reads them at decision time.
- **Domain agents are sequential** (output of one = input of next). HPC/Monitor run in parallel where topology permits.

### 1.5 Design References

This architecture aligns with:
- **LangGraph supervisor pattern**: Central supervisor coordinates specialized worker subagents via tool-based handoff ([LangChain docs](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant))
- **MCP architecture**: Each MCP server independently exposes tools/resources/prompts, serving as the standardized tool layer ([Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture))
- **Anthropic multi-agent research system**: Multiple specialized agents divide exploration work, synthesize findings ([Anthropic engineering blog](https://www.anthropic.com/engineering/multi-agent-research-system))

---

## 2. Agent Contract Schema

### 2.1 Shared Protocol

All agents communicate through strongly-typed Pydantic models. The contract layer is an independent pip-installable package (`contracts/`).

```python
# contracts/agent_task.py

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Any, Generic, TypeVar, Literal

AgentName = Literal["gaussian", "pyscf", "orca", "hpc", "monitor"]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"
    ABANDONED = "abandoned"
    NEEDS_HUMAN = "needs_human"


class Artifact(BaseModel):
    """Agent-produced file or data object."""
    path: str
    type: Literal["fchk", "json", "log", "slurm", "inp", "gjf", "hess"]
    description: str = ""
    artifact_id: str = ""                    # globally unique
    content_hash: str = ""                   # sha256
    producer_agent: AgentName | None = None
    size_bytes: int = 0


TParams = TypeVar("TParams")
TResult = TypeVar("TResult")


class AgentTask(BaseModel, Generic[TParams]):
    """Supervisor → Agent task input."""
    task_id: str                             # globally unique
    trace_id: str                            # end-to-end trace identifier
    parent_run_id: str                       # root run ID from supervisor
    goal: str                                # natural language goal
    agent: AgentName                         # target agent
    params: TParams                          # agent-specific parameters (strongly typed)
    artifacts_in: list[Artifact] = Field(default_factory=list)
    schema_version: str = "v1"               # contract version for compatibility
    max_retries: int = 2
    deadline: datetime | None = None


class AgentResult(BaseModel, Generic[TResult]):
    """Agent → Supervisor task output."""
    task_id: str
    trace_id: str
    status: TaskStatus
    agent: AgentName
    artifacts_out: list[Artifact] = Field(default_factory=list)
    data: TResult | None = None              # agent-specific structured output
    errors: list[str] = Field(default_factory=list)
    next_action: str = ""                    # suggested correction or next step
    retries_used: int = 0
    wall_time_s: float = 0.0
```

### 2.2 Design Rules

- **Use `Field(default_factory=list)` / `Field(default_factory=dict)`** instead of mutable defaults (`[]`, `{}`). Pydantic requires this to prevent shared mutable state across instances.
- **`AgentTask` / `AgentResult` are communication protocols**, not LangGraph graph state. LangGraph state holds execution context; contracts represent sub-task I/O.
- **`agent: AgentName` uses `Literal`** for IDE autocomplete and routing safety, not raw `str`.
- **Add `trace_id`** on every message for monitoring, replay, and failure correlation.
- **Contracts are versioned** (`schema_version: "v1"`). Breaking changes go to `contracts/v2/`.

### 2.3 Domain Agent Contracts

#### Gaussian

```python
# contracts/gaussian_task.py
class GaussianParams(BaseModel):
    method: str = "B3LYP"
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    job_type: Literal["opt", "opt+freq", "sp", "irc"] = "opt"
    additional_keywords: dict[str, str] = Field(default_factory=dict)
    memory_mb: int = 4000
    nprocs: int = 8
    checkpoint_from: str | None = None

class GaussianResult(BaseModel):
    energy: float | None = None
    dipole: list[float] | None = None
    n_imag_freq: int = 0
    converged: bool = False
    checkpoint_path: str | None = None
    wall_time_s: float = 0.0
```

#### PySCF

```python
# contracts/pyscf_task.py
class PySCFParams(BaseModel):
    fchk_path: str
    n_active_electrons: int
    n_active_orbitals: int
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    cas_irrep: str | None = None
    max_memory_mb: int = 8000

class PySCFResult(BaseModel):
    active_orbitals: list[int] = Field(default_factory=list)
    casscf_energy: float | None = None
    natural_occupations: list[float] = Field(default_factory=list)
    orbital_info_path: str | None = None
```

#### Orca

```python
# contracts/orca_task.py
class OrcaParams(BaseModel):
    orbital_info_path: str
    method: Literal["NEVPT2", "DLPNO-CCSD(T)", "CCSD(T)"] = "NEVPT2"
    basis: str = "def2-TZVP"
    charge: int = 0
    spin: int = 1
    memory_mb: int = 16000
    nprocs: int = 16

class OrcaResult(BaseModel):
    energy: float | None = None
    energy_correction: float | None = None
    converged: bool = False
    log_path: str | None = None
    wall_time_s: float = 0.0
```

#### HPC (Infrastructure)

```python
# contracts/hpc_task.py
class JobRequest(BaseModel):
    agent: AgentName
    input_file: str
    rough_slurm: str
    mem_per_cpu_mb: int
    nprocs: int
    walltime_hours: int

class HPCParams(BaseModel):
    jobs: list[JobRequest] = Field(default_factory=list)
    partition: str = "compute"
    account: str = "default"
    email_on_fail: str | None = None

class HPCResult(BaseModel):
    job_ids: dict[str, str] = Field(default_factory=dict)
    submitted_at: datetime | None = None
    node_assignment: dict[str, str] = Field(default_factory=dict)
    final_slurm_paths: dict[str, str] = Field(default_factory=dict)
```

#### Monitor (Infrastructure)

```python
# contracts/monitor_task.py
class MonitorParams(BaseModel):
    watchlist: dict[str, str] = Field(default_factory=dict)
    # {"12345": "gaussian", "12346": "orca"}
    poll_interval_s: int = 60
    max_watch_hours: int = 48

class MonitorEvent(BaseModel):
    id: int = 0                             # monotonic, append-only
    job_id: str
    agent: AgentName
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["JOB_STARTED", "JOB_DONE", "JOB_FAILED",
                        "JOB_TIMEOUT", "JOB_OOM", "AGENT_HEARTBEAT_LOST"]
    log_snippet: str = ""
    error_category: Literal["scf_not_converged", "oom", "mpi_error",
                            "timeout", "unknown", ""] = ""
    suggestion: str = ""

class MonitorResult(BaseModel):
    events: list[MonitorEvent] = Field(default_factory=list)
    failed_jobs: list[str] = Field(default_factory=list)
    completed_jobs: list[str] = Field(default_factory=list)
    summary: str = ""
```

---

## 3. Supergraph — LangGraph Topology

### 3.1 Graph Structure

```
                         ┌────────────────┐
                         │   supervisor   │  create_supervisor()
                         │  (router LLM)  │  注册所有子 agent
                         └───┬───┬───┬───┘
                             │   │   │   handoff (tool call)
              ┌──────────────┼───┼───┼──────────────┐
              │              │   │   │              │
         ┌────▼───┐    ┌────▼──┐ ┌▼──────┐   ┌─────▼─────┐
         │gaussian│    │pyscf  │ │ orca  │   │   hpc     │
         │ agent  │    │agent  │ │ agent │   │  agent    │
         │subgraph│    │subgraph│ │subgraph│   │ subgraph │
         └────────┘    └───────┘ └───────┘   └───────────┘
              │              │       │              │
              └──────────────┼───────┼──────────────┘
                             │       │
                    ┌────────▼───────▼────────┐
                    │     monitor agent       │  ← event-driven,
                    │       subgraph          │    separate from main chain
                    └─────────────────────────┘
```

### 3.2 Terminology

- **Agent**: Logical role (GaussianAgent, MonitorAgent, ...)
- **Graph / Subgraph**: Implementation vehicle — compiled LangGraph `StateGraph`
- Each agent compiles to an independent subgraph; supervisor coordinates via handoff

### 3.3 Supervisor Definition

The supervisor uses `create_supervisor()` from LangGraph. It registers all domain and infrastructure agents as subgraphs.

```python
# agents/supervisor/supervisor.py

from langgraph_supervisor import create_supervisor
from agents.gaussian.agent import gaussian_graph
from agents.pyscf.agent import pyscf_graph
from agents.orca.agent import orca_graph
from agents.hpc.agent import hpc_graph

supervisor = create_supervisor(
    agents=[gaussian_graph, pyscf_graph, orca_graph, hpc_graph],
    model="claude-sonnet-4-6",
    prompt="""
    You are a quantum chemistry computational supervisor.

    ## Responsibilities
    1. Parse chemistry tasks → decompose into ordered sub-tasks
    2. Handoff to domain agents based on dependency order
    3. After each agent completes: decide next action (continue/retry/escalate/human)
    4. When all sub-tasks complete: synthesize final result

    ## Routing Rules
    - Wavefunction prep / HF optimization → gaussian agent
    - Orbital selection / CASSCF / post-HF → pyscf agent
    - NEVPT2 / coupled cluster / excited states → orca agent
    - Slurm generation / resource query / job submission → hpc agent
    - Queue monitoring / log analysis / error feedback → monitor agent

    ## Parallel Strategy
    - Chemical agents (gaussian → pyscf → orca) execute sequentially
    - HPC agent can launch in parallel once all agent compute needs are known
    - Monitor agent activates after any job submission, event-driven

    ## Failure Handling
    - Same agent: max 2 retries
    - Same error type repeating → escalate to NEEDS_HUMAN
    - Upstream agent failure → downstream dependents do not execute
    """,
)
```

### 3.4 Domain Agent Subgraph (Gaussian Example)

```python
# agents/gaussian/agent.py

from langgraph.graph import StateGraph, START, END
from tower_agent_kit.base import BaseAgentState

class GaussianState(BaseAgentState):
    """Gaussian agent internal state — separate from AgentTask protocol."""
    input_written: bool = False
    slurm_template_path: str = ""
    job_id: str = ""


def write_input(state: GaussianState) -> dict:
    """Generate Gaussian input file (.gjf) from params template."""
    params = state.task.params
    input_content = render_gaussian_input(params)
    return {"input_written": True}


def generate_slurm_template(state: GaussianState) -> dict:
    """Generate rough Slurm template (HPC agent refines later)."""
    script = render_slurm_template(state.task)
    return {"slurm_template_path": f"jobs/{state.task.task_id}.sh"}


def run_or_submit(state: GaussianState) -> dict:
    """Execute calculation or delegate to HPC agent for submission."""
    # Domain agent generates input + rough slurm; HPC handles actual submission
    return {}


gaussian_graph = (
    StateGraph(GaussianState)
    .add_node("write_input", write_input)
    .add_node("generate_slurm_template", generate_slurm_template)
    .add_node("run_or_submit", run_or_submit)
    .add_edge(START, "write_input")
    .add_edge("write_input", "generate_slurm_template")
    .add_edge("generate_slurm_template", "run_or_submit")
    .add_edge("run_or_submit", END)
    .compile()
)
```

### 3.5 Monitor Agent (Independent, Event-Driven)

Monitor does not participate in the main handoff chain. It operates independently:

```
                       ┌──────────────┐
                       │  RunStateStore│  ← system single source of truth
                       └───┬───────┬──┘
                           │       │
              ┌────────────▼─┐ ┌───▼────────────┐
              │ MonitorAgent │ │ SupervisorAgent │
              │              │ │                 │
              │ 1.poll jobs  │ │ 3.read events[] │
              │ 2.write      │ │ 4.decide:       │
              │   MonitorEvt │ │   retry/skip/   │
              │   append-only│ │   needs_human   │
              └──────────────┘ └─────────────────┘
```

### 3.6 Complete Orchestration Flow

```
User: "Run NEVPT2 on N2"
  │
  ▼
Supervisor.invoke()
  │  LLM analyzes: gaussian → pyscf → orca, hpc parallel, monitor watch
  │
  ├─handoff→ gaussian_graph.invoke(AgentTask[GaussianParams])
  │            Returns: AgentResult(status=DONE, artifacts=[fchk])
  │
  ├─handoff→ pyscf_graph.invoke(AgentTask[PySCFParams])
  │            Returns: AgentResult(status=DONE, artifacts=[orbitals.json])
  │
  ├─handoff→ orca_graph.invoke(AgentTask[OrcaParams])
  │         └─parallel→ hpc_graph.invoke(AgentTask[HPCParams])
  │                      Returns: AgentResult(status=DONE, job_ids={...})
  │
  │  monitor_graph runs independently, event-driven
  │     ├─ "gaussian job 12345 COMPLETED" → ok
  │     ├─ "orca job 12347 FAILED: SCF not converged"
  │     └─ → supervisor reads event → handoff orca agent with correction
  │
  └─ Synthesize result → return to user
```

---

## 4. Agent Responsibility Boundaries & I/O Contracts

### 4.1 GaussianAgent

| Dimension | Detail |
|-----------|--------|
| **Responsibility** | Generate Gaussian input files; run HF/DFT optimization, frequency, single-point; output converged wavefunction checkpoint |
| **Does NOT do** | Slurm script refinement, cluster submission (→ HPC), log monitoring (→ Monitor) |
| **MCP Tools** | `infra-mcp` (filesystem, template) |
| **Produces** | `.fchk` / `.log` / `.gjf` + rough Slurm template |
| **Failure modes** | Geometry optimization non-convergence, SCF oscillation, symmetry breaking, input parameter conflicts |

**Input**: `AgentTask[GaussianParams]` | **Output**: `AgentResult[GaussianResult]`

### 4.2 PySCFAgent

| Dimension | Detail |
|-----------|--------|
| **Responsibility** | Read upstream wavefunction; select active orbitals (e.g., Pi orbitals); run CASSCF/CASPT2 (PySCF side); output orbital info and reference energy |
| **Does NOT do** | Input generation (Gaussian did it), NEVPT2 (→ Orca), cluster management |
| **MCP Tools** | `infra-mcp` (filesystem: read .fchk, write .json; template: PySCF script) |
| **Produces** | `active_orbitals.json` / `casscf_energy.json` |
| **Failure modes** | Unreasonable orbital selection, CAS space too large OOM, SCF non-convergence |

**Input**: `AgentTask[PySCFParams]` | **Output**: `AgentResult[PySCFResult]`

### 4.3 OrcaAgent

| Dimension | Detail |
|-----------|--------|
| **Responsibility** | Read CASSCF orbital info; generate Orca NEVPT2/coupled-cluster input; initiate calculation |
| **Does NOT do** | HF optimization, orbital selection, cluster management |
| **MCP Tools** | `infra-mcp` (filesystem: write .inp; template: Orca input) |
| **Produces** | `.inp` / `.log` / `.hess` |
| **Failure modes** | NEVPT2 non-convergence, insufficient memory, MPI configuration errors |

**Input**: `AgentTask[OrcaParams]` | **Output**: `AgentResult[OrcaResult]`

### 4.4 HPCAgent (Infrastructure)

| Dimension | Detail |
|-----------|--------|
| **Responsibility** | Collect compute requirements from all agents; query cluster resource status; generate refined Slurm scripts; submit jobs |
| **Does NOT do** | Input generation, chemical computation, error attribution |
| **MCP Tools** | `hpc-mcp` (queue-status), `infra-mcp` (slurm-gen) |
| **Produces** | `*.sh` (final Slurm scripts), `job_ids.json` |
| **Runs** | Parallel with chemical agents: collect all rough Slurm templates, refine, submit |

**Input**: `AgentTask[HPCParams]` | **Output**: `AgentResult[HPCResult]`

### 4.5 MonitorAgent (Infrastructure, Event-Driven)

| Dimension | Detail |
|-----------|--------|
| **Responsibility** | Subscribe to submitted job IDs; poll queue status; parse failure logs; classify and attribute errors; feedback to responsible agent via event store |
| **Does NOT do** | Fix computations (→ agent), submit jobs (→ HPC), parameter validation |
| **MCP Tools** | `hpc-mcp` (queue-status, log-parser) |
| **Produces** | `MonitorEvent` records in `RunStateStore.event_log` |
| **Runs** | Activates after any job submission; event-driven; independent of main handoff chain |

**Input**: `AgentTask[MonitorParams]` | **Output**: `AgentResult[MonitorResult]`

### 4.6 Dependency Graph

```
gaussian ──→ pyscf ──→ orca
   │           │         │
   └───────────┼─────────┘
               │  (chemical agents produce artifacts consumed by downstream)
               
hpc:     depends on all agents' rough slurm templates
monitor: depends on hpc job_ids; writes events for all agents
```

---

## 5. File Structure — Independent Package Layout

### 5.1 Repository Map

```
tower/
├── pyproject.toml                   # tower-core base package
├── .gitignore                       # includes .old_tower/
├── .old_tower/                      # old single-agent code (git ignored)
│
├── src/tower/                       # [RETAIN] Core infrastructure
│   ├── memory/                      #   AsyncPostgresSaver/Store + pool
│   ├── tools/                       #   Built-in tools (safety, registry)
│   ├── state/                       #   [NEW] RunStateStore, ArtifactRegistry,
│   │   ├── run_store.py             #         JobRegistry
│   │   ├── artifact_registry.py
│   │   └── job_registry.py
│   ├── cli/                         #   [NEW] CLI entry (graph-agnostic)
│   └── mcp/                         #   [NEW] MCP client manager
│
├── contracts/                       # [NEW] Shared Pydantic protocols
│   ├── pyproject.toml               #   pip install -e contracts/
│   ├── agent_task.py                #   AgentTask[T] / AgentResult[T] / Artifact
│   ├── gaussian_task.py             #   GaussianParams / GaussianResult
│   ├── pyscf_task.py                #   PySCFParams / PySCFResult
│   ├── orca_task.py                 #   OrcaParams / OrcaResult
│   ├── hpc_task.py                  #   HPCParams / HPCResult
│   └── monitor_task.py              #   MonitorParams / MonitorResult
│
├── tower_agent_kit/                 # [NEW] Lightweight agent scaffold
│   ├── pyproject.toml
│   └── base.py                      #   BaseAgentState, BaseAgentGraph
│
├── agents/                          # [NEW] Each agent is an independent package
│   ├── supervisor/
│   │   ├── pyproject.toml
│   │   ├── supervisor.py            #   create_supervisor() + agent registration
│   │   └── prompts.py               #   Routing rules prompt
│   │
│   ├── gaussian/
│   │   ├── pyproject.toml
│   │   ├── agent.py                 #   GaussianState + subgraph compilation
│   │   ├── nodes.py                 #   write_input, generate_slurm_template, run
│   │   ├── tools.py                 #   Gaussian-specific MCP tool registration
│   │   └── prompts.py               #   Gaussian input template rendering
│   │
│   ├── pyscf/
│   │   ├── pyproject.toml
│   │   ├── agent.py
│   │   ├── nodes.py
│   │   ├── tools.py
│   │   └── prompts.py
│   │
│   ├── orca/
│   │   ├── pyproject.toml
│   │   ├── agent.py
│   │   ├── nodes.py
│   │   ├── tools.py
│   │   └── prompts.py
│   │
│   ├── hpc/
│   │   ├── pyproject.toml
│   │   ├── agent.py
│   │   ├── nodes.py                 #   collect_needs, refine_slurm, submit
│   │   └── tools.py
│   │
│   └── monitor/
│       ├── pyproject.toml
│       ├── agent.py
│       ├── nodes.py                 #   poll_jobs, parse_logs, classify_error
│       └── tools.py
│
├── mcp_servers/                     # [NEW] MCP tool servers
│   ├── infra-mcp/
│   │   └── server.py                #   filesystem + template + slurm-gen
│   └── hpc-mcp/
│       └── server.py                #   queue-status + log-parser
│
├── skills/                          # [RETAIN] Existing dmrg skill pack
│   └── dmrg/                        #   Can later be split into agents/
│
└── tests/
    ├── test_contracts/
    ├── test_agents/
    └── test_mcp_servers/
```

### 5.2 Agent Package Template

```toml
# agents/gaussian/pyproject.toml
[project]
name = "tower-agent-gaussian"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "tower-core>=0.2.0",
    "tower-contracts>=0.1.0",
    "tower-agent-kit>=0.1.0",
    "langgraph>=1.0.0",
    "langchain-core>=0.3.0",
    "pydantic>=2.0",
]
```

### 5.3 Dependency Graph

```
                        tower-core (memory, state store, CLI, MCP client)
                              │
                         contracts (Pydantic schemas)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │ supervisor│    │ domain agents│    │  infra agents│
    │           │    │ gaussian     │    │  hpc         │
    │           │    │ pyscf        │    │  monitor     │
    │           │    │ orca         │    │              │
    └───────────┘    └───────┬──────┘    └───────┬──────┘
                             │                   │
                             └─────────┬─────────┘
                                       │
                              mcp_servers (infra-mcp, hpc-mcp)
```

### 5.4 Team Collaboration Model

```
Developer A              Developer B              Developer C
gaussian agent           pyscf agent              orca agent
agents/gaussian/*        agents/pyscf/*           agents/orca/*
    │                         │                        │
    └─────────────────────────┼────────────────────────┘
                              │ Shared contracts/ + tower-core
                              │
                    Developer D              Developer E
                    hpc agent               monitor agent
                    agents/hpc/*            agents/monitor/*
                         │                        │
                         └───────────┬────────────┘
                                     │ MCP servers
```

Each developer:
1. Works only in their agent directory
2. Depends on `contracts/` for type contracts
3. Accesses shared tools through `tower-core` MCP client
4. Does not need to understand other agents' internals
5. Each agent can be `pip install -e .` independently for testing

---

## 6. Execution Model & State Consistency

### 6.1 Global Run State (Single Source of Truth)

```python
# src/tower/state/run_store.py

class RunState(BaseModel):
    """Global state for one user request. All agents read/write through this."""
    run_id: str
    trace_id: str
    task: str                            # original user task
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)

    # Sub-task status map
    agent_tasks: dict[str, TaskStatus] = Field(default_factory=dict)
    # {"gaussian-001": DONE, "pyscf-001": RUNNING}

    # Global artifact index
    artifacts: dict[str, "ArtifactRecord"] = Field(default_factory=dict)

    # HPC job status
    jobs: dict[str, "JobRecord"] = Field(default_factory=dict)

    # Supervisor decision log
    decisions: list["SupervisorDecision"] = Field(default_factory=list)

    # Event log (append-only) for monitor events
    event_log: list["MonitorEvent"] = Field(default_factory=list)
    last_processed_event_id: int = 0     # event watermark

    # Agent heartbeat tracking
    agent_heartbeats: dict[str, datetime] = Field(default_factory=dict)

    # Execution mode
    mode: Literal["live", "replay", "dry_run"] = "live"
```

### 6.2 Artifact Lifecycle

```
[Upstream agent produces]
       │
       ▼
┌─────────────────┐
│  ArtifactRegistry│  Register artifact
│  .register(      │  - Generate content_hash
│    path, type,   │  - Record producer_agent
│    producer)     │  - Status = READY
└────────┬─────────┘
         │
    ┌────▼────┐
    │ READY   │  Available for downstream consumption
    └────┬────┘
         │  Downstream agent reads it
    ┌────▼────┐
    │CONSUMED │  Read by downstream (for GC decisions)
    └────┬────┘
         │  Upstream agent retries
    ┌────▼────┐
    │ STALE   │  Original producer produced a new version
    └─────────┘
```

```python
class ArtifactStatus(str, Enum):
    PRODUCING = "producing"
    READY = "ready"
    CONSUMED = "consumed"
    STALE = "stale"


class ArtifactRecord(BaseModel):
    artifact_id: str
    path: str
    type: str
    content_hash: str                       # sha256 for version tracking
    producer_agent: AgentName
    producer_task_id: str
    status: ArtifactStatus = ArtifactStatus.PRODUCING
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
```

**Key Constraints:**
- Artifacts are **immutable**. Retry = new artifact_id, old marked STALE.
- Downstream references by `artifact_id`, never by raw path.
- `content_hash` enables Monitor to detect log updates.

### 6.3 Artifact Resolver Layer

```python
class ArtifactResolver:
    """Agents ALWAYS resolve artifacts through this, never by raw path."""

    def resolve(self, artifact_id: str,
                expected_status: ArtifactStatus = ArtifactStatus.READY) -> ArtifactRecord: ...

    def get_path(self, artifact_id: str) -> str: ...
    def get_hash(self, artifact_id: str) -> str: ...
```

**Hard rule: Agent code MUST NOT concatenate file paths directly. All file references go through ArtifactResolver.**

### 6.4 Job State Machine

```
              HPC submit
                 │
            ┌────▼────┐
            │QUEUED   │  Waiting for scheduling
            └────┬────┘
                 │
            ┌────▼────┐
            │RUNNING  │  Executing
            └────┬────┘
                 │
        ┌────────┼────────┐
        │        │        │
   ┌────▼──┐ ┌──▼───┐ ┌──▼──────┐
   │DONE   │ │FAILED│ │TIMEOUT  │
   └───┬───┘ └──┬───┘ └────┬────┘
       │        │           │
       │   ┌────▼────┐      │
       │   │RETRYING │◄─────┘   Monitor feedback → supervisor decision
       │   └────┬────┘
       │        │
       └────────┼──────────────→ Final completion
                │
           ┌────▼────┐
           │ABANDONED│  Max retries exceeded
           └─────────┘
```

### 6.5 State Transition Rules

All state writes must pass transition validation. These are the only legal transitions:

```python
AGENT_TASK_TRANSITIONS = {
    PENDING:   {RUNNING},
    RUNNING:   {DONE, FAILED, NEEDS_HUMAN},
    FAILED:    {RETRYING, ABANDONED},
    RETRYING:  {RUNNING},
    ABANDONED: set(),       # terminal
    DONE:      set(),       # terminal
}

ARTIFACT_TRANSITIONS = {
    PRODUCING: {READY, STALE},
    READY:     {CONSUMED, STALE},
    CONSUMED:  {STALE},
    STALE:     set(),       # terminal
}

JOB_TRANSITIONS = {
    QUEUED:    {RUNNING},
    RUNNING:   {DONE, FAILED, TIMEOUT},
    FAILED:    {RETRYING, ABANDONED},
    TIMEOUT:   {RETRYING, ABANDONED},
    RETRYING:  {QUEUED},
    ABANDONED: set(),       # terminal
    DONE:      set(),       # terminal
}
```

### 6.6 Write Ownership (Concurrency Model)

```
┌──────────────────┬───────────────────────────────────────┐
│ Writer           │ Fields owned                          │
├──────────────────┼───────────────────────────────────────┤
│ Supervisor       │ RunState.status, agent_tasks dispatch,│
│                  │ SupervisorDecision, retry initiation  │
├──────────────────┼───────────────────────────────────────┤
│ Domain Agent     │ Own AgentResult, own Artifact         │
│                  │ (PRODUCING→READY only)               │
├──────────────────┼───────────────────────────────────────┤
│ HPC Agent        │ JobRecord (QUEUED→RUNNING),           │
│                  │ slurm_script_path                     │
├──────────────────┼───────────────────────────────────────┤
│ Monitor Agent    │ MonitorEvent (append-only),           │
│                  │ JobRecord.status (→DONE/FAILED),      │
│                  │ JobRecord.error_category              │
└──────────────────┴───────────────────────────────────────┘

Rules:
- ONE field = ONE writer
- ANY agent can READ any field
- validate_transition() called before every write
```

### 6.7 Monitor → Supervisor Event Protocol

```
                       ┌──────────────┐
                       │  RunStateStore│  ← system single source of truth
                       └───┬───────┬──┘
                           │       │
              ┌────────────▼─┐ ┌───▼────────────┐
              │ MonitorAgent │ │ SupervisorAgent │
              │              │ │                 │
              │ 1.poll jobs  │ │ 3.read events[] │
              │ 2.write      │ │ 4.decide:       │
              │   MonitorEvt │ │   retry/skip/   │
              │   append-only│ │   needs_human   │
              └──────────────┘ └─────────────────┘
```

Event watermark ensures idempotent consumption:

```python
# Supervisor reads only new events
async def read_new_events(run_state: RunState) -> list[MonitorEvent]:
    new = [e for e in run_state.event_log
           if e.id > run_state.last_processed_event_id]
    return sorted(new, key=lambda e: e.id)
```

### 6.8 Anti-Drift Hard Constraints

| Rule | Implementation |
|------|---------------|
| **Single writer per field** | Write ownership table enforced by `validate_transition()` |
| **Immutable artifacts** | Created once, never modified; retry → new artifact_id |
| **Artifact resolution** | All file access through `ArtifactResolver`; raw path concatenation forbidden |
| **Event append-only** | Monitor events never deleted or modified |
| **Supervisor is sole decision-maker** | Only supervisor modifies `RunState.status` and initiates retries |
| **Read-write separation** | Agents read via `RunStateStore.get()`, write via `RunStateStore.update()` |
| **Isolated state per agent** | Each agent's internal state is private; communication only via AgentTask/AgentResult |

---

## 7. Execution Semantics & Fault Tolerance

### 7.1 Retry Semantics

| Operation type | Idempotent? | Retry strategy | Example |
|---------------|------------|----------------|---------|
| **Pure function** | ✅ Yes | Direct retry | Input file rendering, parameter validation |
| **Reversible side-effect** | ⚠️ Conditional | Cleanup → retry | Slurm `scancel` then `sbatch` |
| **Irreversible side-effect** | ❌ No | Block auto-retry, needs human | Overwriting existing `.fchk`, deleting data |

```python
class RetryPolicy(BaseModel):
    max_retries: int = 2
    is_idempotent: bool = True
    requires_cleanup_before_retry: bool = False
    cleanup_steps: list[str] = Field(default_factory=list)
    backoff_s: int = 30
    escalate_after_max: bool = True


AGENT_RETRY_POLICIES: dict[AgentName, RetryPolicy] = {
    "gaussian": RetryPolicy(is_idempotent=True),
    "pyscf":    RetryPolicy(is_idempotent=True),
    "orca":     RetryPolicy(is_idempotent=True),
    "hpc":      RetryPolicy(
        is_idempotent=False,
        requires_cleanup_before_retry=True,
        cleanup_steps=["scancel old job", "remove stale slurm script"],
    ),
    "monitor":  RetryPolicy(is_idempotent=True),
}
```

### 7.2 Partial Failure Recovery

```python
# Dependency graph: {"downstream": {"upstream1", "upstream2"}}
DEP_GRAPH: dict[str, set[str]] = {
    "pyscf": {"gaussian"},
    "orca": {"pyscf"},
    "hpc": set(),          # no chemical dependencies
    "monitor": set(),      # no dependencies
}


def compute_invalidation_scope(
    failed_agent: str,
    dep_graph: dict[str, set[str]],
) -> set[str]:
    """BFS: find all agents that must be invalidated when one fails."""
    affected = {failed_agent}
    queue = [failed_agent]
    while queue:
        current = queue.pop(0)
        for downstream, upstreams in dep_graph.items():
            if current in upstreams and downstream not in affected:
                affected.add(downstream)
                queue.append(downstream)
    return affected

# pyscf fails → invalidate {pyscf, orca}, keep gaussian
# hpc → cancel orca job, re-submit after orca retries
```

### 7.3 Failure Recovery Decision Tree

```
                     ┌──────────────┐
                     │ Agent returns│
                     │ FAILED       │
                     └──────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Auto-fixable? │
                    └───┬───────┬───┘
                        │YES    │NO
                ┌───────▼──┐ ┌──▼──────────┐
                │ retries  │ │ NEEDS_HUMAN │
                │ < 2?     │ │ Stop chain  │
                └──┬───┬───┘ └─────────────┘
                   │YES│NO
          ┌────────▼─┐ ┌▼──────────┐
          │ Idemp    │ │ ABANDONED │
          │ otent?   │ │ Compute    │
          └──┬───┬───┘ │ invalidation│
             │YES│NO   │ scope,     │
     ┌───────▼┐ ┌▼─────┤ report to  │
     │ Direct │ │Cleanup│ user       │
     │ retry  │ │→retry │            │
     └────────┘ └───────┴────────────┘
```

### 7.4 LangGraph Checkpoint Strategy

```python
# Domain agent subgraph — no human interrupt needed internally
gaussian_graph = gaussian_graph.compile(
    checkpointer=async_postgres_saver,
    interrupt_before=[],
)

# Supervisor graph — checkpoint before every handoff
supervisor_graph = create_supervisor(
    agents=[...],
    checkpointer=async_postgres_saver,
    interrupt_before=["handoff"],
)
# Benefit: if supervisor crashes at any handoff point,
# resume from last checkpoint without re-executing completed agents
```

### 7.5 Timeout & Heartbeat

```python
class TimeoutPolicy(BaseModel):
    agent_timeout_s: int = 3600
    job_queue_timeout_s: int = 86400
    heartbeat_interval_s: int = 300       # agent writes heartbeat every 5 min
    heartbeat_timeout_s: int = 900        # 15 min no heartbeat → agent presumed dead


TIMEOUT_POLICY = TimeoutPolicy()
```

MonitorAgent checks heartbeats: if an agent in RUNNING status has no heartbeat for 15 minutes, Monitor writes `AGENT_HEARTBEAT_LOST` event. Supervisor reads it and decides retry or escalate.

### 7.6 Run Replay Mode

```python
class RunMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"        # replay from event_log, no real computation
    DRY_RUN = "dry_run"      # run logic but don't write artifacts


# In REPLAY mode, supervisor replays recorded AgentResults from event_log
# to verify the decision chain is consistent across runs.
```

### 7.7 Consistency Guarantees Summary

| Guarantee | Mechanism |
|-----------|-----------|
| Idempotent retry | `RetryPolicy.is_idempotent` + `cleanup_steps` |
| No duplicate event processing | Event watermark (`last_processed_event_id`) |
| No missed events | `event_log` append-only + supervisor tick consumption |
| Partial failure isolation | `compute_invalidation_scope()` + dependency graph |
| State validity | `validate_transition()` on every write path |
| Reference consistency | `ArtifactResolver`; agents forbidden from raw path construction |
| Crash recovery | LangGraph checkpoint before every handoff |
| Agent death detection | Heartbeat + MonitorAgent fallback |
| Full replayability | `event_log` append-only + `RunMode.REPLAY` |

---

## Appendix A: Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| State drift across agents | Single-writer ownership table; validate_transition() guard |
| Artifact path inconsistency | ArtifactResolver layer; ban raw path usage |
| MCP deployment complexity | Start with 2 MCP servers (infra-mcp + hpc-mcp); split later |
| Agent glue code duplication | `tower-agent-kit` base classes |
| Contract version incompatibility | `schema_version` field; `contracts/v1/` → `v2/` migration path |
| Event loss / duplication | Event watermark + append-only log |
| Agent silent death | Heartbeat + MonitorAgent timeout detection |

## Appendix B: Design References

- **LangGraph supervisor pattern**: Central supervisor coordinates specialized worker subagents via tool-based handoff. ([LangChain docs](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant))
- **LangGraph graph API**: Supports multi-level hierarchies, subgraphs, and sending state to specific nodes. ([LangChain docs](https://docs.langchain.com/oss/python/langgraph/graph-api))
- **MCP architecture**: Independent servers exposing standardized tools/resources/prompts. ([Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture))
- **Anthropic multi-agent research**: Multiple specialized agents divide work, synthesize results. ([Anthropic blog](https://www.anthropic.com/engineering/multi-agent-research-system))
- **Pydantic fields**: `Field(default_factory=...)` for mutable defaults. ([Pydantic docs](https://docs.pydantic.dev/latest/concepts/fields/))
