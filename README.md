# Tower

**Supervisor-led modular multi-agent system for computational chemistry.**

Built on LangGraph + MCP. A supervisor agent decomposes tasks, dispatches to specialized domain agents, collects results, and synthesizes output. All agents are independent — the supervisor alone decides execution order.

```
User: "N2 NEVPT2"
       ↓
Supervisor: plan → gaussian → pyscf → orca
       ↓            ↓         ↓        ↓
    dispatch    .gjf+slurm  orbitals  NEVPT2 energy
       ↓            ↓         ↓        ↓
    collect    ✓ done     ✓ done    ✓ done
       ↓
Synthesize: E(NEVPT2) = -109.52 Ha
```

## Quick Start

```bash
uv sync
uv run tower agents          # list all registered agents
uv run tower run "N2 NEVPT2" # execute a task
```

## Architecture

Three-layer design:

```
┌─────────────────────────────────────────┐
│           SUPERVISOR AGENT              │
│  task → plan → dispatch → synthesize    │
└────┬──────────┬──────────┬─────────────┘
     │          │          │
┌────▼──┐ ┌────▼──┐ ┌────▼──┐  ┌────▼──────────┐
│gaussian│ │pyscf  │ │ orca │  │  hpc + monitor │
│ agent  │ │agent  │ │agent │  │  (infra)       │
└───┬────┘ └───┬───┘ └───┬──┘  └──────┬────────┘
    │          │         │              │
┌───▼──────────▼─────────▼──────────────▼──────┐
│           SHARED MCP TOOL LAYER              │
│  infra-mcp (filesystem, template, slurm)     │
│  hpc-mcp  (queue-status, log-parser)         │
└──────────────────────────────────────────────┘
```

### Agents (6 registered)

| Agent | Type | Role |
|-------|------|------|
| `supervisor` | orchestrator | Task decomposition, dispatch, result synthesis |
| `gaussian` | domain | HF/DFT input generation + output parsing |
| `pyscf` | domain | Orbital selection + CASSCF |
| `orca` | domain | NEVPT2 / coupled-cluster |
| `hpc` | infrastructure | Slurm refinement, resource query, job submission |
| `monitor` | infrastructure | Queue polling, log parsing, error feedback |

All agents are independent — no hard dependencies. The supervisor decides execution order and passes artifacts between agents.

### Agent Lifecycle (Pre/Post Computation)

Each domain agent is called twice per run:

1. **Pre-computation** — query DB → generate input files → rough slurm template → return artifacts
2. HPC agent submits the job, Monitor watches it
3. **Post-computation** — read output → parse results → register artifacts for downstream agents

## Project Structure

```
tower/
├── contracts/                   # frozen Pydantic schemas (pip package)
│   └── src/contracts/           # AgentTask[T], AgentResult[T], domain Params/Result
├── tower_agent_kit/             # agent scaffold (pip package)
│   └── src/tower_agent_kit/     # BaseAgentState, AgentRegistration
├── src/tower/                   # core infrastructure
│   ├── state/                   #   RunStateStore, ArtifactRegistry, JobRegistry
│   ├── memory/                  #   AsyncPostgresSaver/Store + Memory OS
│   ├── mcp/                     #   MCP client (unified tool access)
│   ├── tools/                   #   built-in tools (filesystem, bash, web)
│   └── cli.py                   #   CLI entry point
├── agents/                      # each agent is an independent module
│   ├── supervisor/              #   plan → dispatch → synthesize
│   ├── gaussian/                #   query DB → input → slurm | parse
│   ├── pyscf/                   #   read fchk → orbitals → slurm | parse
│   ├── orca/                    #   read orbitals → inp → slurm | parse
│   ├── hpc/                     #   squeue → refine slurm → sbatch
│   └── monitor/                 #   poll sacct → parse log → classify
├── mcp_servers/                 # MCP tool servers
│   ├── infra-mcp/               #   filesystem, template, slurm-gen
│   └── hpc-mcp/                 #   queue-status, log-parser
├── docs/superpowers/
│   ├── specs/                   #   architecture design spec (8 sections)
│   ├── contracts/               #   frozen workflow/agent/tool contracts
│   └── archive/                 #   old single-agent docs
└── tests/
```

## How to Add a New Agent

1. Create `agents/<name>/` with `__init__.py` and `agent.py`
2. Inherit from `BaseAgentState[YourParams, YourResult]`
3. Implement nodes as pure functions `(state) → dict`
4. Build a `StateGraph` with conditional routing
5. Expose `register() → AgentRegistration`
6. Add your `Params`/`Result` models to `contracts/src/contracts/`

```python
# agents/myagent/agent.py
from tower_agent_kit.base import BaseAgentState, AgentRegistration

class MyState(BaseAgentState[MyParams, MyResult]):
    pass

def my_node(state: MyState) -> dict:
    return {"node_history": state.node_history + ["my_node"]}

def _finalize(state: MyState) -> dict:
    return {"agent_result": state.to_agent_result("myagent")}

my_graph = (
    StateGraph(MyState)
    .add_node("my_node", my_node)
    .add_node("finalize", _finalize)
    .add_edge(START, "my_node")
    .add_edge("my_node", "finalize")
    .add_edge("finalize", END)
    .compile()
)

def register() -> AgentRegistration:
    return AgentRegistration(name="myagent", subgraph=my_graph, ...)
```

## Design Documents

| Document | Description |
|----------|-------------|
| [Architecture Spec](docs/superpowers/specs/2026-06-17-tower-multi-agent-design.md) | 8-section full architecture (layers, contracts, supergraph, agent boundaries, state consistency, fault tolerance, Memory OS) |
| [Workflow Contract](docs/superpowers/contracts/workflow_contract.md) | Run lifecycle, handoff semantics, artifact lifecycle, event flow, invariants |
| [Agent Contract](docs/superpowers/contracts/agent_contract.md) | Per-agent frozen interfaces (input/output, nodes, tools, retry policy) |
| [Tool Contract](docs/superpowers/contracts/tool_contract.md) | MCP tool schemas (9 tools, 2 servers, idempotency, error codes) |

## Key Design Principles

- **Contracts are frozen**. All agent communication uses `AgentTask[T]` → `AgentResult[T]`. No side channels.
- **Artifacts are immutable**. Referenced by `artifact_id`, never by raw path. Retry → new artifact_id.
- **State transitions are validated**. `validate_transition()` guards every write. Illegal transitions raise errors.
- **Single writer per field**. Only the designated writer may modify each RunState field.
- **Agents are independent**. No agent knows about other agents. The supervisor owns all orchestration.
- **Memory is event-derived**. The Memory Compiler reads execution traces → produces knowledge. Agents never write memory directly.

## Development

```bash
uv sync
uv run pytest tests/ -v
```

Python 3.12+ required. 57 tests passing.

## License

MIT
