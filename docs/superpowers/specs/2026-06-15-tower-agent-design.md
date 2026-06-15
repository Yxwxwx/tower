# Tower Agent Framework — Design Spec

**Date:** 2026-06-15
**Status:** Approved
**Author:** Sun Xinyu

## 1. Overview

Tower is a general-purpose agent framework (CLI-first, Python) built on LangGraph and MCP, with domain specialization via installable **Skill Packs**. The first Skill Pack targets **quantum chemistry / DMRG / NQS** computational research.

### 1.1 Goals

- Build a **generic agent runtime** that is not coupled to any domain
- Inject domain expertise through **pluggable Skill Packs** (system prompt, tools, workflows, knowledge, evals)
- Support **CLI-first** interaction: `tower run "..."` for autonomous task execution, `tower chat` for interactive sessions
- Embed **evaluation from Day 1**: every skill pack ships with eval sets; every change can be regression-tested
- Produce structured, audit-able traces for every agent run
- Serve as a learning vehicle for agent architecture concepts and a portfolio/resume project

### 1.2 Non-Goals (v0)

- Web UI — CLI only
- Multi-tenant SaaS deployment
- Production-grade horizontal scaling
- Chat-only interaction without tool execution

## 2. Core Architecture

### 2.1 Three Fundamental Abstractions

The framework introduces exactly three first-class concepts:

#### Skill

A Python package that bundles everything an agent needs for a specific domain:

```
skill.yaml          → metadata, system prompt, rules
mcp_servers/        → domain-specific MCP server implementations
workflows/          → reusable LangGraph subgraphs for common task patterns
knowledge/          → seed data for vector/persistent knowledge
eval/               → task sets for regression testing
```

#### Agent Runtime

Domain-agnostic engine. It does four things:

1. **Load Skill** — parse `skill.yaml`, assemble system prompt, initialize MCP clients, connect knowledge bases
2. **Run State Machine** — execute the plan→act→observe→refine→respond cycle via LangGraph
3. **Call Tools** — discover and invoke tools exclusively through MCP protocol
4. **Record Everything** — structured event log for every node execution, tool call, and state transition

#### Memory

Three tiers, differentiated by lifecycle:

| Tier | Lifecycle | Content | Storage |
|------|-----------|---------|---------|
| Working | Single task | Current state, intermediate results, tool outputs | LangGraph state (in-process) |
| Session | Single CLI session | Conversation summary, project context | SQLite |
| Persistent | Cross-session | User preferences, verified knowledge, tool patterns | Postgres + pgvector |

### 2.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Agent orchestration | LangGraph (`StateGraph`) | Explicit state machine, durable execution, streaming, human-in-the-loop |
| Tool protocol | MCP (Model Context Protocol) | Standardized tool discovery and invocation; model-agnostic |
| LLM provider | Anthropic (primary), OpenAI (optional) | Claude API via `anthropic` SDK |
| Model config | LiteLLM (optional) | Multi-provider routing if needed |
| Vector search | pgvector | Embedding-based retrieval for knowledge bases |
| Tracing | Custom structured logger → SQLite | Every node, tool call, state transition recorded |
| CLI | `click` or `typer` | Standard Python CLI |
| Package management | `uv` / `pip` | Skill Packs as standard Python packages |

### 2.3 Project Directory Structure

```text
# Skills/ can live in the monorepo during initial development.
# Long-term, each skill is a separate pip-installable package (e.g. tower-skill-dmrg).
tower/
├── pyproject.toml
├── README.md
├── src/
│   └── tower/
│       ├── __init__.py
│       ├── main.py                  # CLI entry point
│       ├── cli/
│       │   ├── chat.py              # tower chat
│       │   ├── run.py               # tower run
│       │   └── eval.py              # tower eval
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── orchestrator.py      # LangGraph StateGraph construction
│       │   ├── skill_loader.py      # Load & validate skill.yaml
│       │   ├── mcp_client.py        # MCP stdio/sse client manager
│       │   └── state.py             # AgentState TypedDict schema
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── nodes.py             # plan, act, observe, refine, respond
│       │   └── edges.py             # conditional routing logic
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── working.py           # LangGraph state access
│       │   ├── session.py           # SQLite session store
│       │   └── persistent.py        # Postgres + pgvector
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── mcp_adapter.py       # MCP tool → LangChain tool adapter
│       │   └── builtin/             # Always-available tools
│       │       ├── filesystem.py
│       │       └── python_runner.py
│       ├── tracing/
│       │   ├── __init__.py
│       │   ├── logger.py            # Structured event logger
│       │   └── playback.py          # Trace replay
│       └── eval/
│           ├── __init__.py
│           ├── runner.py            # Eval harness
│           └── metrics.py           # Success rate, cost, latency
├── skills/
│   └── dmrg/                        # dmrg-skill-pack (separate package)
│       ├── pyproject.toml
│       └── src/dmrg_skill/
│           ├── skill.yaml
│           ├── mcp_servers/
│           │   ├── runner.py
│           │   ├── knowledge.py
│           │   └── analyzer.py
│           ├── workflows/
│           │   ├── convergence.py
│           │   └── paper_review.py
│           ├── knowledge/
│           │   └── seed_papers.json
│           └── eval/
│               ├── code_tasks.yaml
│               └── compute_tasks.yaml
└── tests/
    ├── test_runtime/
    ├── test_memory/
    ├── test_graph/
    └── test_eval/
```

## 3. Agent State Schema

```python
from typing import TypedDict, Annotated, Sequence, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolInvocation(TypedDict, total=False):
    tool: str
    intent: str
    expected_output: str
    observed_output: Any
    was_useful: bool


class AgentState(TypedDict, total=False):
    # ── Messages ──
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Task ──
    task: str
    plan: list[str]
    current_step_index: int

    # ── Tool execution ──
    tool_calls_pending: list[dict]
    tool_results: dict[str, Any]
    tool_history: list[ToolInvocation]

    # ── Observation & Refinement ──
    observation: str
    refinement_needed: bool
    refinement_action: str
    retry_count: int
    max_retries: int

    # ── Memory ──
    retrieved_context: list[str]
    working_memory: dict[str, Any]

    # ── Approval ──
    pending_approval: bool
    approval_action: str
    approval_rationale: str
    approved_actions: list[str]

    # ── Tracing ──
    trace_id: str
    node_history: list[str]

    # ── Final ──
    final_response: str
    task_complete: bool
```

Key design rule: **State is for current workflow, not for persistence.** Long-lived data lives in the Memory tiers.

## 4. StateGraph Workflow

### 4.1 Graph Topology

```text
                    ┌──────────┐
                    │   plan   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌─────│   act    │◄──────────────┐
              │     └────┬─────┘               │
              │          │                     │
              │     ┌────▼─────┐               │
              │     │ observe  │               │
              │     └────┬─────┘               │
              │          │                     │
              │     ┌────▼─────┐  needs_refine │
              │     │  refine  │───────────────┘
              │     └────┬─────┘
              │          │ done
              │     ┌────▼─────┐
              └─────│ respond  │
                    └──────────┘
```

### 4.2 Node Descriptions

| Node | Responsibility |
|------|---------------|
| `plan` | Decompose task into steps; select tools for each step; populate `plan` and `tool_calls_pending` |
| `act` | Execute pending tool calls via MCP client; populate `tool_results` and `tool_history` |
| `observe` | Parse tool outputs; detect errors, non-convergence, missing data; populate `observation` and `refinement_needed` |
| `refine` | Decide corrective action (retry with new params, use different tool, ask user); update plan; increment `retry_count` |
| `respond` | Synthesize final answer from tool results; populate `final_response` and `task_complete` |

### 4.3 Conditional Edges

- `observe` → `refine` when `refinement_needed == True` AND `retry_count < max_retries`
- `observe` → `respond` when `refinement_needed == False` OR all steps complete
- `refine` → `act` (loop back)
- `plan` → [`human_approval`] → `act` when a pending action requires user confirmation (file deletion, code overwrite, external request, git commit, high-cost computation)

### 4.4 Human-in-the-Loop

Before executing any **destructive or high-cost tool call**, the graph pauses at a built-in approval gate:

```python
DESTRUCTIVE_ACTIONS = {
    "filesystem.delete", "filesystem.write",
    "git.commit", "git.push",
    "dmrg-runner.run",  # expensive computation
}
```

The CLI prompts the user: `Allow dmrg-runner.run with params {M=200, model="H4"}? [y/N/rationale]`. Decision is recorded in `approved_actions`.

## 5. Skill Pack Interface

### 5.1 skill.yaml Format

```yaml
name: dmrg
version: "0.1.0"
description: "DMRG/NQS computational chemistry agent"

system_prompt: |
  You are Tower Agent, running in DMRG mode.

  ## Tools Available
  - dmrg-runner: Run DMRG calculations
  - dmrg-knowledge: Search papers, retrieve code templates
  - dmrg-analyzer: Check convergence, compare energies, plot spectra
  - filesystem: Read/write files
  - python-runner: Execute Python code
  - git: Version control

  ## Rules
  - Validate input files before running calculations
  - Code-review all generated code before execution
  - Cite sources when choosing parameters
  - Flag uncertain results explicitly

rules:
  - "Always validate input files before computation"
  - "Review generated code before execution"
  - "Cite parameter choices with references"
  - "Flag uncertain results as [UNCERTAIN]"

mcp_servers:
  - name: dmrg-runner
    command: ["python", "-m", "dmrg_skill.mcp_servers.runner"]
  - name: dmrg-knowledge
    command: ["python", "-m", "dmrg_skill.mcp_servers.knowledge"]
  - name: dmrg-analyzer
    command: ["python", "-m", "dmrg_skill.mcp_servers.analyzer"]

tools:
  - name: dmrg-runner.run
    intent_template: "Run DMRG calculation with {params}"
  - name: dmrg-knowledge.search
    intent_template: "Search DMRG knowledge base for {query}"
  - name: dmrg-analyzer.check-convergence
    intent_template: "Check DMRG convergence in {output_file}"

knowledge_bases:
  - name: dmrg-papers
    type: pgvector
    embedding_model: "text-embedding-3-small"
    seed_file: "./knowledge/seed_papers.json"
  - name: dmrg-code-templates
    type: pgvector
    embedding_model: "text-embedding-3-small"
    seed_file: "./knowledge/code_templates.json"

workflows:
  - name: convergence-check
    module: "dmrg_skill.workflows.convergence"
    description: "Run DMRG, check convergence, increase M if not converged"
  - name: paper-review
    module: "dmrg_skill.workflows.paper_review"
    description: "Search, download, parse, and summarize a paper"

eval_sets:
  - name: code-tasks
    file: "./eval/code_tasks.yaml"
  - name: compute-tasks
    file: "./eval/compute_tasks.yaml"
```

### 5.2 Skill Pack as Python Package

```toml
# skills/dmrg/pyproject.toml
[project]
name = "tower-skill-dmrg"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "tower>=0.1.0",
    "mcp>=1.0.0",
    "langgraph>=0.2.0",
]

[project.entry-points."tower.skills"]
dmrg = "dmrg_skill:create_skill"
```

This is a standard Python package — installable, versioned, testable independently of tower core.

## 6. MCP Integration

### 6.1 Tool Discovery & Execution

```python
# tower/runtime/mcp_client.py (sketch)
class MCPClientManager:
    """Manages multiple MCP server connections (stdio/sse)."""

    def __init__(self, server_configs: list[MCPConfig]):
        self.servers = {cfg.name: MCPClient(cfg) for cfg in server_configs}

    async def discover_tools(self) -> list[ToolSchema]:
        """Fetch tool schemas from all connected servers."""
        ...

    async def call_tool(self, tool_name: str, params: dict) -> ToolResult:
        """Route tool call to the correct MCP server, execute, return result."""
        ...

    def to_langchain_tools(self) -> list[BaseTool]:
        """Convert MCP tool schemas → LangChain tools for agent use."""
        ...
```

### 6.2 DM Domain MCP Servers

| Server | Type | Tools |
|--------|------|-------|
| `dmrg-runner` | stdio | `run`, `stop`, `get-status`, `parse-log` |
| `dmrg-knowledge` | stdio | `search-papers`, `get-code-template`, `get-formula` |
| `dmrg-analyzer` | stdio | `check-convergence`, `compare-energies`, `plot-spectrum` |

Plus universal MCP servers (filesystem, git) that ship with tower.

## 7. Tracing & Observability

### 7.1 Event Schema

```python
@dataclass
class TraceEvent:
    trace_id: str
    timestamp: str
    node: str                      # plan | act | observe | refine | respond
    event_type: str                # node_enter | node_exit | tool_call | tool_result | error
    state_snapshot: dict           # abridged state at this point
    tool_calls: list[dict] | None  # tools invoked in this step
    token_usage: dict | None       # {input_tokens, output_tokens, model}
    error: str | None
    duration_ms: int
```

### 7.2 CLI Commands

```bash
tower trace list                    # List recent traces
tower trace show <trace-id>         # Show full trace
tower trace playback <trace-id>     # Replay trace against current code
```

## 8. Evaluation System

### 8.1 Eval Task Format

```yaml
# skills/dmrg/eval/compute_tasks.yaml
tasks:
  - id: h4-ground-state
    task: "Compute the ground state energy of an H4 chain with M=100"
    expected_tools: ["dmrg-runner.run"]
    expected_output_contains: ["ground state energy", "bond dimension"]
    success_criteria:
      type: "numeric_range"
      field: "energy"
      range: [-2.0, -1.8]
      converged: true

  - id: hubard-1d-code-gen
    task: "Generate DMRG code for a 1D Hubbard model"
    expected_tools: ["dmrg-knowledge.get-code-template"]
    expected_output_contains: ["ITensor", "MPO", "sweep"]
    success_criteria:
      type: "code_runs"
      timeout_seconds: 30
```

### 8.2 Eval Metrics

- **Task completion rate**: % of tasks where agent produced a final response
- **Tool call accuracy**: % of expected tools actually invoked
- **Output quality**: % of tasks where output contains expected strings
- **Domain criteria**: numeric range match, code execution success
- **Efficiency**: average wall time, average token cost per task

### 8.3 CLI

```bash
tower eval dmrg              # Run full dmrg eval suite
tower eval dmrg --task h4    # Run single task
tower eval dmrg --compare    # Compare against last run
```

## 9. CLI Design

```bash
# Autonomous task execution
tower run "Compute H4 chain ground state energy with DMRG, M=100"
tower run --skill dmrg "Generate code for 1D Hubbard model"
tower run --no-approval "..."  # Skip approval prompts (for trusted tasks)

# Interactive chat
tower chat              # Default skill
tower chat --skill dmrg # With dmrg skill loaded

# Evaluation
tower eval dmrg
tower eval dmrg --output json

# Tracing
tower trace list
tower trace show <id>
tower trace playback <id>

# Skill management
tower skill list        # List installed skills
tower skill info dmrg   # Show skill details
```

## 10. Development Phases

### Phase 1: Core Runtime (2-3 weeks)
- LangGraph orchestration with plan→act→observe→refine→respond
- AgentState schema
- MCP client manager (stdio transport)
- 3 built-in tools: filesystem, python_runner, git
- SQLite session memory
- Structured tracing to file
- `tower run` and `tower chat` CLI

### Phase 2: Skill Pack System (1-2 weeks)
- skill.yaml loader and validator
- Skill-to-runtime wiring (system prompt assembly, MCP server startup, knowledge base connection)
- `tower --skill dmrg` flag
- Skill Pack entry-point discovery

### Phase 3: DMRG Skill Pack (2-3 weeks)
- dmrg-runner MCP server (wrap existing pydmrg/driver.py)
- dmrg-knowledge MCP server (paper search, code templates)
- dmrg-analyzer MCP server (convergence check, energy comparison)
- 2-3 DMRG workflows (convergence check, paper review)
- Seed knowledge base (10-20 curated papers, 5-10 code templates)
- 30+ eval tasks across code, compute, paper categories

### Phase 4: Memory & Retrieval (1-2 weeks)
- pgvector for persistent knowledge
- Embedding-based retrieval integration into plan/act nodes
- Session→persistent memory promotion

### Phase 5: Approval & Safety (1 week)
- Destructive action detection
- Human-in-the-loop node
- Approval CLI prompt

### Phase 6: Eval & Polish (1-2 weeks)
- eval runner with metrics
- Trace playback
- Documentation (README, architecture, skill-pack guide)
- Demo recording / walkthrough

## 11. Design Decisions & Trade-offs

### Why LangGraph, not just ReAct loop?
A ReAct loop (think → act → observe → repeat) is a single implicit loop. LangGraph gives explicit nodes and edges, making the control flow auditable, testable, and extensible (add approval nodes, parallel branches, subgraphs). This is critical for evaluation — you can assert "the agent visited the plan node before the act node."

### Why MCP, not direct tool integration?
MCP decouples tools from the runtime. Adding a new tool does not change tower core. It also makes the project model-agnostic — the same MCP tools work with Claude, GPT, or any other provider.

### Why Skill Packs, not hardcoded sub-agents?
Skill Packs make domain specialization a packaging problem, not an architecture problem. A new domain = a new pip-installable package. This is the same design pattern as Claude Code's skills/slash-commands, but more structured (system prompt + tools + workflows + evals bundled together).

### Why CLI-only first?
A web UI adds significant complexity (auth, state management across sessions, real-time streaming) without improving the agent's core capabilities. CLI gets the agent working, and a web UI can be layered on the same runtime later.

## 12. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| LangGraph learning curve | Start with minimal 3-node graph, add complexity incrementally |
| MCP server reliability (stdio process management) | Use subprocess management with health checks and restart logic |
| DMRG computation is slow/unpredictable | Async MCP tool calls; configurable timeouts; dmrg-runner reports progress |
| Skill Pack interface too rigid | Start with a minimal interface (just `create_skill()` → dict), evolve as needs emerge |
| Claude API cost | Implement token tracking from Day 1; support local models via Ollama as opt-in |
