# Tower Refactor — Quantum Chemistry Agent Design Spec

**Date:** 2026-06-17
**Status:** Approved
**Author:** Sun Xinyu

## 1. Overview

Refactor Tower from a generic agent loop to a quantum chemistry computing agent, while keeping the core runtime domain-agnostic. Two primary changes:

1. **Specialize the Plan→Act→Observe→Refine→Respond state machine** for quantum chemistry: plan designs calculation workflows, act runs calculations in background (interrupt/resume), observe detects domain errors (SCF convergence, MPI, etc.), refine auto-fixes and retries, respond outputs structured results.

2. **Convert memory and store layers to async**, using LangGraph's `AsyncPostgresSaver` and `AsyncPostgresStore`.

### Design Principles

- **Core generic, domain in skill**: Tower core knows nothing about quantum chemistry. All domain logic lives in `skills/<name>/hooks/`.
- **Progressive async**: Only memory layer goes async. Graph nodes stay sync (LangGraph `interrupt()` naturally handles async I/O boundaries).
- **Decoupled extensibility**: Every graph node exposes hook points. Skill packs optionally implement hooks. Unimplemented hooks fall back to default behavior.

## 2. File Structure

```
tower/
├── src/tower/
│   ├── state.py                    # [MODIFY] Extended AgentState
│   ├── memory/
│   │   ├── pool.py                 # [MODIFY] Sync→AsyncConnectionPool
│   │   ├── short_term.py           # [MODIFY] PostgresSaver→AsyncPostgresSaver
│   │   └── long_term.py            # [MODIFY] PostgresStore→AsyncPostgresStore
│   ├── graph/
│   │   ├── nodes.py                # [MODIFY] Nodes call skill hooks at injection points
│   │   ├── edges.py                # [MODIFY] Updated routing logic
│   │   └── hooks.py                # [NEW] Hook protocol definitions (ErrorDetector, etc.)
│   ├── runtime/
│   │   ├── orchestrator.py         # [MODIFY] Integrate skill hooks, async checkpointer
│   │   └── skill_loader.py         # [MODIFY] Load skill.yaml + hooks modules
│   └── tools/                      # Unchanged
│
└── skills/                         # [NEW] Independent skill paths
    └── dmrg/
        ├── skill.yaml              # Skill metadata + system prompt
        ├── hooks/
        │   ├── __init__.py
        │   ├── plan.py             # Plan node preprocess/validate
        │   ├── act.py              # Act node param validation / result enrichment
        │   ├── observe.py          # Error detector registry
        │   ├── refine.py           # Fix strategy registry
        │   └── respond.py          # Result formatter
        └── detectors/
            ├── __init__.py
            ├── base.py             # BaseErrorDetector protocol
            ├── scf.py              # [IMPLEMENT] SCF convergence detector (PySCF)
            ├── python_error.py     # [IMPLEMENT] Python traceback/runtime error detector
            ├── mpi.py              # [STUB] MPI error detector (placeholder)
            ├── gaussian.py         # [STUB] Gaussian error detector (placeholder)
            ├── orca.py             # [STUB] ORCA error detector (placeholder)
            └── vasp.py             # [STUB] VASP error detector (placeholder)
```

### Stub File Convention

Placeholder detectors (`mpi.py`, `gaussian.py`, `orca.py`, `vasp.py`) contain only a class skeleton:

```python
# skills/dmrg/detectors/gaussian.py  (stub — not yet implemented)
from tower.graph.hooks import ErrorDetector, ErrorInfo

class GaussianErrorDetector:
    """Detect Gaussian calculation errors.

    TODO: Implement when Gaussian support is added.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
```

This makes it trivially easy for developers to add new software support: copy a stub, implement `detect()`, register in `hooks/observe.py`.

## 3. Agent State Schema

```python
class AgentState(TypedDict, total=False):
    # ── Messages ──
    messages: Annotated[list, add_messages]

    # ── Task & Plan ──
    task: str
    plan: list[dict]
    # plan[i] = {
    #     "name": str,              # tool name
    #     "args": dict,             # tool parameters
    #     "id": str,                # LLM-generated tool_call_id
    #     "step_type": str,         # [NEW] "computation" | "analysis" | "io"
    #     "expected_output": str,   # [NEW] human-readable expected result
    # }
    current_step_index: int

    # ── Tool execution ──
    tool_results: dict[str, Any]
    tool_history: list[dict]        # [NEW] explicit history, not inferred from messages

    # ── Observation & Error ──
    observation: str
    error_info: dict | None         # [NEW] structured error from detectors
    # error_info = {
    #     "error_type": str,        # "scf_not_converged" | "python_error" | "mpi_error" | ...
    #     "message": str,
    #     "suggestion": str,        # human-readable fix suggestion
    #     "can_auto_fix": bool,     # True if refine can auto-correct
    # }
    refinement_needed: bool
    retry_count: int
    max_retries: int
    retry_pending: bool

    # ── Background task ──
    background_task: dict | None    # [NEW]
    # background_task = {
    #     "task_id": str,
    #     "status": "running" | "completed" | "failed",
    #     "started_at": str,        # ISO timestamp
    # }

    # ── Multi-pass control ──
    pass_count: int

    # ── Final ──
    final_response: str
    task_complete: bool
    node_history: list[str]

    # ── Runtime injection (not persisted) ──
    _runtime: dict
```

### Key Changes from Current State

| Change | Rationale |
|--------|-----------|
| `tool_history: list[dict]` | Explicitly track all tool invocations, decoupled from message list |
| `error_info: dict \| None` | Domain error detectors produce structured errors; refine uses `error_type` to select fix strategy |
| `background_task: dict \| None` | Track long-running calculations across interrupt/resume boundaries |
| `plan[].step_type` | Help observe node distinguish computation steps (deep error detection) from I/O steps (simple check) |
| `plan[].expected_output` | Guide observe node and respond node for quality assessment |

## 4. Hook Protocol

Defined in `src/tower/graph/hooks.py`. Each node has a corresponding hook protocol. Skills implement them optionally.

### 4.1 Error Detection Base

```python
@dataclass
class ErrorInfo:
    error_type: str          # "scf_not_converged" | "python_error" | "mpi_error" | ...
    message: str             # human-readable description
    suggestion: str          # fix suggestion for the user/refine node
    can_auto_fix: bool       # True → refine node can auto-correct and retry


@dataclass
class CorrectionAction:
    action: str              # "retry_with_params" | "skip" | "ask_user"
    new_params: dict | None  # corrected parameters (for retry_with_params)


class ErrorDetector(Protocol):
    """Detect a specific class of computation errors.

    Each software (PySCF, Gaussian, ORCA, ...) and error type (SCF, MPI, OOM, ...)
    gets its own detector class. Observe node runs all registered detectors
    against the computation output.
    """
    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None: ...
```

### 4.2 Per-Node Hook Protocols

```python
# ── Plan node ──
class PlanHooks(Protocol):
    def preprocess(self, task: str, state: dict) -> str: ...
    def validate_plan(self, plan: list[dict]) -> list[dict]: ...

# ── Act node ──
class ActHooks(Protocol):
    def pre_run(self, tool_call: dict, state: dict) -> dict: ...
    def post_run(self, result: dict, state: dict) -> dict: ...

# ── Observe node ──
class ObserveHooks(Protocol):
    def get_error_detectors(self) -> list[ErrorDetector]: ...

# ── Refine node ──
class RefineHooks(Protocol):
    def get_fix_strategies(self) -> dict[str, callable]: ...
    # Returns {"scf_not_converged": fix_fn, "python_error": fix_fn, ...}
    # Each fix_fn: (ErrorInfo, step: dict) → CorrectionAction

# ── Respond node ──
class RespondHooks(Protocol):
    def format_result(self, results: dict, state: dict) -> str: ...
```

### 4.3 Hook Loading

`SkillLoader` loads hooks from `skills/<name>/hooks/`:

```python
class Skill:
    name: str
    system_prompt: str
    plan_hooks: PlanHooks | None = None
    act_hooks: ActHooks | None = None
    observe_hooks: ObserveHooks | None = None
    refine_hooks: RefineHooks | None = None
    respond_hooks: RespondHooks | None = None
```

If a hook module is missing or the class doesn't implement the protocol, the field stays `None` and the node uses its default behavior.

## 5. Graph Topology

### 5.1 Node Flow

```
                         ┌──────────┐
                         │   plan   │  Design calculation plan
                         └────┬─────┘
                              │
                     ┌────────▼────────┐
                     │    approval     │  User confirmation (optional)
                     └────────┬────────┘
                              │
                         ┌────▼─────┐
              ┌─────────│   act    │  Start background calc → interrupt
              │         └────┬─────┘
              │              │
              │     ┌────────▼────────┐
              │     │  [graph paused] │  CLI polls task status
              │     └────────┬────────┘
              │              │ resume
              │         ┌────▼─────┐
              │         │ observe  │  Domain error detection
              │         └────┬─────┘
              │              │
              │         ┌────▼─────┐  needs_refine && retries_left
              │         │  refine  │──────────────────┐
              │         └────┬─────┘                  │
              │              │ done                   │
              │         ┌────▼─────┐                  │
              └─────────│ respond  │                  │
                        └────┬─────┘                  │
                             │                        │
                             ▼                        │
                           END      ◄─────────────────┘
```

### 5.2 Act Node: Interrupt/Resume

```python
def act_node(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    skill = state["_runtime"].get("skill")

    if idx >= len(plan):
        return {"node_history": [...]}

    tc = plan[idx]

    # 1. Skill hook: pre-run param validation
    if skill and skill.act_hooks:
        tc["args"] = skill.act_hooks.pre_run(tc, state)

    # 2. Execute tool
    result = execute_tool(tc)

    # 3. If tool returned a background task, pause graph
    if result.get("background_task"):
        task_id = result["background_task"]["task_id"]
        final_result = interrupt({
            "type": "background_task",
            "task_id": task_id,
        })
        # ← graph resumes here with Command(resume=final_result)
        result = final_result

    # 4. Skill hook: post-run result enrichment
    if skill and skill.act_hooks:
        result = skill.act_hooks.post_run(result, state)

    return {
        "tool_results": {**state["tool_results"], f"step_{idx}": result},
        "current_step_index": idx + 1,
        "background_task": None,
        "node_history": state.get("node_history", []) + ["act"],
    }
```

### 5.3 CLI Polling Loop

```python
# main.py — stream-based execution with interrupt handling
config = {"configurable": {"thread_id": session_id}}

for event in graph.stream(initial_state, config):
    if event.get("__interrupt__"):
        interrupt_data = event["__interrupt__"][0].value

        if interrupt_data["type"] == "background_task":
            task_id = interrupt_data["task_id"]

            # Poll until completion
            result = await poll_task(task_id)

            # Resume graph with computation result
            for event in graph.stream(Command(resume=result), config):
                yield event
    else:
        yield event
```

### 5.4 Routing Logic

```python
# edges.py — updated conditional routing

def route_after_plan(state) -> str:
    if state.get("task_complete"):
        return "respond"
    plan = state.get("plan", [])
    return "act" if plan else "respond"

def route_after_observe(state) -> str:
    if state.get("refinement_needed") and state["retry_count"] < state["max_retries"]:
        return "refine"
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    if idx >= len(plan):
        return "respond"
    return "act"

def route_after_refine(state) -> str:
    error = state.get("error_info", {})
    if error.get("can_auto_fix") and state.get("retry_count", 0) < state.get("max_retries", 3):
        return "act"
    return "respond"
```

## 6. Observe → Refine Error Handling Loop

### 6.1 Observe Node

```
act result
    ↓
┌─────────────────────────────────────────────┐
│ observe_node                                │
│                                             │
│ 1. Basic check: result["error"] / returncode│
│ 2. Get detectors from skill hook:           │
│    detectors = skill.observe_hooks          │
│                   .get_error_detectors()    │
│ 3. Run each detector against stdout/stderr: │
│    detector.detect(stdout, stderr, rc)      │
│    ├── ScfConvergenceDetector (PySCF)       │
│    ├── PythonErrorDetector                  │
│    ├── MpiErrorDetector (stub)             │
│    ├── GaussianErrorDetector (stub)        │
│    ├── OrcaErrorDetector (stub)            │
│    └── VaspErrorDetector (stub)            │
│ 4. First match → state["error_info"]        │
│ 5. Set refinement_needed accordingly        │
└─────────────────────────────────────────────┘
```

### 6.2 Implemented Detectors

#### ScfConvergenceDetector (`skills/dmrg/detectors/scf.py`)

Detects PySCF SCF convergence failure patterns:

- `"SCF not converged"` → extracts iteration count, suggests increasing M or adjusting damping
- Supports both PySCF and ORCA SCF message formats (ORCA detection logic is a stub)

#### PythonErrorDetector (`skills/dmrg/detectors/python_error.py`)

Detects Python runtime errors in computation output:

- `Traceback (most recent call last)` → extracts exception type and line
- `SyntaxError`, `ImportError`, `ModuleNotFoundError` → can_auto_fix = False (needs code change)
- `MemoryError` → can_auto_fix = True (reduce bond dimension)

### 6.3 Refine Node

```
error_info set
    ↓
┌───────────────────────────────────────────┐
│ refine_node                               │
│                                           │
│ 1. Check retry_count < max_retries        │
│ 2. Get fix strategies from skill hook:    │
│    strategies = skill.refine_hooks        │
│                   .get_fix_strategies()   │
│ 3. Look up strategy by error_type:        │
│    "scf_not_converged" → increase M       │
│    "python_error" → parse error, decide   │
│    "mpi_error" → ask_user (stub)        │
│ 4. If can_auto_fix:                       │
│    - Update plan[idx-1]["args"] with      │
│      corrected parameters                 │
│    - Return retry_pending=True            │
│ 5. If cannot auto-fix:                    │
│    - Set refinement_needed=False           │
│    - Route to respond with error message  │
└───────────────────────────────────────────┘
```

### 6.4 Fix Strategies (Initial Implementation)

| error_type | Strategy | Action |
|-----------|----------|--------|
| `scf_not_converged` | Double bond dimension M (capped at 2000) | `retry_with_params` |
| `python_error` (MemoryError) | Halve bond dimension M (min 50) | `retry_with_params` |
| `python_error` (SyntaxError/ImportError) | Cannot auto-fix | `ask_user` |
| `mpi_error` (stub) | Cannot auto-fix | `ask_user` |
| `gaussian_error` (stub) | Not implemented | `ask_user` |
| `orca_error` (stub) | Not implemented | `ask_user` |

### 6.5 Refine Node Logic

```python
def refine_node(state: AgentState) -> dict:
    error_info = state.get("error_info", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    plan = list(state.get("plan", []))
    idx = state.get("current_step_index", 0)
    skill = state["_runtime"].get("skill")

    # 1. Max retries exceeded → give up
    if retry_count >= max_retries:
        return {
            "refinement_needed": False,
            "final_response": f"步骤失败，已重试 {max_retries} 次：{error_info.get('message', '')}",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    # 2. Look up fix strategy from skill
    correction = None
    if skill and skill.refine_hooks:
        strategies = skill.refine_hooks.get_fix_strategies()
        strategy_fn = strategies.get(error_info.get("error_type", ""))
        if strategy_fn:
            correction = strategy_fn(error_info, plan[idx - 1] if idx > 0 else {})

    # 3. Apply correction
    if correction and correction.action == "retry_with_params" and correction.new_params:
        if idx > 0:
            plan[idx - 1]["args"] = correction.new_params
        return {
            "plan": plan,
            "retry_count": retry_count + 1,
            "retry_pending": True,
            "node_history": state.get("node_history", []) + ["refine"],
        }

    # 4. Cannot fix → ask user via interrupt, or fallback to LLM
    if correction and correction.action == "ask_user":
        return {
            "refinement_needed": False,
            "final_response": f"计算失败：{error_info.get('message')}。建议：{error_info.get('suggestion')}",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    # 5. No strategy → fallback to LLM-based correction (existing logic)
    ...
```

## 7. Async Memory Layer

### 7.1 pool.py — AsyncConnectionPool

```python
# Before: sync psycopg_pool.ConnectionPool
# After: async psycopg_pool.AsyncConnectionPool

from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None

async def get_pool(conn_string: str) -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conn_string, min_size=1, max_size=3, open=True,
        )
    return _pool

async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
```

### 7.2 short_term.py — AsyncPostgresSaver

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def create_checkpointer(conn_string: str) -> AsyncPostgresSaver:
    pool = await get_pool(conn_string)
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver
```

### 7.3 long_term.py — AsyncPostgresStore

All methods become async (`aput`, `aget`, `asearch`). Lazy initialization via `_ensure_setup()`.

```python
from langgraph.store.postgres.aio import AsyncPostgresStore

class LongTermMemory:
    def __init__(self, conn_string: str, user_id: str = "default"):
        self._conn_string = conn_string
        self._store: AsyncPostgresStore | None = None
        self._namespace = ("users", user_id)

    async def _ensure_setup(self):
        if self._store is None:
            pool = await get_pool(self._conn_string)
            self._store = AsyncPostgresStore(pool)
            await self._store.setup()

    async def put(self, key: str, value: dict): ...
    async def get(self, key: str) -> dict | None: ...
    async def search(self, filter_keys=None, limit=50) -> list[dict]: ...
    async def add_fact(self, fact: str, category: str = "general"): ...
```

### 7.4 Graph Nodes Stay Sync

Graph nodes (`plan_node(state) → dict`) remain synchronous. Memory access patterns:

- **Checkpointer**: Managed by LangGraph automatically on `graph.invoke()` / `graph.astream()` — no node code change needed.
- **LongTermMemory**: If a node needs long-term memory, the orchestrator pre-loads data into state before `graph.invoke()`, or the node uses `asyncio.run()` for simple reads. Full async node support can be added later.

### 7.5 CLI Entry Point

```python
# main.py
def main():
    asyncio.run(async_main())

async def async_main():
    checkpointer = await create_checkpointer(conn_string)
    graph = build_graph(checkpointer=checkpointer)
    # ... CLI logic
```

## 8. skill.yaml Updates

```yaml
name: dmrg
version: "0.2.0"
description: "DMRG/NQS quantum chemistry computational agent"

system_prompt: |
  You are a quantum chemistry computational agent.
  Design calculation plans, execute them, detect errors, and report results.

## Rules
- Validate input parameters before running calculations
- Check SCF convergence after every calculation
- Auto-fix convergence issues when possible
- Report energies with units and precision

hooks:
  package: "dmrg_skill.hooks"    # Python package for hook modules

mcp_servers:
  - name: dmrg-runner
    command: ["python", "-m", "dmrg_skill.mcp_servers.runner"]

tools:
  - name: dmrg-runner.run
    intent_template: "Run DMRG calculation with {params}"
```

## 9. Development Phases (Refactor)

### Phase A: Core Infrastructure (async memory + hook protocol)
- Convert `pool.py` to `AsyncConnectionPool`
- Convert `short_term.py` to `AsyncPostgresSaver`
- Convert `long_term.py` to `AsyncPostgresStore`
- Create `graph/hooks.py` with hook protocols
- Update `skill_loader.py` to load hook modules
- **Files**: `memory/*`, `graph/hooks.py`, `runtime/skill_loader.py`

### Phase B: State & Graph Updates
- Extend `AgentState` with `tool_history`, `error_info`, `background_task`
- Update `act_node` with interrupt/resume support
- Update `observe_node` with skill detector pipeline
- Update `refine_node` with fix strategy dispatch
- Update `edges.py` routing logic
- **Files**: `state.py`, `graph/nodes.py`, `graph/edges.py`, `runtime/orchestrator.py`

### Phase C: Skill Pack Structure
- Create `skills/dmrg/` directory structure
- Create `skill.yaml`
- Implement `skills/dmrg/hooks/` modules for all 5 nodes
- Implement detectors: `scf.py`, `python_error.py`
- Create stubs: `mpi.py`, `gaussian.py`, `orca.py`, `vasp.py`
- **Files**: `skills/dmrg/**`

### Phase D: Integration & Testing
- Update CLI for async entry + interrupt/resume loop
- Integration tests with mock LLM + background tasks
- Test SCF convergence detection with real PySCF output samples
- Test fix strategy dispatch
- **Files**: `main.py`, `tests/`

## 10. Design Decisions

### Why interrupt/resume instead of asyncio subprocess?
LangGraph's `interrupt()` saves a checkpoint before pausing. If the CLI process crashes during a multi-hour calculation, the user can resume from the checkpoint — no lost state. `asyncio.create_subprocess_exec` doesn't provide this durability.

### Why keep graph nodes sync?
The progressive approach minimizes risk. LangGraph's `interrupt()` and async `astream()` handle the async I/O boundary. Making nodes `async def` can be done later when MCP servers need async communication.

### Why detectors as separate classes instead of YAML regex patterns?
Regex patterns work for simple cases but fail for multi-line context (e.g., MPI errors often span several lines with rank/traceback info). Python classes give full flexibility without over-engineering.

### Why stub files for unimplemented detectors?
They serve as documentation and templates. A developer adding Gaussian support sees `gaussian.py` with the right class skeleton and method signature — zero ambiguity about what to implement.

## 11. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| AsyncConnectionPool API differs from sync | Both are `psycopg_pool`, API surface nearly identical |
| Interrupt/resume breaks existing multi-pass logic | `background_task` is set only for long computations; quick tools (read_file, bash) don't trigger interrupt |
| Skill hook loading fails silently | SkillLoader logs warnings for missing hook modules; nodes check `if skill and skill.xxx_hooks` before calling |
| Detector false positives | Detectors match specific error patterns; observe takes the FIRST match only; ordering in `get_error_detectors()` determines priority |
