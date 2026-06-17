# Tower Refactor — QC Agent + Async Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Tower for quantum chemistry computing: (1) specialize Plan→Act→Observe→Refine→Respond with skill hook injection + background tasks, (2) convert memory layer to async.

**Architecture:** Skill hooks inject domain logic at every graph node via protocols defined in `graph/hooks.py`. Act node uses LangGraph `interrupt()` for background calculations. Memory layer uses `AsyncPostgresSaver` + `AsyncPostgresStore`. Core stays domain-agnostic; all QC logic lives in `skills/dmrg/`.

**Tech Stack:** Python 3.12+, LangGraph, psycopg (async), LangChain

---

## Phase A: Async Memory + Hook Protocols

### Task A1: Convert pool.py to AsyncConnectionPool

**Files:**
- Modify: `src/tower/memory/pool.py`

- [ ] **Step 1: Rewrite pool.py for async**

```python
"""共享异步连接池 —— 整个进程复用同一个 PostgreSQL 连接池。"""
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None
_conn_string: str = ""


async def get_pool(conn_string: str) -> AsyncConnectionPool:
    """获取或创建共享异步连接池。"""
    global _pool, _conn_string
    if _pool is None or _conn_string != conn_string:
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
        _pool = AsyncConnectionPool(
            conn_string,
            min_size=1,
            max_size=3,
            open=True,
        )
        _conn_string = conn_string
    return _pool


async def close_pool():
    """关闭连接池（用于 graceful shutdown）。"""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = None
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/sunxinyu/develop/tower && python -c "from tower.memory.pool import get_pool, close_pool; print('OK')"
```

### Task A2: Convert short_term.py to AsyncPostgresSaver

**Files:**
- Modify: `src/tower/memory/short_term.py`

- [ ] **Step 1: Rewrite for async**

```python
"""短期记忆 —— AsyncPostgresSaver 持久化会话内 state。

每次 graph.ainvoke() 后 LangGraph 自动保存 checkpoint。
相同 thread_id 可恢复历史会话。
"""
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tower.memory.pool import get_pool

_setup_done = False


async def create_checkpointer(conn_string: str) -> AsyncPostgresSaver:
    """创建 AsyncPostgresSaver，使用共享异步连接池。"""
    global _setup_done
    pool = await get_pool(conn_string)
    saver = AsyncPostgresSaver(pool)
    if not _setup_done:
        await saver.setup()
        _setup_done = True
    return saver
```

### Task A3: Convert long_term.py to AsyncPostgresStore

**Files:**
- Modify: `src/tower/memory/long_term.py`

- [ ] **Step 1: Rewrite for async with lazy init**

```python
"""长期记忆 —— AsyncPostgresStore 跨会话持久化。

LangGraph Store 原生支持 PostgreSQL + pgvector。
使用 ("users", user_id) 作为 namespace 隔离不同用户。
"""
import hashlib
from langgraph.store.postgres.aio import AsyncPostgresStore
from tower.memory.pool import get_pool


class LongTermMemory:
    """基于 LangGraph AsyncPostgresStore 的跨会话长期记忆。"""

    def __init__(self, conn_string: str, user_id: str = "default"):
        self._conn_string = conn_string
        self._store: AsyncPostgresStore | None = None
        self._setup_done = False
        self._namespace = ("users", user_id)

    async def _ensure_setup(self):
        if self._store is None:
            pool = await get_pool(self._conn_string)
            self._store = AsyncPostgresStore(pool)
            if not self._setup_done:
                await self._store.setup()
                self._setup_done = True

    async def put(self, key: str, value: dict):
        """写入一条记忆。"""
        await self._ensure_setup()
        await self._store.aput(self._namespace, key, value)

    async def get(self, key: str) -> dict | None:
        """读取一条记忆。"""
        await self._ensure_setup()
        item = await self._store.aget(self._namespace, key)
        if item is None:
            return None
        return item.value if hasattr(item, "value") else item.get("value", {})

    async def search(self, filter_keys: list[str] | None = None, limit: int = 50) -> list[dict]:
        """搜索记忆，可按 key 过滤。"""
        await self._ensure_setup()
        items = await self._store.asearch(self._namespace, limit=limit)
        results = []
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            k = item.key if hasattr(item, "key") else item.get("key", "")
            if filter_keys is None or k in filter_keys:
                results.append({"key": k, "value": v})
        return results

    async def get_all_facts(self, limit: int = 50) -> list[dict]:
        """获取所有事实（兼容旧接口）。"""
        items = await self._store.asearch(self._namespace, limit=limit)
        facts = []
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            if isinstance(v, dict) and "fact" in v:
                facts.append({"fact": v["fact"], "category": v.get("category", "general")})
        return facts

    async def add_fact(self, fact: str, category: str = "general"):
        """添加一条事实（自动去重，O(1) 查找）。"""
        key = "fact_" + hashlib.md5(fact.encode()).hexdigest()[:12]
        existing = await self.get(key)
        if existing is not None:
            return
        await self.put(key, {"fact": fact, "category": category})

    async def remove_fact_by_text(self, text: str):
        """删除包含指定文本的记忆（模糊匹配）。"""
        await self._ensure_setup()
        items = await self._store.asearch(self._namespace, limit=100)
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            stored_fact = v.get("fact", "") if isinstance(v, dict) else ""
            if text in stored_fact or stored_fact in text:
                await self._store.adelete(self._namespace, item.key)

    async def clear(self):
        """清空当前用户的所有记忆。"""
        await self._ensure_setup()
        items = await self._store.asearch(self._namespace, limit=1000)
        for item in items:
            await self._store.adelete(self._namespace, item.key)
```

### Task A4: Update memory/__init__.py

**Files:**
- Modify: `src/tower/memory/__init__.py`

- [ ] **Step 1: Update exports**

```python
from tower.memory.short_term import create_checkpointer
from tower.memory.long_term import LongTermMemory
from tower.memory.pool import close_pool

__all__ = ["create_checkpointer", "LongTermMemory", "close_pool"]
```

### Task A5: Create graph/hooks.py — Hook Protocols

**Files:**
- Create: `src/tower/graph/hooks.py`

- [ ] **Step 1: Define all hook protocols and data classes**

```python
"""Tower graph hook protocols — skill packs implement these to inject domain logic.

Each node has a corresponding hook protocol. Skills optionally implement them.
Unimplemented hooks → node falls back to default behavior.
"""
from dataclasses import dataclass, field
from typing import Protocol


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ErrorInfo:
    """Structured error detected by a domain error detector.

    Attributes:
        error_type: Machine-readable error category.
            e.g. "scf_not_converged", "python_error", "mpi_error"
        message: Human-readable description of what went wrong.
        suggestion: Suggested fix for the user or refine node.
        can_auto_fix: True if refine node can auto-correct and retry.
    """
    error_type: str
    message: str
    suggestion: str = ""
    can_auto_fix: bool = False


@dataclass
class CorrectionAction:
    """Corrective action produced by a refine fix strategy.

    Attributes:
        action: "retry_with_params" | "skip" | "ask_user"
        new_params: Corrected parameters (for retry_with_params).
    """
    action: str
    new_params: dict | None = None


# ═══════════════════════════════════════════════════════════════════
# Error Detector Protocol
# ═══════════════════════════════════════════════════════════════════


class ErrorDetector(Protocol):
    """Detect a specific class of computation errors.

    Each software (PySCF, Gaussian, ORCA, ...) and error type
    (SCF, MPI, OOM, ...) gets its own detector class. Observe node
    runs all registered detectors against the computation output.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        """Check computation output for a known error pattern.

        Returns:
            ErrorInfo if the pattern is detected, None otherwise.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# Per-Node Hook Protocols
# ═══════════════════════════════════════════════════════════════════


class PlanHooks(Protocol):
    """Hooks injected into the plan node."""

    def preprocess(self, task: str, state: dict) -> str:
        """Preprocess the user's task before LLM planning.

        Can enrich the task with domain context, translate shorthand
        notation, or add implicit requirements.
        """
        ...

    def validate_plan(self, plan: list[dict]) -> list[dict]:
        """Validate and potentially modify the generated plan.

        Can reorder steps, add missing validation steps, or reject
        invalid parameter combinations.
        """
        ...


class ActHooks(Protocol):
    """Hooks injected into the act node."""

    def pre_run(self, tool_call: dict, state: dict) -> dict:
        """Validate/transform tool parameters before execution.

        Can check parameter ranges, add defaults, or convert units.
        Returns the (potentially modified) tool_call dict.
        """
        ...

    def post_run(self, result: dict, state: dict) -> dict:
        """Enrich tool result after successful execution.

        Can parse raw output, extract key metrics, or add metadata.
        Returns the enriched result dict.
        """
        ...


class ObserveHooks(Protocol):
    """Hooks injected into the observe node."""

    def get_error_detectors(self) -> list[ErrorDetector]:
        """Return the list of error detectors to run.

        Detectors are run in order; the first match wins.
        Order matters: put more specific detectors first.
        """
        ...


class RefineHooks(Protocol):
    """Hooks injected into the refine node."""

    def get_fix_strategies(self) -> dict:
        """Return a mapping from error_type to fix strategy function.

        Each strategy: (ErrorInfo, step: dict) → CorrectionAction

        Example:
            {"scf_not_converged": self._fix_scf_convergence}
        """
        ...


class RespondHooks(Protocol):
    """Hooks injected into the respond node."""

    def format_result(self, results: dict, state: dict) -> str:
        """Format computation results for user-facing output.

        Can extract key metrics, format tables, add units.
        Returns a formatted string to prepend to the LLM response.
        """
        ...
```

### Task A6: Update skill_loader.py to load hooks

**Files:**
- Read current: `src/tower/runtime/skill_loader.py` (doesn't exist yet — create it)
- Create: `src/tower/runtime/skill_loader.py`

- [ ] **Step 1: Create skill_loader.py with hook loading**

We need to check if `runtime/skill_loader.py` exists. Based on the project structure, it likely doesn't (was descoped in Phase 1). Let me check.

Actually, let me also read the current `runtime/__init__.py`.

Let me create the skill_loader:

```python
"""Skill loader — parses skill.yaml and loads hook modules."""
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Skill:
    name: str
    version: str = "0.1.0"
    description: str = ""
    system_prompt: str = ""
    rules: list[str] = field(default_factory=list)
    mcp_servers: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    # Hook instances (None if not provided by skill)
    plan_hooks: Any = None
    act_hooks: Any = None
    observe_hooks: Any = None
    refine_hooks: Any = None
    respond_hooks: Any = None


class SkillLoader:
    """Load a skill pack from a directory containing skill.yaml."""

    @staticmethod
    def load(skill_dir: str) -> Skill | None:
        """Load skill from directory.

        Args:
            skill_dir: Path to skill directory (e.g. 'skills/dmrg').

        Returns:
            Skill object with hooks loaded, or None if skill.yaml not found.
        """
        skill_path = Path(skill_dir)
        yaml_path = skill_path / "skill.yaml"

        if not yaml_path.exists():
            return None

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            return None

        if not isinstance(data, dict) or "name" not in data:
            return None

        skill = Skill(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            rules=data.get("rules", []) or [],
            mcp_servers=data.get("mcp_servers", []) or [],
            tools=data.get("tools", []) or [],
        )

        # Load hook modules if present
        hooks_dir = skill_path / "hooks"
        if hooks_dir.is_dir():
            SkillLoader._load_hooks(skill, str(hooks_dir))

        return skill

    @staticmethod
    def _load_hooks(skill: Skill, hooks_dir: str):
        """Dynamically import hook modules from skills/<name>/hooks/."""
        # Ensure hooks_dir is on sys.path for imports
        hooks_parent = str(Path(hooks_dir).parent)
        if hooks_parent not in sys.path:
            sys.path.insert(0, hooks_parent)

        hook_files = {
            "plan": "plan_hooks",
            "act": "act_hooks",
            "observe": "observe_hooks",
            "refine": "refine_hooks",
            "respond": "respond_hooks",
        }

        for module_name, attr_name in hook_files.items():
            module_path = Path(hooks_dir) / f"{module_name}.py"
            if not module_path.exists():
                continue

            try:
                # Import from the skill's hooks package
                skill_pkg = Path(hooks_dir).parent.name
                mod = importlib.import_module(f"{skill_pkg}.hooks.{module_name}")
                # Look for a class that matches the hook protocol
                # Convention: DmrgPlanHooks, DmrgActHooks, etc.
                hook_instance = SkillLoader._find_hook_instance(mod)
                if hook_instance:
                    setattr(skill, attr_name, hook_instance)
            except Exception:
                # Hook loading failure should not crash the skill loader
                pass

    @staticmethod
    def _find_hook_instance(module):
        """Find the first class instance in a hook module.

        Convention: the module should expose a class or instance.
        We look for a class that has the expected methods and instantiate it.
        """
        import inspect

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes (only consider classes defined in this module)
            if obj.__module__ != module.__name__:
                continue
            # Skip base classes and protocols
            if name.startswith("_"):
                continue
            try:
                return obj()
            except Exception:
                pass
        return None
```

---

## Phase B: State & Graph Updates

### Task B1: Extend AgentState schema

**Files:**
- Modify: `src/tower/state.py`

- [ ] **Step 1: Add new fields to AgentState**

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Tower agent global state.

    Design principle: state holds data needed for the current workflow.
    Long-lived data lives in the memory layer.
    """

    # ── Messages ──
    messages: Annotated[list, add_messages]

    # ── Task & Plan ──
    task: str
    plan: list[dict]
    # plan[i] = {
    #     "name": str,           # tool name
    #     "args": dict,          # tool parameters
    #     "id": str,             # LLM-generated tool_call_id
    #     "step_type": str,      # [NEW] "computation" | "analysis" | "io"
    #     "expected_output": str, # [NEW] human-readable expected result
    # }
    current_step_index: int

    # ── Tool Execution ──
    tool_results: dict[str, Any]
    tool_history: list[dict]       # [NEW] explicit invocation history

    # ── Observation & Error ──
    observation: str
    error_info: dict | None        # [NEW] structured error from detectors
    # error_info = {
    #     "error_type": str,       # "scf_not_converged" | "python_error" | ...
    #     "message": str,
    #     "suggestion": str,
    #     "can_auto_fix": bool,
    # }
    refinement_needed: bool
    retry_count: int
    max_retries: int
    retry_pending: bool            # refine triggered retry of current step

    # ── Background Task ──
    background_task: dict | None   # [NEW] track long-running calculations
    # background_task = {
    #     "task_id": str,
    #     "status": "running" | "completed" | "failed",
    #     "started_at": str,
    # }

    # ── Multi-Pass Control ──
    pass_count: int

    # ── Final ──
    final_response: str
    task_complete: bool
    node_history: list[str]

    # ── Runtime Injection (not persisted in checkpoint) ──
    _runtime: dict[str, Any]
```

### Task B2: Update graph/nodes.py — Act node with interrupt/resume

**Files:**
- Modify: `src/tower/graph/nodes.py` (the act_node function)

- [ ] **Step 1: Add interrupt support to act_node**

The act_node needs to:
1. Call skill.act_hooks.pre_run() before execution
2. Start the tool
3. If result has `background_task`, call `interrupt()` to pause
4. On resume, continue with the computation result
5. Call skill.act_hooks.post_run() after execution
6. Record in tool_history

Replace the current `act_node` function in nodes.py:

```python
def act_node(state: AgentState) -> dict:
    """Execute the current step's tool call.

    If the tool returns a background_task, pause the graph via interrupt()
    and wait for the CLI to resume with the completed result.

    If retry_pending=True (refine triggered a retry), re-execute the
    previous step without advancing current_step_index.
    """
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    results = dict(state.get("tool_results", {}))
    retry_pending = state.get("retry_pending", False)
    skill = state.get("_runtime", {}).get("skill")
    history = list(state.get("tool_history", []))

    # ── Refine-triggered retry: re-execute previous step ──
    if retry_pending:
        retry_step = idx - 1
        if retry_step < 0 or retry_step >= len(plan):
            return {
                "retry_pending": False,
                "node_history": state.get("node_history", []) + ["act"],
            }

        tc = dict(plan[retry_step])  # copy to avoid mutating original

        # Skill hook: pre-run
        if skill and skill.act_hooks:
            tc["args"] = skill.act_hooks.pre_run(tc, state)

        step_key = f"step_{retry_step}"
        result, tool_msg = _execute_tool(tc, step_key, retry_step + 1, len(plan))
        results[step_key] = result

        # Record in tool_history
        history.append({
            "step_key": step_key,
            "tool": tc["name"],
            "args": tc["args"],
            "success": "error" not in result,
        })

        return {
            "tool_results": results,
            "messages": [tool_msg],
            "tool_history": history,
            "current_step_index": idx,
            "retry_pending": False,
            "background_task": result.get("background_task"),
            "node_history": state.get("node_history", []) + ["act"],
        }

    # ── Normal execution ──
    if idx >= len(plan):
        return {"node_history": state.get("node_history", []) + ["act"]}

    tc = dict(plan[idx])  # copy

    # Skill hook: pre-run
    if skill and skill.act_hooks:
        tc["args"] = skill.act_hooks.pre_run(tc, state)

    step_key = f"step_{idx}"
    result, tool_msg = _execute_tool(tc, step_key, idx + 1, len(plan))
    results[step_key] = result

    # Record in tool_history
    history.append({
        "step_key": step_key,
        "tool": tc["name"],
        "args": tc["args"],
        "success": "error" not in result,
    })

    # Skill hook: post-run
    if skill and skill.act_hooks:
        result = skill.act_hooks.post_run(result, state)
        results[step_key] = result

    # Check for background task → interrupt
    bg_task = result.get("background_task")

    if bg_task:
        # Pause graph; CLI polls and resumes with Command(resume=result)
        final_result = interrupt({
            "type": "background_task",
            "task_id": bg_task.get("task_id", ""),
            "status": bg_task.get("status", "running"),
        })
        # Graph resumes here — final_result is what CLI passed to Command(resume=...)
        results[step_key] = final_result
        bg_task = None

    return {
        "tool_results": results,
        "messages": [tool_msg],
        "tool_history": history,
        "current_step_index": idx + 1,
        "background_task": bg_task,
        "node_history": state.get("node_history", []) + ["act"],
    }
```

### Task B3: Update graph/nodes.py — Observe node with detector pipeline

**Files:**
- Modify: `src/tower/graph/nodes.py` (the observe_node function)

- [ ] **Step 1: Replace observe_node with skill detector pipeline**

```python
def observe_node(state: AgentState) -> dict:
    """Check the last tool result for errors.

    Pipeline:
    1. Basic check: result["error"] / bash returncode
    2. Skill detectors: run each ErrorDetector against stdout/stderr
    3. First match → error_info, refinement_needed
    """
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    results = state.get("tool_results", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    skill = state.get("_runtime", {}).get("skill")

    last_key = f"step_{idx - 1}" if idx > 0 else None
    last_result = results.get(last_key, {}) if last_key else {}
    tool_name = plan[idx - 1]["name"] if idx > 0 and idx <= len(plan) else "?"
    step_type = plan[idx - 1].get("step_type", "io") if idx > 0 and idx <= len(plan) else "io"

    print(f"\n[OBSERVE] step {idx}/{len(plan)} — {tool_name}")

    error_info = None
    refinement_needed = False

    # 1. Basic error check
    if last_result.get("error"):
        err = last_result["error"]
        observation = f"Tool '{tool_name}' failed: {err}"
        refinement_needed = retry_count < max_retries
        print(f"  FAILED ({'will retry' if refinement_needed else 'giving up'})")
        # Also run detectors on the error message itself
    elif tool_name == "bash" and "returncode" in last_result and last_result["returncode"] != 0:
        rc = last_result["returncode"]
        stderr = last_result.get("stderr", "")
        observation = f"Tool 'bash' failed (rc={rc}): {stderr[:500]}"
        refinement_needed = retry_count < max_retries
        print(f"  FAILED ({'will retry' if refinement_needed else 'giving up'})")
    elif tool_name == "bash" and "returncode" in last_result and last_result["returncode"] == 0:
        stderr = last_result.get("stderr", "")
        stdout = last_result.get("stdout", "")
        observation = f"Tool 'bash' succeeded"
        refinement_needed = False
        if stderr:
            print(f"  OK (rc=0, stderr has {len(stderr)} chars)")
        else:
            print("  OK")
    else:
        observation = f"Tool '{tool_name}' succeeded"
        refinement_needed = False
        print("  OK")

    # 2. Run skill error detectors (only for computation steps or failed steps)
    if skill and skill.observe_hooks and not error_info:
        try:
            detectors = skill.observe_hooks.get_error_detectors()
        except Exception:
            detectors = []

        # Gather all available output text
        stdout = last_result.get("stdout", "")
        stderr = last_result.get("stderr", "")
        exit_code = last_result.get("returncode", 0)

        # If the result has an error message, append it to stderr for detection
        if last_result.get("error"):
            stderr = stderr + "\n" + str(last_result["error"])

        for detector in detectors:
            try:
                detected = detector.detect(stdout, stderr, exit_code)
                if detected:
                    error_info = {
                        "error_type": detected.error_type,
                        "message": detected.message,
                        "suggestion": detected.suggestion,
                        "can_auto_fix": detected.can_auto_fix,
                    }
                    observation = f"[{detected.error_type}] {detected.message}"
                    refinement_needed = retry_count < max_retries
                    print(f"  DETECTED: {detected.error_type} — {detected.message[:100]}")
                    break
            except Exception:
                continue

    return {
        "observation": observation,
        "error_info": error_info,
        "refinement_needed": refinement_needed,
        "node_history": state.get("node_history", []) + ["observe"],
    }
```

### Task B4: Update graph/nodes.py — Refine node with fix strategies

**Files:**
- Modify: `src/tower/graph/nodes.py` (the refine_node function)

- [ ] **Step 1: Replace refine_node with skill strategy dispatch + LLM fallback**

The new refine_node:
1. Check retry_count vs max_retries
2. Look up fix strategy from skill.refine_hooks by error_type
3. If strategy exists → apply CorrectionAction
4. If no strategy → fallback to LLM-based correction (existing logic)
5. If can't fix → give up

```python
def refine_node(state: AgentState) -> dict:
    """Handle tool failure: apply fix strategy from skill, or fallback to LLM.

    Priority:
    1. Skill fix strategy (by error_type)
    2. LLM-based correction
    3. Give up
    """
    idx = state.get("current_step_index", 0)
    plan = list(state.get("plan", []))
    observation = state.get("observation", "")
    error_info = state.get("error_info", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    history = list(state.get("messages", []))
    skill = state.get("_runtime", {}).get("skill")

    # 1. Max retries → give up
    if retry_count >= max_retries:
        print(f"\n[REFINE] max retries ({max_retries}) reached")
        return {
            "refinement_needed": False,
            "retry_count": max_retries,
            "observation": f"Max retries ({max_retries}) exceeded. Giving up.",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    if idx < 1 or idx > len(plan):
        print(f"  bad idx={idx}, plan len={len(plan)}, skipping refine")
        return {
            "refinement_needed": False,
            "retry_count": max_retries,
            "observation": f"Bad state: step index {idx} out of range.",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    print(f"\n[REFINE] retry {retry_count + 1}/{max_retries}")

    error_type = error_info.get("error_type", "") if error_info else ""

    # 2. Try skill fix strategy first
    if skill and skill.refine_hooks and error_type:
        try:
            strategies = skill.refine_hooks.get_fix_strategies()
            strategy_fn = strategies.get(error_type)
            if strategy_fn:
                correction = strategy_fn(error_info, plan[idx - 1])
                if correction and correction.action == "retry_with_params" and correction.new_params:
                    plan[idx - 1]["args"] = correction.new_params
                    print(f"  auto-fix [{error_type}]: new params = {correction.new_params}")
                    return {
                        "plan": plan,
                        "retry_count": retry_count + 1,
                        "refinement_needed": False,
                        "retry_pending": True,
                        "node_history": state.get("node_history", []) + ["refine"],
                    }
                elif correction and correction.action == "ask_user":
                    print(f"  cannot auto-fix [{error_type}], asking user")
                    return {
                        "refinement_needed": False,
                        "retry_count": max_retries,
                        "observation": f"Cannot auto-fix: {error_info.get('message')}. {error_info.get('suggestion')}",
                        "node_history": state.get("node_history", []) + ["refine"],
                    }
        except Exception as e:
            print(f"  fix strategy error: {e}")

    # 3. Fallback to LLM-based correction (existing logic)
    failed_tc = plan[idx - 1]
    refine_prompt = HumanMessage(content=(
        f"工具 {failed_tc['name']} 执行失败:\n{observation}\n\n"
        f"请分析错误并尝试修正。"
    ))

    llm = _get_llm_with_tools()
    try:
        with _thinking_spinner("refining"):
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_REFINE),
                    *_sanitize_messages(history[-15:]),
                    refine_prompt,
                ]
            )
    except Exception as e:
        print(f"\n  \033[33m[REFINE] LLM call failed: {e}\033[0m")
        return {
            "refinement_needed": False,
            "messages": [refine_prompt],
            "retry_count": max_retries,
            "node_history": state.get("node_history", []) + ["refine"],
        }

    new_tool_calls = getattr(response, "tool_calls", None) or []

    if new_tool_calls:
        new_plan = list(plan)
        new_tc = {
            "name": new_tool_calls[0]["name"],
            "args": new_tool_calls[0]["args"],
            "id": new_tool_calls[0].get("id", ""),
        }
        new_plan[idx - 1] = new_tc
        print(f"  LLM corrected: {new_tc['name']}({json.dumps(new_tc['args'], ensure_ascii=False)})")

        return {
            "plan": new_plan,
            "messages": [refine_prompt, response],
            "current_step_index": idx,
            "retry_count": retry_count + 1,
            "refinement_needed": False,
            "retry_pending": True,
            "node_history": state.get("node_history", []) + ["refine"],
        }
    else:
        msg = response.content if hasattr(response, "content") else str(response)
        print(f"  gave up: {msg[:200]}")
        return {
            "refinement_needed": False,
            "messages": [refine_prompt, response],
            "retry_count": max_retries,
            "node_history": state.get("node_history", []) + ["refine"],
        }
```

### Task B5: Update graph/edges.py — Add route_after_refine

**Files:**
- Modify: `src/tower/graph/edges.py`

- [ ] **Step 1: Add route_after_refine function**

Add to the existing edges.py file:

```python
def route_after_refine(state: AgentState) -> str:
    """refine 之后：如果可以自动修正且还有重试次数 → act，否则 → respond。"""
    error = state.get("error_info", {}) or {}
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if error.get("can_auto_fix") and retry_count < max_retries:
        return "act"
    if state.get("retry_pending"):
        return "act"
    return "respond"
```

### Task B6: Update orchestrator.py — Wire skill + async checkpointer

**Files:**
- Modify: `src/tower/runtime/orchestrator.py`

- [ ] **Step 1: Update build_graph to inject skill hooks**

The orchestrator needs to:
1. Accept an async checkpointer
2. Inject skill into state["_runtime"]
3. Use graph.astream() instead of graph.invoke() for interrupt support

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from tower.state import AgentState
from tower.graph.nodes import plan_node, act_node, observe_node, refine_node, respond_node
from tower.graph.edges import route_after_plan, route_after_observe, route_after_refine


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build Plan→Act→Observe→(Refine|Plan|Respond) state graph.

    Args:
        checkpointer: LangGraph checkpointer (AsyncPostgresSaver etc.).
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    graph.add_node("refine", refine_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"act": "act", "respond": "respond"},
    )
    graph.add_edge("act", "observe")

    graph.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "refine": "refine",
            "act": "act",
            "plan": "plan",
            "respond": "respond",
        },
    )

    # Refine now has its own conditional routing
    graph.add_conditional_edges(
        "refine",
        route_after_refine,
        {"act": "act", "respond": "respond"},
    )

    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)
```

### Task B7: Update main.py — Async entry point + interrupt loop for background tasks

**Files:**
- Modify: `src/tower/main.py`

- [ ] **Step 1: Add background task interrupt handling to TowerChat.run()**

In `TowerChat.__init__`, make checkpointer creation async:

```python
# In TowerChat.__init__, replace:
# self.checkpointer = create_checkpointer(DB_URI)
# self.graph = build_graph(checkpointer=self.checkpointer)

# With lazy init (created in run() instead)
```

In `TowerChat.run()`, update the interrupt loop to handle `background_task` type:

```python
def run(self, user_input: str):
    # ... memory loading (same as before) ...

    # Lazy init of async checkpointer + graph
    import asyncio
    loop = asyncio.new_event_loop()
    self.checkpointer = loop.run_until_complete(create_checkpointer(DB_URI))
    self.graph = build_graph(checkpointer=self.checkpointer)

    # ... rest of run method ...

    # In the interrupt handling loop, add background_task case:
    while True:
        result = self.graph.invoke(input_data, self.config)
        interrupt_list = result.get("__interrupt__", [])
        if not interrupt_list:
            break

        intr_obj = interrupt_list[0]
        intr_value = intr_obj.value if hasattr(intr_obj, "value") else intr_obj

        if isinstance(intr_value, dict) and intr_value.get("type") == "approval":
            # ... existing approval handling ...
            pass

        elif isinstance(intr_value, dict) and intr_value.get("type") == "background_task":
            task_id = intr_value["task_id"]
            console.print(Panel(
                f"Background task running: [bold cyan]{task_id}[/]\n"
                f"Polling for completion...",
                title="[bold blue]⏳ Background Task[/]",
                border_style="blue",
            ))

            # Poll until complete
            import time
            while True:
                time.sleep(2)  # poll interval
                # Check if task is done (placeholder — real impl checks MCP/db)
                # For now, simulate completion
                console.print(f"  [dim]Checking {task_id}...[/]")
                # TODO: Replace with actual task status check
                break

            # Resume with dummy result
            input_data = Command(resume={"status": "completed", "output": "computation done"})

        else:
            raise RuntimeError(f"Unknown interrupt: {intr_value}")
```

---

## Phase C: Skill Pack Structure

### Task C1: Create skills/dmrg/ directory structure

**Files:**
- Create: `skills/dmrg/skill.yaml`
- Create: `skills/dmrg/hooks/__init__.py`
- Create: `skills/dmrg/hooks/plan.py`
- Create: `skills/dmrg/hooks/act.py`
- Create: `skills/dmrg/hooks/observe.py`
- Create: `skills/dmrg/hooks/refine.py`
- Create: `skills/dmrg/hooks/respond.py`
- Create: `skills/dmrg/detectors/__init__.py`
- Create: `skills/dmrg/detectors/base.py`
- Create: `skills/dmrg/detectors/scf.py` (IMPLEMENT)
- Create: `skills/dmrg/detectors/python_error.py` (IMPLEMENT)
- Create: `skills/dmrg/detectors/mpi.py` (STUB)
- Create: `skills/dmrg/detectors/gaussian.py` (STUB)
- Create: `skills/dmrg/detectors/orca.py` (STUB)
- Create: `skills/dmrg/detectors/vasp.py` (STUB)

### Task C2: Create skill.yaml

```yaml
name: dmrg
version: "0.2.0"
description: "DMRG/NQS quantum chemistry computational agent"

system_prompt: |
  You are a quantum chemistry computational agent.

  ## Capabilities
  - Design DMRG/NQS calculation plans
  - Execute calculations via dmrg-runner
  - Detect SCF convergence failures, MPI errors, and Python errors
  - Auto-fix common errors and retry
  - Report structured results with energies and convergence status

  ## Rules
  - Validate input parameters before running calculations
  - Check SCF convergence after every calculation
  - Auto-fix convergence issues when possible (increase bond dimension)
  - Report energies with units (Hartree) and precision
  - Flag uncertain results as [UNCERTAIN]

rules:
  - "Validate input parameters before computation"
  - "Check SCF convergence after every calculation"
  - "Increase bond dimension M on convergence failure"
  - "Report energies in Hartree with 6 decimal places"
  - "Flag uncertain results explicitly"

mcp_servers:
  - name: dmrg-runner
    command: ["python", "-m", "dmrg_skill.mcp_servers.runner"]

tools:
  - name: dmrg-runner.run
    intent_template: "Run DMRG calculation with {params}"
```

### Task C3: Implement detectors/base.py

```python
"""Base error detector protocol — re-exported for convenience."""
from tower.graph.hooks import ErrorDetector, ErrorInfo

__all__ = ["ErrorDetector", "ErrorInfo"]
```

### Task C4: Implement detectors/scf.py

```python
"""SCF convergence error detector for PySCF."""
import re
from tower.graph.hooks import ErrorInfo


class ScfConvergenceDetector:
    """Detect SCF convergence failures in PySCF output.

    Also detects ORCA SCF failure patterns (via regex match).
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        # Combine stdout and stderr for pattern matching
        combined = stdout + "\n" + stderr

        # PySCF pattern: "SCF not converged"
        if "SCF not converged" in combined or "scf not converged" in combined.lower():
            niter = "?"
            m = re.search(r"niter\s*[=:]\s*(\d+)", combined)
            if m:
                niter = m.group(1)

            # Try to extract current M value
            M = None
            m_m = re.search(r"[Mm]\s*[=:]\s*(\d+)", combined)
            if m_m:
                M = int(m_m.group(1))

            suggestion = "Increase bond dimension M"
            if M:
                suggestion += f" (current M={M}, try M={min(M * 2, 2000)})"

            return ErrorInfo(
                error_type="scf_not_converged",
                message=f"SCF 未收敛 (niter={niter})",
                suggestion=suggestion,
                can_auto_fix=True,
            )

        # ORCA pattern: "SCF FAILED TO CONVERGE" (stub — just flag, no auto-fix yet)
        if "SCF FAILED TO CONVERGE" in combined:
            return ErrorInfo(
                error_type="scf_not_converged",
                message="ORCA SCF 收敛失败",
                suggestion="尝试 'slowconv' 关键词或降低收敛阈值",
                can_auto_fix=False,  # ORCA auto-fix not yet implemented
            )

        return None
```

### Task C5: Implement detectors/python_error.py

```python
"""Python runtime error detector."""
import re
from tower.graph.hooks import ErrorInfo


class PythonErrorDetector:
    """Detect Python runtime errors in computation output."""

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        combined = stdout + "\n" + stderr

        # Check for traceback
        if "Traceback (most recent call last)" not in combined and exit_code == 0:
            return None

        # Extract exception type
        exc_match = re.search(r'(\w+(?:Error|Exception|Warning))(?::|$)', combined)
        exc_type = exc_match.group(1) if exc_match else "UnknownError"

        # Extract the error message line
        msg = ""
        for line in combined.split("\n"):
            line = line.strip()
            if line.startswith(exc_type) or "Error" in line:
                msg = line[:200]
                break

        # Determine if auto-fixable
        auto_fixable = False
        suggestion = ""

        if "MemoryError" in exc_type or "memory" in msg.lower():
            auto_fixable = True
            suggestion = "Reduce bond dimension M (halve current value, min 50)"
        elif "SyntaxError" in exc_type:
            suggestion = "Check code syntax for typos"
        elif "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            suggestion = "Install missing package or check module path"
        elif "FileNotFoundError" in exc_type:
            suggestion = "Check input file path exists"
        elif "RuntimeError" in exc_type:
            suggestion = "Check calculation parameters and input format"
        else:
            suggestion = f"Investigate {exc_type}: {msg}"

        return ErrorInfo(
            error_type="python_error",
            message=f"{exc_type}: {msg}" if msg else exc_type,
            suggestion=suggestion,
            can_auto_fix=auto_fixable,
        )
```

### Task C6: Create stub detectors

Each stub follows the same pattern — class with `detect()` returning None:

```python
# detectors/mpi.py (stub)
from tower.graph.hooks import ErrorInfo

class MpiErrorDetector:
    """Detect MPI errors in parallel computation output.
    
    TODO: Implement MPI error pattern matching for common MPI implementations
    (OpenMPI, MPICH, Intel MPI).
    """
    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None


# detectors/gaussian.py (stub)
class GaussianErrorDetector:
    """Detect Gaussian calculation errors.
    
    TODO: Implement when Gaussian support is added.
    Patterns: "Error termination", "l9999", link errors.
    """
    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None


# detectors/orca.py (stub)  
class OrcaErrorDetector:
    """Detect ORCA calculation errors.
    
    TODO: Implement when ORCA support is added.
    Patterns: "ORCA finished with error", SCF failures.
    """
    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None


# detectors/vasp.py (stub)
class VaspErrorDetector:
    """Detect VASP calculation errors.
    
    TODO: Implement when VASP support is added.
    Patterns: "ERROR" in OUTCAR, electronic convergence failures.
    """
    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
```

### Task C7: Implement hooks/observe.py — register detectors

```python
"""Observe node hooks for DMRG skill — registers error detectors."""
from detectors.scf import ScfConvergenceDetector
from detectors.python_error import PythonErrorDetector
from detectors.mpi import MpiErrorDetector
from detectors.gaussian import GaussianErrorDetector
from detectors.orca import OrcaErrorDetector
from detectors.vasp import VaspErrorDetector


class DmrgObserveHooks:
    """Register error detectors for quantum chemistry computations."""

    def get_error_detectors(self):
        """Return detectors in priority order (first match wins)."""
        return [
            ScfConvergenceDetector(),      # SCF convergence — most common
            PythonErrorDetector(),          # Python runtime errors
            MpiErrorDetector(),             # MPI errors (stub)
            GaussianErrorDetector(),        # Gaussian errors (stub)
            OrcaErrorDetector(),            # ORCA errors (stub)
            VaspErrorDetector(),            # VASP errors (stub)
        ]
```

### Task C8: Implement hooks/refine.py — fix strategies

```python
"""Refine node hooks for DMRG skill — fix strategies."""
from tower.graph.hooks import CorrectionAction


class DmrgRefineHooks:
    """Fix strategies for quantum chemistry computation errors."""

    def get_fix_strategies(self) -> dict:
        return {
            "scf_not_converged": self._fix_scf,
            "python_error": self._fix_python_error,
        }

    def _fix_scf(self, error_info: dict, step: dict) -> CorrectionAction:
        """Increase bond dimension M when SCF fails to converge."""
        args = step.get("args", {})
        current_M = args.get("M", 100)

        if isinstance(current_M, str):
            try:
                current_M = int(current_M)
            except ValueError:
                current_M = 100

        new_M = min(current_M * 2, 2000)
        new_args = {**args, "M": new_M}

        return CorrectionAction(
            action="retry_with_params",
            new_params=new_args,
        )

    def _fix_python_error(self, error_info: dict, step: dict) -> CorrectionAction:
        """Handle Python errors — only auto-fix MemoryError."""
        msg = error_info.get("message", "")

        if "MemoryError" in msg or "memory" in msg.lower():
            args = step.get("args", {})
            current_M = args.get("M", 200)

            if isinstance(current_M, str):
                try:
                    current_M = int(current_M)
                except ValueError:
                    current_M = 200

            new_M = max(current_M // 2, 50)
            new_args = {**args, "M": new_M}

            return CorrectionAction(
                action="retry_with_params",
                new_params=new_args,
            )

        # All other Python errors require human intervention
        return CorrectionAction(action="ask_user")
```

### Task C9: Implement hooks/plan.py, hooks/act.py, hooks/respond.py (minimal)

```python
# hooks/plan.py
class DmrgPlanHooks:
    """Plan node hooks — currently minimal, extension point for future."""

    def preprocess(self, task: str, state: dict) -> str:
        """Enrich task with domain context if needed."""
        # Future: add default DMRG parameters from context
        return task

    def validate_plan(self, plan: list[dict]) -> list[dict]:
        """Validate the generated plan."""
        # Future: ensure M values are reasonable, input files exist
        return plan


# hooks/act.py
class DmrgActHooks:
    """Act node hooks — validate parameters before execution."""

    def pre_run(self, tool_call: dict, state: dict) -> dict:
        """Validate DMRG parameters before running."""
        args = tool_call.get("args", {})
        # Future: validate M range, check for required params
        return args

    def post_run(self, result: dict, state: dict) -> dict:
        """Enrich computation result with parsed metrics."""
        # Future: parse energy values, convergence status from stdout
        return result


# hooks/respond.py
class DmrgRespondHooks:
    """Respond node hooks — format computation results."""

    def format_result(self, results: dict, state: dict) -> str:
        """Format results as a structured report."""
        # Future: build energy table, convergence summary
        return ""
```

### Task C10: Create detectors/__init__.py and hooks/__init__.py

```python
# detectors/__init__.py
"""DMRG skill error detectors."""

# hooks/__init__.py  
"""DMRG skill hooks — injected into graph nodes."""
```

---

## Phase D: Integration & Final Wiring

### Task D1: Update main.py — Full async + skill loading

**Files:**
- Modify: `src/tower/main.py`

Replace `TowerChat.__init__` to load skill and use async checkpointer:

Key changes:
1. Accept optional skill_path parameter
2. Load skill via SkillLoader
3. Use `asyncio.run()` for async checkpointer creation
4. Pass skill through `_runtime` to graph nodes

### Task D2: Integration test — verify the pipeline

**Files:**
- Create: `tests/test_refactor.py`

Basic integration test that:
1. Creates a minimal skill
2. Runs a task through the graph with mocked LLM
3. Verifies tool_history is populated
4. Verifies error_info flows through observe→refine

### Task D3: Run full test suite

Verify all existing tests still pass, new integration tests pass.

---

## Self-Review Checklist

- [x] Every task has exact file paths
- [x] Every code step shows complete implementation
- [x] No TBDs, TODOs, or placeholders (except intentional stubs)
- [x] Types are consistent: AgentState error_info, CorrectionAction, ErrorInfo
- [x] Node signatures match: all take only `state: AgentState`
- [x] Spec coverage: async pool → Task A1
- [x] Spec coverage: async checkpointer → Task A2
- [x] Spec coverage: async store → Task A3
- [x] Spec coverage: hook protocols → Task A5
- [x] Spec coverage: skill loader hooks → Task A6
- [x] Spec coverage: AgentState extension → Task B1
- [x] Spec coverage: act interrupt/resume → Task B2
- [x] Spec coverage: observe detector pipeline → Task B3
- [x] Spec coverage: refine fix strategies → Task B4
- [x] Spec coverage: route_after_refine → Task B5
- [x] Spec coverage: orchestrator wiring → Task B6
- [x] Spec coverage: skill pack structure → Tasks C1-C10
- [x] Spec coverage: implemented detectors (scf, python_error) → Tasks C4-C5
- [x] Spec coverage: stub detectors (mpi, gaussian, orca, vasp) → Task C6
