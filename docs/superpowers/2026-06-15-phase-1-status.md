# Phase 1 Implementation Status

**Date:** 2026-06-15 (updated 2026-06-16)
**Related:** [[2026-06-15-tower-agent-design]] (spec), [[2026-06-15-tower-phase-1-plan]] (plan)

## Overview

Phase 1 implementation is **in progress**. Core runtime is functional (Plan→Act→Observe→Refine→Respond loop works end-to-end), but substantial deviations from the original plan exist. Several planned modules were descoped or replaced with simpler alternatives. **58 tests pass** across filesystem tools, safety layer, and web tools.

## Implemented vs. Planned

### What WAS Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| `AgentState` schema | ✅ Simplified | `src/tower/state.py` — 14 fields vs. 25 in the plan. No `ToolInvocation` TypedDict, no `_runtime` injection, no tool_history. |
| Graph nodes (plan/act/observe/refine/respond) | ✅ Done | `src/tower/graph/nodes.py` — 5 nodes, functional. plan_node uses DeepSeek tool-calling instead of text-based plan decomposition. |
| Edge routing | ✅ Done | `src/tower/graph/edges.py` — `route_after_plan`, `route_after_observe`. Multi-pass loop: observe → plan → act when more tools needed. |
| Orchestrator | ✅ Done | `src/tower/runtime/orchestrator.py` — `build_graph()` with Postgres checkpointer support. |
| Built-in tools: filesystem | ✅ Expanded | `src/tower/tools/builtin/filesystem.py` — 9 tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `glob_tool`, `grep`, `move_file`, `copy_file`, `delete_file`. All use LangChain `@tool` decorator + safety checks. |
| Built-in tools: bash | ✅ Extra | `src/tower/tools/builtin/bash.py` — subprocess-based bash execution (not in the plan, replaces python_runner). |
| Built-in tools: web | ✅ Extra | `src/tower/tools/builtin/web.py` — `web_fetch` (HTTP fetch with HTML→text conversion) and `web_search` (Tavily API). Not in the original plan. |
| Safety layer | ✅ Extra | `src/tower/tools/safety.py` — System directory blacklist, root privilege check, dangerous bash pattern detection, path resolution with symlink traversal protection, git repo boundary check. |
| Tool registry | ✅ Extra | `src/tower/tools/registry.py` — centralized `TOOLS` list and `TOOL_BY_NAME` dict. 12 tools total. |
| Short-term memory | ✅ Postgres | `src/tower/memory/short_term.py` — `PostgresSaver` via LangGraph, replacing planned SQLite session store. |
| Long-term memory | ✅ Postgres | `src/tower/memory/long_term.py` — `LongTermMemory` class wrapping LangGraph `PostgresStore`. Auto-extracts user facts via LLM, handles contradiction removal. |
| Connection pool | ✅ Created | `src/tower/memory/pool.py` — shared `psycopg_pool` ConnectionPool with atexit cleanup. |
| CLI (chat + run + sessions) | ✅ Done | `src/tower/main.py` — interactive chat with prompt_toolkit (history, auto-suggest), `tower run` for single tasks, `tower sessions` for listing. |
| Tests | ✅ Partial | 3 test files, **58 tests passing**: `test_filesystem_tools.py` (32 tests covering all 9 file tools), `test_safety.py` (15 tests for safety checks), `test_web_tools.py` (6 tests for web fetch/search). Run via `uv run pytest tests/`. |

### What was NOT Implemented (deviations from plan)

| Component | Plan Reference | Reason |
|-----------|---------------|--------|
| Tracing module (`TraceEvent` + `TraceLogger`) | Task 3 | Descoped. No `src/tower/tracing/` directory exists. |
| Session memory (`SessionStore` with SQLite) | Task 4 | Replaced by Postgres-based `PostgresSaver` for short-term memory. |
| `python_runner` tool | Task 6 | Replaced by `bash` tool. Multi-process code execution was deemed overkill for Phase 1. |
| MCP Client Manager | Task 7 | Descoped. Tool registration uses a simple `TOOLS` list + `TOOL_BY_NAME` dict instead of MCP protocol. |
| Skill Loader | Task 8 | Descoped. No `skill.yaml` parsing. System prompt is hardcoded in `nodes.py`. |
| `skills/default/skill.yaml` | Task 9 Step 1 | Descoped. Default behavior is baked into the system prompts in `nodes.py`. |
| CLI `trace` and `skill` commands | Task 13 | Not implemented. Only `chat`, `run`, `sessions` commands exist. |
| Integration tests | Task 14 | Not written. Existing tests only cover tools and safety, not graph execution or orchestrator. |
| Graph node / orchestrator unit tests | Tasks 9-12 | Not written. Existing tests only cover the tool layer, not the graph execution logic. |
| `ToolInvocation` TypedDict | Task 2 | Simplified away. Observation is a plain string, no per-step invocation tracking. |

### Key Architectural Differences

1. **LLM Provider: DeepSeek instead of Anthropic**
   - Plan specified `langchain-anthropic` + Claude. Implementation uses `langchain-deepseek` + DeepSeek V4 Flash.
   - This required a `_sanitize_messages()` workaround for DeepSeek API's strict tool-call ordering requirement.
   - The workaround is embedded in `nodes.py` rather than in a provider adapter layer (see findings.json).

2. **Tool Design: LangChain @tool instead of plain functions**
   - Plan specified tools as plain functions returning `dict`. Implementation uses LangChain's `@tool` decorator, making tools directly bindable to LLMs via `.bind_tools()`.

3. **Plan Generation: Tool-calling instead of text decomposition**
   - Plan specified LLM generates a numbered text plan, then act_node parses tool names from text heuristically. Implementation uses DeepSeek's native tool-calling: the LLM emits structured `tool_calls`, which populate the plan directly.

4. **Memory: Postgres instead of SQLite**
   - Plan specified SQLite for session memory. Implementation uses PostgreSQL for both short-term (checkpoints via PostgresSaver) and long-term (facts via PostgresStore).

5. **No `_runtime` injection**
   - Plan specified injecting `llm`, `tool_executor`, `trace_logger` into `state["_runtime"]`. Implementation uses module-level globals (`_llm` singleton in nodes.py) and direct imports instead.

6. **No MCP protocol**
   - Phase 1 plan included MCP client manager for tool discovery. Implementation skips MCP entirely — tools are hardcoded in the registry.

## Known Issues (from findings.json)

See `findings.json` in the repo root for 7 documented bugs:

1. **observe_node misclassifies file tool success as failure** — relies on `returncode` key which only bash tool produces
2. **Tool result truncation** — content silently truncated to 2000 chars in `act_node`
3. **Wasteful memory extraction** — LLM called on every interaction including trivial greetings
4. **Duplicate LLM instances** — `TowerChat` creates a second LLM for memory extraction while `nodes.py` has a cached singleton
5. **Provider workaround in wrong layer** — `_sanitize_messages` should be in a provider adapter
6. **No connection cleanup** — `LongTermMemory` opens SQLite connections that are never closed (note: this finding may be stale since implementation now uses Postgres)
7. **Bidirectional index motion** — `refine_node` decrements `current_step_index`, creating confusing state traces
8. **Single-value observation** — `observation` field only captures the most recent tool's status

## What's Left for Phase 1 Completion

### High Priority (core functionality gaps)
- [ ] **Graph-level tests** — No tests for graph nodes, edges, or orchestrator. Plan specified 30+ tests; currently 58 pass but only at the tool/safety layer.
- [ ] **Integration tests** — verify end-to-end graph execution with mocked LLM.
- [ ] **Fix findings.json bugs** — especially #1 (observe_node file tool misclassification) which is a correctness bug.
- [ ] **Provider adapter layer** — extract `_sanitize_messages` from nodes.py into a provider-specific adapter.

### Medium Priority (planned but descoped)
- [ ] **`python_runner` tool** — if needed; currently replaced by `bash` tool.
- [ ] **Tracing module** — `TraceEvent` + `TraceLogger` for structured observability.
- [ ] **CLI `skill` and `trace` commands** — `tower skill list` and `tower trace list/show`.

### Low Priority (Phase 2+)
- [ ] **MCP Client Manager** — real stdio-based MCP server process management.
- [ ] **Skill Loader** — parse `skill.yaml` for dynamic system prompt and tool config.
- [ ] **`skills/default/skill.yaml`** — default skill definition file.

## File Inventory

```
src/tower/
├── __init__.py                    # Package init, version
├── main.py                        # CLI: chat, run, sessions
├── state.py                       # AgentState TypedDict (14 fields)
├── graph/
│   ├── nodes.py                   # 5 nodes: plan, act, observe, refine, respond
│   └── edges.py                   # 2 conditional edges (route_after_plan, route_after_observe)
├── runtime/
│   └── orchestrator.py            # build_graph() + Postgres checkpointer
├── tools/
│   ├── registry.py                # TOOLS list, TOOL_BY_NAME dict (12 tools)
│   ├── safety.py                  # Path validation, bash safety, system dir blacklist
│   └── builtin/
│       ├── bash.py                # bash tool (subprocess, 30s timeout)
│       ├── filesystem.py          # 9 tools: read_file, write_file, edit_file,
│       │                           #   list_directory, glob_tool, grep,
│       │                           #   move_file, copy_file, delete_file
│       └── web.py                 # web_fetch, web_search (Tavily API)
├── memory/
│   ├── __init__.py                # Re-exports
│   ├── short_term.py              # PostgresSaver via connection pool
│   ├── long_term.py               # LongTermMemory (PostgresStore wrapper)
│   └── pool.py                    # Shared psycopg connection pool
├── cli/                           # ❌ NOT CREATED
├── tracing/                       # ❌ NOT CREATED
└── eval/                          # ❌ NOT CREATED

tests/
├── conftest.py                    # Shared fixtures (tmpdir)
├── test_filesystem_tools.py       # 32 tests — all 9 file tools
├── test_safety.py                 # 15 tests — path validation, bash safety
└── test_web_tools.py              # 6 tests — web_fetch, web_search

skills/                            # ❌ NOT CREATED (no skill.yaml)
```

## Verdict

The implementation is a **functional MVP** — the core Plan→Act→Observe→Refine→Respond loop works with 12 tools (bash, 8 filesystem tools, 2 web tools, safety layer), Postgres-backed memory, and an interactive CLI. However, it differs significantly from the plan: Anthropic→DeepSeek, SQLite→Postgres, text-plans→tool-calls, and tracing/skill-loader were entirely skipped. 58 tool/safety tests pass, but graph-level and integration tests are still missing. The 8 known bugs in findings.json should be addressed before considering Phase 1 complete.
