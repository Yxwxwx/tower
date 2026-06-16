# Tower

A general-purpose agent framework built on LangGraph + DeepSeek, with domain specialization via installable **Skill Packs**.

```
Plan → Act → Observe → Refine → Respond
```

## Quick Start

```bash
# Install dependencies
uv sync

# Interactive chat
uv run tower chat

# Single task execution
uv run tower run "list all Python files in src/"

# List past sessions
uv run tower sessions
```

## Architecture

Tower uses a LangGraph `StateGraph` with 5 nodes:

| Node | Role |
|------|------|
| `plan` | Decompose task into tool calls via LLM tool-calling |
| `act` | Execute tools (bash, filesystem, web) with approval gates |
| `observe` | Check success/failure; detect errors |
| `refine` | LLM-based error recovery (retry with corrections) |
| `respond` | Synthesize final answer from tool results |

### Memory

- **Short-term** — LangGraph `PostgresSaver` for checkpoint/state persistence across sessions
- **Long-term** — `PostgresStore` for cross-session user facts, auto-extracted via LLM

### Tools (12 built-in)

| Category | Tools |
|----------|-------|
| Shell | `bash` |
| File I/O | `read_file`, `write_file`, `edit_file` |
| Browse | `list_directory`, `glob_tool`, `grep` |
| File Mgmt | `move_file`, `copy_file`, `delete_file` |
| Web | `web_fetch`, `web_search` |

### Safety

All file and bash operations go through `tools/safety.py`:
- System directory blacklist (`/etc`, `/usr`, `/System`, etc.)
- Root privilege detection
- Dangerous bash pattern blocking (`sudo`, `rm -rf /`, fork bombs, etc.)
- Symlink traversal protection
- Git repo boundary awareness

## Development

```bash
# Run tests (58 passing)
uv run pytest tests/ -v

# Python 3.14+ required
uv sync
```

## Status

Phase 1 core runtime is **functional** — see [Phase 1 Status](docs/superpowers/2026-06-15-phase-1-status.md) for detailed actual-vs-planned comparison.

Key deviations from the original design spec:
- **DeepSeek** instead of Anthropic (requires `_sanitize_messages` workaround)
- **PostgreSQL** instead of SQLite for memory
- **Tool-calling** instead of text-based plan decomposition
- **No MCP protocol** yet — tools are hardcoded in registry
- **No tracing/skill-loader** yet — descoped for Phase 2

## License

MIT
