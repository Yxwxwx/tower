# Workflow Contract — Tower Multi-Agent System

**Version:** 1.0
**Status:** Frozen
**Date:** 2026-06-17

This document defines the system-level execution rules: run lifecycle, agent invocation protocol, handoff semantics, and event flow. Every agent and infrastructure component MUST conform to this contract.

---

## 1. Run Lifecycle

### 1.1 Run States

```
                    User submits task
                          │
                    ┌─────▼─────┐
                    │  CREATED  │  RunState created, supervisor begins
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  RUNNING  │  Agents being dispatched
                    └─────┬─────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
        ┌─────▼────┐ ┌────▼────┐ ┌───▼────────┐
        │COMPLETED │ │ FAILED  │ │NEEDS_HUMAN │
        └──────────┘ └─────────┘ └────────────┘
```

### 1.2 Run State Transitions

| From | To | Trigger |
|------|-----|---------|
| `CREATED` | `RUNNING` | Supervisor begins dispatching first agent |
| `RUNNING` | `COMPLETED` | All agent tasks DONE, final result synthesized |
| `RUNNING` | `FAILED` | A non-recoverable error occurred (ABANDONED without fallback) |
| `RUNNING` | `NEEDS_HUMAN` | Supervisor escalated: max retries exceeded or irreversible failure |

Only the supervisor may change `RunState.status`.

### 1.3 Run Identifiers

Every run has three identifiers:

| Field | Scope | Example | Set by |
|-------|-------|---------|--------|
| `run_id` | This user request | `"n2-nevpt2-20260617-001"` | CLI entry |
| `trace_id` | End-to-end trace | `"trace-a1b2c3d4"` | CLI entry |
| `task_id` | Individual agent task | `"n2-nevpt2-gaussian-001"` | Supervisor |

`trace_id` is propagated to every `AgentTask` and `AgentResult`. All log lines, MCP calls, and MonitorEvents include `trace_id`.

---

## 2. Agent Invocation Protocol

### 2.1 Handoff Sequence

```
Supervisor                     Agent Subgraph
    │                               │
    │  1. AgentTask (handoff)       │
    │──────────────────────────────>│
    │                               │  2. Execute nodes
    │                               │     (write_input → slurm → run)
    │                               │
    │  3. AgentResult (return)      │
    │<──────────────────────────────│
    │                               │
    │  4. Supervisor decides:       │
    │     - DONE → next agent       │
    │     - FAILED → retry/escalate │
```

### 2.2 Contract Guarantees

1. **Exactly-once dispatch**: Supervisor hands off a task exactly once. Retries create a new `task_id`.
2. **Structured I/O only**: All communication uses `AgentTask[T]` / `AgentResult[T]`. No side-channel messages.
3. **Artifact references, not paths**: Agents reference artifacts by `artifact_id`. The `ArtifactResolver` maps to paths.
4. **Timeout**: If an agent does not return within `TimeoutPolicy.agent_timeout_s`, the supervisor marks it `FAILED`.

### 2.3 Agent Registration

Each agent registers with the supervisor by providing:

```python
@dataclass
class AgentRegistration:
    name: AgentName                          # "gaussian" | "pyscf" | ...
    subgraph: CompiledStateGraph             # compiled LangGraph subgraph
    retry_policy: RetryPolicy                # idempotency + cleanup rules
    timeout_s: int = 3600                    # per-invocation timeout
    dependencies: set[AgentName] = set()     # upstream agents it depends on
```

---

## 3. Handoff Semantics

### 3.1 When Supervisor Hands Off

The supervisor hands off to an agent when:

1. All of the agent's `dependencies` are `DONE` (their artifacts are READY).
2. The agent's required `artifacts_in` are available in the `ArtifactRegistry`.
3. No higher-priority agent is in `NEEDS_HUMAN` (blocking).

### 3.2 When Control Returns to Supervisor

Control returns to supervisor after:

1. Agent returns `AgentResult` (any status).
2. Agent timeout expires (supervisor marks `FAILED`).
3. Monitor reports an `AGENT_HEARTBEAT_LOST` event (supervisor marks `FAILED`).

### 3.3 Parallel Execution

- Agents with **no mutual dependencies** MAY execute in parallel.
- **HPC agent** MAY be dispatched in parallel with the last chemical agent (once all Slurm templates are available).
- **Monitor agent** is dispatched independently (event-driven, not part of the linear chain).

---

## 4. Artifact Lifecycle

### 4.1 States

```
PRODUCING → READY → CONSUMED
                ↘ STALE
```

### 4.2 Rules

| Rule | Description |
|------|-------------|
| Immutable | Once READY, artifact content never changes. |
| New version = new ID | Retry produces a new `artifact_id`; old one marked `STALE`. |
| Resolution by ID | Downstream agents reference `artifact_id`, not file path. |
| Producer ownership | Only the producing agent may transition `PRODUCING → READY`. |
| Consumer marking | Downstream agent marks `READY → CONSUMED` after successful read. |

### 4.3 Registration

When an agent produces an artifact:

```python
await artifact_registry.register(ArtifactRecord(
    artifact_id=f"{task_id}-fchk",
    path="jobs/n2-nevpt2-gaussian-001/N2_opt.fchk",
    type="fchk",
    content_hash=sha256(file_content),
    producer_agent="gaussian",
    producer_task_id=task_id,
    status=ArtifactStatus.READY,
))
```

---

## 5. Event Flow (Monitor ↔ Supervisor)

### 5.1 Monitor Events

Monitor writes to `RunStateStore.event_log` (append-only). Events have monotonic `id` fields.

```
MonitorAgent                    RunStateStore                  SupervisorAgent
    │                               │                               │
    │  poll_jobs()                  │                               │
    │  ───→ queue-status MCP        │                               │
    │  parse_logs()                 │                               │
    │  classify_error()             │                               │
    │                               │                               │
    │  append_event(MonitorEvent)   │                               │
    │──────────────────────────────>│                               │
    │                               │                               │
    │                               │  read_new_events()            │
    │                               │<──────────────────────────────│
    │                               │                               │
    │                               │  decide: retry / escalate     │
```

### 5.2 Event Consumption

Supervisor uses a **watermark** (`last_processed_event_id`) to consume events exactly once:

```python
new_events = [e for e in event_log if e.id > last_processed_event_id]
for event in sorted(new_events, key=lambda e: e.id):
    handle_event(event)
    last_processed_event_id = event.id
```

### 5.3 Event Types

| Event Type | Meaning | Supervisor Action |
|-----------|---------|-------------------|
| `JOB_STARTED` | Job began running | Record, no action |
| `JOB_DONE` | Job completed successfully | Mark agent DONE, dispatch next |
| `JOB_FAILED` | Job exited with error | Read `error_category`, decide retry/escalate |
| `JOB_TIMEOUT` | Job exceeded walltime | Retry with more walltime or escalate |
| `JOB_OOM` | Job ran out of memory | Retry with more memory or reduce M |
| `AGENT_HEARTBEAT_LOST` | Agent stopped responding | Mark FAILED, retry or escalate |

---

## 6. Retry & Escalation

### 6.1 Retry Policy

| Agent | Idempotent? | Cleanup Before Retry? | Max Retries |
|-------|------------|----------------------|-------------|
| gaussian | Yes | No | 2 |
| pyscf | Yes | No | 2 |
| orca | Yes | No | 2 |
| hpc | No | Yes (`scancel` old job) | 2 |
| monitor | Yes | No | 2 |

### 6.2 Escalation Triggers

Supervisor escalates to `NEEDS_HUMAN` when:

1. Same agent fails with the **same error_category** on all retries.
2. `compute_invalidation_scope()` shows 3+ agents would need re-execution.
3. An irreversible side-effect operation fails (e.g., data deletion).

### 6.3 Partial Failure

When agent X fails:

1. Call `compute_invalidation_scope(X, DEP_GRAPH)` to find all affected downstream agents.
2. Mark all affected artifacts as `STALE`.
3. Cancel any running HPC jobs for affected agents.
4. Re-dispatch from X (or escalate).

---

## 7. Checkpoint & Recovery

### 7.1 Checkpoint Points

| Trigger | What is saved |
|---------|---------------|
| Before each handoff | Full `RunState` to `AsyncPostgresSaver` |
| After each agent returns | `AgentResult` + updated `artifact_registry` |
| After each MonitorEvent | Append to `event_log` |

### 7.2 Recovery

If the supervisor process crashes:

1. On restart, load `RunState` from checkpoint.
2. Check `agent_tasks` status map: any agent in `RUNNING` → re-evaluate (heartbeat or timeout).
3. Replay `event_log` from `last_processed_event_id`.
4. Continue from the last incomplete handoff.

---

## 8. Sequence Diagram: Full "N2 NEVPT2" Run

```
CLI          Supervisor    Gaussian    PySCF     Orca      HPC       Monitor   RunStateStore
 │               │            │          │         │         │          │          │
 │  run("N2")    │            │          │         │         │          │          │
 │──────────────>│            │          │         │         │          │          │
 │               │  CREATE    │          │         │         │          │          │
 │               │─────────────────────────────────────────────────────────────>│
 │               │            │          │         │         │          │          │
 │               │  handoff   │          │         │         │          │          │
 │               │───────────>│          │         │         │          │          │
 │               │            │  DONE    │         │         │          │          │
 │               │<───────────│          │         │         │          │          │
 │               │            │          │         │         │          │          │
 │               │  handoff              │         │         │          │          │
 │               │──────────────────────>│         │         │          │          │
 │               │            │          │  DONE   │         │          │          │
 │               │<──────────────────────│         │         │          │          │
 │               │            │          │         │         │          │          │
 │               │  handoff                        │         │          │          │
 │               │─────────────────────────────────>│         │          │          │
 │               │            │          │         │         │          │          │
 │               │  handoff (parallel)              │         │          │          │
 │               │───────────────────────────────────────────>│          │          │
 │               │            │          │         │  DONE   │          │          │
 │               │<───────────────────────────────────────────│          │          │
 │               │            │          │         │         │          │          │
 │               │  dispatch (event-driven)                             │          │
 │               │──────────────────────────────────────────────────────>│          │
 │               │            │          │         │         │  poll    │          │
 │               │            │          │         │         │<─────────>│          │
 │               │            │          │         │         │  events  │          │
 │               │  read events                                         │          │
 │               │<─────────────────────────────────────────────────────│          │
 │               │            │          │         │         │          │          │
 │               │  COMPLETED           │         │         │          │          │
 │  result       │            │          │         │         │          │          │
 │<──────────────│            │          │         │         │          │          │
```

---

## 9. Invariants (MUST NOT Violate)

1. **No agent directly modifies another agent's state.** Communication only via AgentTask/AgentResult + RunStateStore.
2. **No agent uses raw file paths.** All artifact access via ArtifactResolver.
3. **No event is processed twice.** Watermark enforced.
4. **No illegal state transition.** `validate_transition()` on every write.
5. **No silent agent death.** Heartbeat + MonitorAgent timeout detection.
6. **One writer per field.** Ownership table in Section 6.6 of design spec.
7. **Artifacts are immutable.** Retry = new artifact_id.
