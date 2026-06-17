"""MemoryCompiler — transforms execution traces into structured knowledge.

Runs AFTER a Run completes. Reads RunState + events → produces MemoryRecords.

Principles:
- Execution ≠ Memory: raw checkpoints are not knowledge.
- Memory must be compressed: patterns, not raw logs.
- Memory is event-derived: compiler reads events. Agents NEVER write memory directly.
"""
import hashlib
from typing import TYPE_CHECKING

from tower.memory.models import MemoryRecord, MemoryType

if TYPE_CHECKING:
    from tower.state.run_store import RunState
    from tower.memory.long_term import LongTermMemory


def _hash_id(content: str) -> str:
    """Deterministic memory_id from content."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


class MemoryCompiler:
    """Compress execution traces into structured MemoryRecords.

    MVP scope:
    - Extract task patterns (agent chains)
    - Extract failure patterns (from MonitorEvents)
    - Deduplicate by content hash
    - LLM-based summarization deferred to post-MVP
    """

    def __init__(self, ltm: "LongTermMemory"):
        self.ltm = ltm

    async def compile_run(self, run_state: "RunState") -> list[MemoryRecord]:
        """Compile one completed run into memory records."""
        records: list[MemoryRecord] = []

        # 1. Task pattern (the workflow itself as a reusable template)
        task_pattern = await self._extract_task_pattern(run_state)
        if task_pattern:
            records.append(task_pattern)

        # 2. Failure patterns (what went wrong and how it was fixed)
        failure_patterns = await self._extract_failure_patterns(run_state)
        records.extend(failure_patterns)

        # 3. Store all records (with dedup via memory_id)
        stored = []
        for record in records:
            memory_id = await self.ltm.add_record(record)
            if memory_id:  # None = duplicate skipped
                stored.append(record)

        return stored

    async def _extract_task_pattern(self, run_state) -> MemoryRecord | None:
        """Compress agent chain into a reusable pipeline template.

        Only records pipelines that completed successfully.
        """
        from contracts.agent_task import TaskStatus

        if run_state.status != TaskStatus.DONE:
            return None

        # Extract agent sequence from task IDs
        agent_sequence: list[str] = []
        for task_id in run_state.agent_tasks:
            # "n2-nevpt2-gaussian-001" → guess agent from task_id
            for agent in ["gaussian", "pyscf", "orca", "hpc", "monitor"]:
                if agent in task_id.lower():
                    agent_sequence.append(agent)
                    break

        content = f"Pipeline: {' → '.join(agent_sequence)} for task: {run_state.task}"

        return MemoryRecord(
            memory_id=_hash_id(content),
            trace_id=run_state.trace_id,
            content=content,
            memory_type=MemoryType.SUCCESSFUL_PIPELINE,
            confidence=1.0,
        )

    async def _extract_failure_patterns(self, run_state) -> list[MemoryRecord]:
        """Extract error→fix patterns from MonitorEvents.

        Only records failures that were later resolved (retry succeeded).
        """
        records = []
        for event in run_state.event_log:
            if event.event_type == "JOB_FAILED" and event.error_category:
                was_fixed = self._was_error_fixed(event, run_state)
                confidence = 0.9 if was_fixed else 0.5

                content = (
                    f"Failure: {event.error_category} in {event.agent}. "
                    f"Suggestion: {event.suggestion}. "
                    f"{'Fixed by retry.' if was_fixed else 'Not resolved.'}"
                )

                records.append(MemoryRecord(
                    memory_id=_hash_id(content),
                    trace_id=run_state.trace_id,
                    content=content,
                    memory_type=MemoryType.FAILURE_PATTERN,
                    confidence=confidence,
                ))
        return records

    @staticmethod
    def _was_error_fixed(event, run_state) -> bool:
        """Check if an error was later resolved by retry."""
        agent = event.agent
        from contracts.agent_task import TaskStatus
        for task_id, status in run_state.agent_tasks.items():
            if agent in task_id and status == TaskStatus.DONE:
                return True
        return False
