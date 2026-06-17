"""MemoryRetriever — finds relevant past experience for a new task.

Supervisor calls this before planning. Returns failure patterns,
successful pipelines, and user preferences.

MVP: type-filtered key-value lookup (no embedding vectors).
Post-MVP: pgvector semantic search.
"""

from typing import TYPE_CHECKING

from tower.memory.models import MemoryRecord, MemoryType

if TYPE_CHECKING:
    from tower.memory.long_term import LongTermMemory


class MemoryRetriever:
    """Retrieve relevant memories for a given task.

    Supervisor calls:
    - get_failure_patterns(agent) → known issues for that agent
    - get_successful_pipelines(task_hint) → pipelines for similar tasks
    - get_user_preferences() → user-specific settings
    """

    def __init__(self, ltm: "LongTermMemory"):
        self.ltm = ltm

    async def get_failure_patterns(
        self,
        agent: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Get known failure patterns, optionally filtered by agent."""
        records = await self.ltm.search_by_type(
            MemoryType.FAILURE_PATTERN,
            limit=limit,
        )
        if agent:
            records = [r for r in records if agent.lower() in r.content.lower()]
        return records

    async def get_successful_pipelines(self, limit: int = 5) -> list[MemoryRecord]:
        """Get pipelines that completed successfully."""
        return await self.ltm.search_by_type(
            MemoryType.SUCCESSFUL_PIPELINE,
            limit=limit,
        )

    async def get_user_preferences(self, limit: int = 20) -> list[MemoryRecord]:
        """Get user preference records."""
        return await self.ltm.search_by_type(
            MemoryType.USER_PREFERENCE,
            limit=limit,
        )

    async def get_all_relevant(
        self,
        task: str,
        agent: str | None = None,
        limit: int = 10,
    ) -> dict[str, list[MemoryRecord]]:
        """Get all relevant memories for a task.

        Returns a dict keyed by MemoryType, for supervisor prompt enrichment.
        Post-MVP: uses embedding similarity search.
        MVP: returns type-filtered records.
        """
        return {
            "failures": await self.get_failure_patterns(agent, limit),
            "pipelines": await self.get_successful_pipelines(limit),
            "preferences": await self.get_user_preferences(limit),
        }
