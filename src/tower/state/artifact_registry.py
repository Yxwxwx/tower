"""ArtifactRegistry — immutable artifact lifecycle tracking.

Artifacts are immutable after READY. Retry → new artifact_id, old → STALE.
Agents MUST resolve artifacts by artifact_id, never by raw path.
"""
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field

from contracts.agent_task import ArtifactStatus, AgentName

# ═══════════════════════════════════════════════════════════════════
# Transition table
# ═══════════════════════════════════════════════════════════════════

ARTIFACT_TRANSITIONS: dict[ArtifactStatus, set[ArtifactStatus]] = {
    ArtifactStatus.PRODUCING: {ArtifactStatus.READY, ArtifactStatus.STALE},
    ArtifactStatus.READY:     {ArtifactStatus.CONSUMED, ArtifactStatus.STALE},
    ArtifactStatus.CONSUMED:  {ArtifactStatus.STALE},
    ArtifactStatus.STALE:     set(),           # terminal
}


# ═══════════════════════════════════════════════════════════════════
# ArtifactRecord
# ═══════════════════════════════════════════════════════════════════


class ArtifactRecord(BaseModel):
    """One artifact in the registry."""
    artifact_id: str                          # globally unique
    path: str                                 # filesystem path
    type: str                                 # "fchk" | "json" | "log" | "slurm" | ...
    content_hash: str = ""                    # sha256, computed on registration
    producer_agent: AgentName | None = None
    producer_task_id: str = ""
    status: ArtifactStatus = ArtifactStatus.PRODUCING
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════
# ArtifactRegistry
# ═══════════════════════════════════════════════════════════════════


class ArtifactRegistry:
    """Central artifact registry — part of RunStateStore.

    Agents register produced artifacts here. Downstream agents resolve
    by artifact_id through ArtifactResolver (thin wrapper).
    """

    @staticmethod
    async def register(
        run_state,                      # RunState
        artifact_id: str,
        path: str,
        artifact_type: str,
        producer_agent: AgentName,
        producer_task_id: str,
        content: bytes | str,
    ) -> ArtifactRecord:
        """Register a new artifact. Computes content_hash automatically."""
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        content_hash = hashlib.sha256(content_bytes).hexdigest()

        record = ArtifactRecord(
            artifact_id=artifact_id,
            path=path,
            type=artifact_type,
            content_hash=content_hash,
            producer_agent=producer_agent,
            producer_task_id=producer_task_id,
            status=ArtifactStatus.READY,
            size_bytes=len(content_bytes),
        )

        run_state.artifacts[artifact_id] = record
        return record

    @staticmethod
    async def resolve(
        run_state,
        artifact_id: str,
        expected_status: ArtifactStatus = ArtifactStatus.READY,
    ) -> ArtifactRecord:
        """Resolve an artifact_id to its record.

        Raises:
            KeyError: artifact_id not found.
            ValueError: artifact status != expected_status.
        """
        record = run_state.artifacts.get(artifact_id)
        if record is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        if record.status != expected_status:
            raise ValueError(
                f"Artifact {artifact_id} is {record.status.value}, "
                f"expected {expected_status.value}"
            )
        return record

    @staticmethod
    async def mark_consumed(run_state, artifact_id: str, consumer: AgentName):
        """Mark an artifact as consumed by a downstream agent."""
        record = run_state.artifacts.get(artifact_id)
        if record is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        if record.status != ArtifactStatus.READY:
            raise ValueError(
                f"Cannot consume artifact {artifact_id}: status is {record.status.value}"
            )
        record.status = ArtifactStatus.CONSUMED

    @staticmethod
    async def mark_stale(run_state, artifact_id: str):
        """Mark an artifact as stale (upstream produced a new version)."""
        record = run_state.artifacts.get(artifact_id)
        if record is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        record.status = ArtifactStatus.STALE
