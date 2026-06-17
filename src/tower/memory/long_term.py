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

    async def search(
        self, filter_keys: list[str] | None = None, limit: int = 50
    ) -> list[dict]:
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
                facts.append(
                    {"fact": v["fact"], "category": v.get("category", "general")}
                )
        return facts

    async def add_fact(self, fact: str, category: str = "general"):
        """添加一条事实（自动去重，O(1) 查找）。"""
        # 用 MD5 做确定性 key，跨进程一致且防碰撞
        key = "fact_" + hashlib.md5(fact.encode()).hexdigest()[:12]
        existing = await self.get(key)
        if existing is not None:
            return  # 已存在
        await self.put(key, {"fact": fact, "category": category})

    async def remove_fact_by_text(self, text: str):
        """删除包含指定文本的记忆（模糊匹配，用于移除矛盾事实）。"""
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

    # ═══════════════════════════════════════════════════════════════
    # Memory OS — add_record, search_by_type, mark_accessed
    # ═══════════════════════════════════════════════════════════════

    async def add_record(self, record) -> str | None:
        """Add a MemoryRecord with deduplication.

        Returns memory_id if stored, None if duplicate.
        On duplicate: averages confidence scores.
        """
        from tower.memory.models import MemoryRecord

        await self._ensure_setup()

        existing = await self.get(record.memory_id)
        if existing is not None:
            # Update confidence: average
            old_conf = existing.get("confidence", 1.0)
            if isinstance(record, MemoryRecord):
                record.confidence = (old_conf + record.confidence) / 2
            # Update access metadata
            existing["last_accessed_at"] = record.last_accessed_at
            existing["access_count"] = existing.get("access_count", 0) + 1
            await self.put(record.memory_id, existing)
            return None  # duplicate

        await self.put(record.memory_id, record.model_dump())
        return record.memory_id

    async def search_by_type(
        self,
        memory_type,
        limit: int = 50,
        **filters,
    ) -> list:
        """Type-filtered search without embedding.

        Args:
            memory_type: MemoryType enum value or string.
            limit: Max records to return.
            **filters: Additional key-value filters on record fields.

        Returns:
            List of MemoryRecord objects.
        """
        from tower.memory.models import MemoryRecord

        await self._ensure_setup()
        items = await self._store.asearch(self._namespace, limit=limit * 3)
        records = []
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            if isinstance(v, dict) and v.get("memory_type") == memory_type:
                try:
                    records.append(MemoryRecord(**v))
                except Exception:
                    continue
        return records[:limit]

    async def mark_accessed(self, memory_id: str):
        """Update access metadata for a memory record."""
        existing = await self.get(memory_id)
        if existing:
            from datetime import datetime

            existing["last_accessed_at"] = datetime.now().isoformat()
            existing["access_count"] = existing.get("access_count", 0) + 1
            await self.put(memory_id, existing)
