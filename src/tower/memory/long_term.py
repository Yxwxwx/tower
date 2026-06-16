"""长期记忆 —— PostgresStore 跨会话持久化。

LangGraph Store 原生支持 PostgreSQL + pgvector。
使用 ("users", user_id) 作为 namespace 隔离不同用户。
"""
import hashlib
from langgraph.store.postgres import PostgresStore
from tower.memory.pool import get_pool

_setup_done = False


class LongTermMemory:
    """基于 LangGraph PostgresStore 的跨会话长期记忆。"""

    def __init__(self, conn_string: str, user_id: str = "default"):
        global _setup_done
        self._store = PostgresStore(get_pool(conn_string))
        if not _setup_done:
            self._store.setup()
            _setup_done = True
        self._namespace = ("users", user_id)

    def put(self, key: str, value: dict):
        """写入一条记忆。"""
        self._store.put(self._namespace, key, value)

    def get(self, key: str) -> dict | None:
        """读取一条记忆。"""
        item = self._store.get(self._namespace, key)
        if item is None:
            return None
        return item.value if hasattr(item, "value") else item.get("value", {})

    def search(self, filter_keys: list[str] | None = None) -> list[dict]:
        """搜索记忆，可按 key 过滤。"""
        items = self._store.search(self._namespace, limit=50)
        results = []
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            k = item.key if hasattr(item, "key") else item.get("key", "")
            if filter_keys is None or k in filter_keys:
                results.append({"key": k, "value": v})
        return results

    def get_all_facts(self, limit: int = 50) -> list[dict]:
        """获取所有事实（兼容旧接口）。"""
        items = self._store.search(self._namespace, limit=limit)
        facts = []
        for item in items:
            v = item.value if hasattr(item, "value") else item.get("value", {})
            if isinstance(v, dict) and "fact" in v:
                facts.append({"fact": v["fact"], "category": v.get("category", "general")})
        return facts

    def add_fact(self, fact: str, category: str = "general"):
        """添加一条事实（自动去重，O(1) 查找）。"""
        # 用 MD5 做确定性 key，跨进程一致且防碰撞
        key = "fact_" + hashlib.md5(fact.encode()).hexdigest()[:12]
        existing = self.get(key)
        if existing is not None:
            return  # 已存在
        self.put(key, {"fact": fact, "category": category})

    def remove_fact_by_text(self, text: str):
        """删除包含指定文本的记忆（模糊匹配，用于移除矛盾事实）。"""
        for item in self._store.search(self._namespace, limit=100):
            v = item.value if hasattr(item, "value") else item.get("value", {})
            stored_fact = v.get("fact", "") if isinstance(v, dict) else ""
            if text in stored_fact or stored_fact in text:
                self._store.delete(self._namespace, item.key)

    def clear(self):
        """清空当前用户的所有记忆。"""
        for item in self._store.search(self._namespace, limit=1000):
            self._store.delete(self._namespace, item.key)
