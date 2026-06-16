"""短期记忆 —— PostgresSaver 持久化会话内 state。

每次 graph.invoke() 后 LangGraph 自动保存 checkpoint。
相同 thread_id 可恢复历史会话。
"""
from langgraph.checkpoint.postgres import PostgresSaver
from tower.memory.pool import get_pool

_setup_done = False


def create_checkpointer(conn_string: str) -> PostgresSaver:
    """创建 PostgresSaver，使用共享连接池。"""
    global _setup_done
    pool = get_pool(conn_string)
    saver = PostgresSaver(pool)
    if not _setup_done:
        saver.setup()
        _setup_done = True
    return saver
