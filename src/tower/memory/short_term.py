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
