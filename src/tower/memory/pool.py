"""共享异步连接池 —— 整个进程复用同一个 PostgreSQL 连接池。"""
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None
_conn_string: str = ""


async def get_pool(conn_string: str) -> AsyncConnectionPool:
    """获取或创建共享异步连接池。"""
    global _pool, _conn_string
    if _pool is None or _conn_string != conn_string:
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
        _pool = AsyncConnectionPool(
            conn_string,
            min_size=1,
            max_size=3,
            open=True,
        )
        _conn_string = conn_string
    return _pool


async def close_pool():
    """关闭连接池（用于 graceful shutdown）。"""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = None
