"""共享连接池 —— 整个进程复用同一个 PostgreSQL 连接池。"""
import atexit
import psycopg_pool

_pool: psycopg_pool.ConnectionPool | None = None
_conn_string: str = ""


def get_pool(conn_string: str) -> psycopg_pool.ConnectionPool:
    """获取或创建共享连接池。"""
    global _pool, _conn_string
    if _pool is None or _conn_string != conn_string:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
        _pool = psycopg_pool.ConnectionPool(
            conn_string,
            min_size=1,
            max_size=3,
            open=True,
        )
        _conn_string = conn_string
    return _pool


def _cleanup():
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass


atexit.register(_cleanup)
