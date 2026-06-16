"""pytest 配置和共享 fixtures。"""

import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmpdir():
    """创建临时目录，测试结束后自动清理。"""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
