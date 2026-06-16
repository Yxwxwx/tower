"""Tool 注册表 —— 集中管理所有 LLM 可调用的工具。"""

from tower.tools.builtin.bash import bash
from tower.tools.builtin.filesystem import (
    read_file,
    write_file,
    edit_file,
    list_directory,
    glob_tool,
    grep,
    move_file,
    copy_file,
    delete_file,
)
from tower.tools.builtin.web import web_fetch, web_search

TOOLS = [
    # 执行
    bash,
    # 文件读/写/编辑
    read_file,
    write_file,
    edit_file,
    # 浏览/查找
    list_directory,
    glob_tool,
    grep,
    # 文件管理
    move_file,
    copy_file,
    delete_file,
    # 网络
    web_fetch,
    web_search,
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}
