"""共享安全校验 —— 所有文件工具和 bash 工具的安全层。

规则：
1. 禁止操作系统关键目录（/etc, /usr, /System 等）
2. 不允许 root 权限执行
3. 路径优先在当前工作目录下解析
4. bash 拒绝 sudo 等危险模式
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

# ============================================================
# 系统目录黑名单
# ============================================================

FORBIDDEN_DIRS: list[str] = [
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "/var/log",
    "/var/run",
    # macOS
    "/System",
    "/Library/System",
    "/Library/Extensions",
    "/Library/LaunchDaemons",
    # Windows
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\System32",
]

# 写操作 / 删除操作 额外禁止的目录
FORBIDDEN_WRITE_DIRS: list[str] = [
    # 用户级关键目录
    os.path.expanduser("~/.ssh"),
    os.path.expanduser("~/.gnupg"),
    os.path.expanduser("~/Library/Keychains"),
]

DANGEROUS_BASH_PATTERNS: list[str] = [
    "sudo ",
    "rm -rf /",
    "rm -rf ~",
    "mkfs.",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "> /dev/sda",
    "chmod 777 /",
    "chown -R",
]


# ============================================================
# 安全检查函数
# ============================================================

def check_root_privilege() -> bool:
    """检测是否以 root 运行。"""
    try:
        return os.geteuid() == 0
    except AttributeError:
        # Windows 上没有 geteuid
        return False


def is_forbidden_dir(path: str, write_op: bool = False) -> str | None:
    """检查路径是否在黑名单中。返回命中的禁止目录，或 None。

    Args:
        path: 要检查的绝对路径
        write_op: 是否是写/删除操作（会检查额外的禁止目录）
    """
    path_str = str(path).rstrip(os.sep)

    for forbidden in FORBIDDEN_DIRS:
        if path_str == forbidden or path_str.startswith(forbidden + os.sep):
            return forbidden

    if write_op:
        for forbidden in FORBIDDEN_WRITE_DIRS:
            if path_str == forbidden or path_str.startswith(forbidden + os.sep):
                return forbidden

    return None


def resolve_safe_path(path: str, write_op: bool = False) -> Path:
    """安全解析路径：相对路径基于 cwd，拒绝系统目录。

    Args:
        path: 用户提供的路径
        write_op: 是否是写/删除操作

    Returns:
        解析后的绝对 Path

    Raises:
        PermissionError: 目标在禁止目录中
    """
    if check_root_privilege():
        raise PermissionError("Running as root is not allowed for safety")

    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p

    # 先检查未解析的路径（防止 /etc → /private/etc 这类 symlink 绕过黑名单）
    forbidden = is_forbidden_dir(str(p), write_op=write_op)
    if forbidden:
        raise PermissionError(
            f"Safety: cannot operate on system directory '{forbidden}'. "
            f"Tower tools are restricted to project directories."
        )

    p = p.resolve()

    # 解析后再检查一次（防止 ../../etc 绕过）
    forbidden = is_forbidden_dir(str(p), write_op=write_op)
    if forbidden:
        raise PermissionError(
            f"Safety: cannot operate on system directory '{forbidden}'. "
            f"Tower tools are restricted to project directories."
        )

    return p


def validate_bash_command(command: str) -> str:
    """检查 bash 命令是否包含危险模式。

    Raises:
        PermissionError: 以 root 运行
        ValueError: 包含危险命令模式
    """
    if check_root_privilege():
        raise PermissionError("Running bash as root is not allowed")

    for pattern in DANGEROUS_BASH_PATTERNS:
        if pattern in command.lower():
            raise ValueError(
                f"Safety: dangerous command pattern '{pattern}' detected. "
                f"Tower cannot execute commands that could damage the system."
            )

    return command


def find_git_root() -> Path | None:
    """查找当前目录所在的 git 仓库根。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=Path.cwd(),
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _is_temp_path(path: Path) -> bool:
    """检测路径是否在系统临时目录下（测试 / 临时文件场景）。"""
    try:
        tmp = Path(tempfile.gettempdir()).resolve()
        return str(path.resolve()).startswith(str(tmp))
    except Exception:
        return False


def ensure_in_project(path: Path, write_op: bool = False) -> Path:
    """确保路径在项目目录内（如果有 git repo，限制在 repo 内）。

    读操作：仅警告（允许读取外部配置文件等）。
    写操作：硬阻止——拒绝向 git repo 之外的目录写入。
    临时目录（/tmp 等）自动豁免，避免影响测试。
    """
    git_root = find_git_root()
    if not git_root:
        return path

    path_str = str(path.resolve())
    repo_str = str(git_root.resolve())

    if path_str.startswith(repo_str):
        return path  # 在 repo 内，放行

    # 临时目录豁免（测试 fixture 等场景）
    if _is_temp_path(path):
        return path

    if write_op:
        raise PermissionError(
            f"Safety: cannot write to '{path}' — it is outside the git "
            f"repository root '{git_root}'. Tower write operations are "
            f"restricted to the project directory."
        )

    # 读操作：软限制
    import warnings
    warnings.warn(
        f"Path '{path}' is outside the git repository root '{git_root}'. "
        f"Consider working within the project directory.",
        stacklevel=2,
    )
    return path


# ============================================================
# 工具结果格式化辅助
# ============================================================

def ok(message: str) -> dict:
    """生成成功结果。"""
    return {"ok": message}


def error(message: str) -> dict:
    """生成错误结果。"""
    return {"error": message}
