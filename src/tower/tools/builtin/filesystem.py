"""内置文件工具 —— 读、写、编辑、搜索、管理文件。

所有写入/删除操作都会经过 safety 模块校验：
- 禁止操作系统关键目录
- 不允许 root 权限
- 相对路径基于 cwd 解析
"""

import fnmatch
from pathlib import Path
from langchain_core.tools import tool
from tower.tools.safety import resolve_safe_path, validate_bash_command, ensure_in_project


# ============================================================
# 辅助函数
# ============================================================

def _safe_resolve(path: str, write_op: bool = False) -> Path:
    """解析并校验路径。"""
    resolved = resolve_safe_path(path, write_op=write_op)
    return ensure_in_project(resolved, write_op=write_op)


# ============================================================
# 已有工具（升级为 safety-aware）
# ============================================================


@tool
def read_file(path: str) -> dict:
    """Read the contents of a file.

    Use for: checking file contents, reading configs, examining code,
    understanding error messages, reviewing log output.

    Args:
        path: Path to the file (relative or absolute).
    """
    try:
        p = _safe_resolve(path, write_op=False)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if p.is_dir():
            return {"error": f"Path is a directory, not a file: {path}"}
        content = p.read_text()
        total = len(content)
        return {"path": str(p), "content": content[:10000], "total_chars": total}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@tool
def write_file(path: str, content: str) -> dict:
    """Write content to a file, creating parent directories if needed.

    Use for: creating new scripts, saving output, writing config files.
    Will OVERWRITE the file if it already exists.

    Args:
        path: Path to the file (relative or absolute).
        content: The text content to write.
    """
    try:
        p = _safe_resolve(path, write_op=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"ok": f"Wrote {len(content)} chars to {p}"}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 精确编辑工具
# ============================================================


@tool
def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """Replace old_string with new_string in a file. The old_string must
    appear exactly once in the file — this is a safety feature to prevent
    accidental edits. Whitespace and indentation matter.

    Use for: fixing bugs, renaming variables, updating config values,
    adding/removing lines — any targeted change without rewriting the whole file.

    Args:
        path: Path to the file.
        old_string: The exact text to replace (must match exactly once).
        new_string: The replacement text.
    """
    try:
        p = _safe_resolve(path, write_op=True)
        if not p.exists():
            return {"error": f"File not found: {path}"}

        content = p.read_text()

        count = content.count(old_string)
        if count == 0:
            return {
                "error": (
                    f"old_string not found in {path}. "
                    f"Check that the text matches exactly (whitespace, indentation, newlines)."
                ),
                "hint": "Try read_file first to check the file's exact content.",
            }
        if count > 1:
            return {
                "error": (
                    f"old_string appears {count} times in {path}. "
                    f"It must be unique so the edit is unambiguous. "
                    f"Please include more surrounding context to make it unique."
                ),
                "occurrences": count,
            }

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content)

        # 计算变化统计
        added = len(new_string) - len(old_string)
        return {
            "ok": f"Edited {p}: replaced {len(old_string)} chars → {len(new_string)} chars (Δ{added:+d})",
            "path": str(p),
        }
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 浏览工具
# ============================================================


@tool
def list_directory(path: str) -> dict:
    """List the contents of a directory. Returns file names, types (file/dir/symlink),
    and file sizes. Entries are sorted alphabetically with directories first.

    Use for: exploring project structure, checking what files exist,
    navigating unfamiliar codebases.

    Args:
        path: Path to the directory (relative or absolute).
    """
    try:
        p = _safe_resolve(path, write_op=False)
        if not p.exists():
            return {"error": f"Directory not found: {path}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = []
        for entry in sorted(p.iterdir()):
            try:
                etype = "dir" if entry.is_dir() else "symlink" if entry.is_symlink() else "file"
                size = entry.stat().st_size if entry.is_file() else None
            except OSError:
                etype = "unknown"
                size = None

            entries.append({
                "name": entry.name,
                "type": etype,
                "size": size,
            })

        return {
            "path": str(p),
            "entries": entries,
            "count": len(entries),
        }
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 查找工具
# ============================================================


@tool
def glob_tool(pattern: str, path: str = ".") -> dict:
    """Find files matching a glob pattern. Supports ** for recursive matching.

    Use for: finding all Python files in a project, locating test files,
    finding config files by pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.toml").
        path: Base directory to search from (default: current directory).
    """
    try:
        p = _safe_resolve(path, write_op=False)
        if not p.exists():
            return {"error": f"Directory not found: {path}"}

        matches = sorted(p.glob(pattern))
        # 过滤掉隐藏目录和 __pycache__
        filtered = [
            str(m) for m in matches
            if "__pycache__" not in str(m)
            and not any(part.startswith(".") and part != "." for part in m.parts[len(p.parts):])
        ]

        return {
            "pattern": pattern,
            "base": str(p),
            "files": filtered[:200],
            "count": len(filtered),
        }
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@tool
def grep(pattern: str, path: str = ".", glob: str = "*") -> dict:
    """Search for a text pattern in files. Uses Python regex by default
    (case-sensitive). Supports file filtering by glob.

    Use for: finding where a function is defined, searching for error messages,
    locating TODO comments, finding all uses of a class or variable.

    Args:
        pattern: Regex pattern or plain text to search for.
        path: Directory to search in (default: current directory).
        glob: File glob filter (e.g., "*.py", "*.{py,js,ts}").
    """
    try:
        import re as _re

        p = _safe_resolve(path, write_op=False)
        if not p.exists():
            return {"error": f"Directory not found: {path}"}

        # 编译正则
        try:
            regex = _re.compile(pattern)
        except _re.error:
            # 如果不是合法的正则，当作字面量搜索
            regex = _re.compile(_re.escape(pattern))

        results = []
        file_count = 0
        match_limit = 100

        for file_path in sorted(p.rglob(glob)):
            # 跳过隐藏目录和缓存
            parts = file_path.parts
            if any(part.startswith(".") for part in parts if part not in (".", "..")):
                continue
            if "__pycache__" in parts or "node_modules" in parts or ".git" in parts:
                continue
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text()
            except Exception:
                continue

            file_results = []
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    file_results.append({"line": i, "content": line.strip()[:200]})

            if file_results:
                file_count += 1
                results.append({
                    "file": str(file_path),
                    "matches": file_results[:20],  # 每个文件最多 20 条
                    "match_count": len(file_results),
                })

            if len(results) >= 50:  # 最多 50 个文件
                break

        return {
            "pattern": pattern,
            "base": str(p),
            "results": results,
            "files_matched": file_count,
            "total_matches": sum(r["match_count"] for r in results),
        }
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 文件管理工具
# ============================================================


@tool
def move_file(source: str, destination: str) -> dict:
    """Move or rename a file or directory.

    Use for: renaming files, moving files between directories,
    reorganizing project structure.

    Args:
        source: Current path of the file/directory.
        destination: New path for the file/directory.
    """
    try:
        src = _safe_resolve(source, write_op=True)
        dst = _safe_resolve(destination, write_op=True)

        if not src.exists():
            return {"error": f"Source not found: {source}"}
        if dst.exists():
            return {"error": f"Destination already exists: {destination}"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {"ok": f"Moved {src} → {dst}"}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@tool
def copy_file(source: str, destination: str) -> dict:
    """Copy a file to a new location. Preserves file content.
    Does NOT preserve file metadata (permissions, timestamps).

    Use for: duplicating templates, creating backups before editing,
    copying config files.

    Args:
        source: Path to the source file.
        destination: Path for the copy.
    """
    try:
        src = _safe_resolve(source, write_op=False)
        dst = _safe_resolve(destination, write_op=True)

        if not src.exists():
            return {"error": f"Source not found: {source}"}
        if src.is_dir():
            return {"error": f"Source is a directory, use bash cp -r for directories: {source}"}
        if dst.exists():
            return {"error": f"Destination already exists: {destination}"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())
        return {"ok": f"Copied {src} → {dst}"}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@tool
def delete_file(path: str) -> dict:
    """Delete a file. This is DESTRUCTIVE and requires user approval.
    Will NOT delete directories (use bash for that).

    Use for: removing temporary files, cleaning up generated output,
    deleting old scripts.

    Args:
        path: Path to the file to delete.
    """
    try:
        p = _safe_resolve(path, write_op=True)

        if not p.exists():
            return {"error": f"File not found: {path}"}
        if p.is_dir():
            return {"error": f"Path is a directory, use bash rm -rf for directories: {path}"}

        p.unlink()
        return {"ok": f"Deleted {p}"}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
