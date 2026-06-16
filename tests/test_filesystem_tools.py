"""测试文件系统工具。"""

import pytest
from tower.tools.builtin.filesystem import (
    read_file, write_file, edit_file, list_directory,
    glob_tool, grep, move_file, copy_file, delete_file,
)


class TestReadFile:
    def test_read_existing_file(self, tmpdir):
        f = tmpdir / "hello.txt"
        f.write_text("hello world")
        result = read_file.invoke({"path": str(f)})
        assert result["content"] == "hello world"
        assert "error" not in result

    def test_read_nonexistent_file(self):
        result = read_file.invoke({"path": "/nonexistent/xyz123.txt"})
        assert "error" in result

    def test_read_directory_returns_error(self, tmpdir):
        result = read_file.invoke({"path": str(tmpdir)})
        assert "error" in result

    def test_read_system_path_blocked(self):
        result = read_file.invoke({"path": "/etc/passwd"})
        assert "error" in result
        assert "system directory" in result["error"]


class TestWriteFile:
    def test_write_new_file(self, tmpdir):
        f = tmpdir / "output.txt"
        result = write_file.invoke({"path": str(f), "content": "hello"})
        assert "ok" in result
        assert f.read_text() == "hello"

    def test_write_overwrites_existing(self, tmpdir):
        f = tmpdir / "existing.txt"
        f.write_text("old")
        result = write_file.invoke({"path": str(f), "content": "new"})
        assert "ok" in result
        assert f.read_text() == "new"

    def test_write_creates_parent_dirs(self, tmpdir):
        f = tmpdir / "sub" / "deep" / "file.txt"
        result = write_file.invoke({"path": str(f), "content": "nested"})
        assert "ok" in result
        assert f.read_text() == "nested"

    def test_write_system_path_blocked(self):
        result = write_file.invoke({"path": "/etc/tower_test.txt", "content": "test"})
        assert "error" in result


class TestEditFile:
    def test_basic_replacement(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("x = 1\ny = 2\nz = 3\n")
        result = edit_file.invoke({
            "path": str(f), "old_string": "y = 2", "new_string": "y = 42"
        })
        assert "ok" in result
        assert f.read_text() == "x = 1\ny = 42\nz = 3\n"

    def test_old_string_not_found(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("hello")
        result = edit_file.invoke({
            "path": str(f), "old_string": "nonexistent", "new_string": "replaced"
        })
        assert "error" in result
        assert "not found" in result["error"]

    def test_duplicate_old_string(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("TODO: fix bug\nTODO: fix bug\n")
        result = edit_file.invoke({
            "path": str(f), "old_string": "TODO: fix bug", "new_string": "DONE"
        })
        assert "error" in result
        assert "appears 2 times" in result["error"]
        assert result["occurrences"] == 2

    def test_empty_old_string_rejected(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("hello")
        result = edit_file.invoke({
            "path": str(f), "old_string": "", "new_string": "x"
        })
        assert "error" in result

    def test_file_not_found(self):
        result = edit_file.invoke({
            "path": "/nonexistent/x.py", "old_string": "a", "new_string": "b"
        })
        assert "error" in result


class TestListDirectory:
    def test_lists_files_and_dirs(self, tmpdir):
        (tmpdir / "a.txt").write_text("a")
        (tmpdir / "b.txt").write_text("b")
        (tmpdir / "subdir").mkdir()
        result = list_directory.invoke({"path": str(tmpdir)})
        assert "error" not in result
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names
        assert result["count"] == 3

    def test_empty_directory(self, tmpdir):
        result = list_directory.invoke({"path": str(tmpdir)})
        assert result["entries"] == []
        assert result["count"] == 0

    def test_nonexistent_directory(self):
        result = list_directory.invoke({"path": "/nonexistent/dir"})
        assert "error" in result

    def test_not_a_directory(self, tmpdir):
        f = tmpdir / "file.txt"
        f.write_text("content")
        result = list_directory.invoke({"path": str(f)})
        assert "error" in result


class TestGlobTool:
    def test_find_python_files(self, tmpdir):
        (tmpdir / "a.py").write_text("")
        (tmpdir / "b.py").write_text("")
        (tmpdir / "c.txt").write_text("")
        result = glob_tool.invoke({"pattern": "*.py", "path": str(tmpdir)})
        assert result["count"] == 2
        assert any("a.py" in f for f in result["files"])

    def test_recursive_search(self, tmpdir):
        sub = tmpdir / "sub"
        sub.mkdir()
        (tmpdir / "root.py").write_text("")
        (sub / "deep.py").write_text("")
        result = glob_tool.invoke({"pattern": "**/*.py", "path": str(tmpdir)})
        assert result["count"] == 2

    def test_no_matches(self, tmpdir):
        result = glob_tool.invoke({"pattern": "*.rs", "path": str(tmpdir)})
        assert result["count"] == 0


class TestGrep:
    def test_find_pattern_in_files(self, tmpdir):
        (tmpdir / "a.py").write_text("def foo():\n    pass\n")
        (tmpdir / "b.py").write_text("def bar():\n    pass\n")
        result = grep.invoke({"pattern": "def foo", "path": str(tmpdir)})
        assert "error" not in result
        assert result["files_matched"] == 1

    def test_no_match(self, tmpdir):
        (tmpdir / "code.py").write_text("hello world")
        result = grep.invoke({"pattern": "nonexistent_xyz", "path": str(tmpdir)})
        assert result["files_matched"] == 0

    def test_multiple_files_matched(self, tmpdir):
        (tmpdir / "a.py").write_text("TODO: refactor")
        (tmpdir / "b.py").write_text("TODO: fix")
        result = grep.invoke({"pattern": "TODO", "path": str(tmpdir)})
        assert result["files_matched"] == 2


class TestMoveFile:
    def test_rename_file(self, tmpdir):
        src = tmpdir / "old.txt"
        src.write_text("data")
        result = move_file.invoke({
            "source": str(src), "destination": str(tmpdir / "new.txt")
        })
        assert "ok" in result
        assert not src.exists()
        assert (tmpdir / "new.txt").read_text() == "data"

    def test_source_not_found(self):
        result = move_file.invoke({
            "source": "/nonexistent/x.txt", "destination": "/tmp/y.txt"
        })
        assert "error" in result

    def test_destination_exists(self, tmpdir):
        src = tmpdir / "a.txt"
        dst = tmpdir / "b.txt"
        src.write_text("a")
        dst.write_text("b")
        result = move_file.invoke({
            "source": str(src), "destination": str(dst)
        })
        assert "error" in result


class TestCopyFile:
    def test_basic_copy(self, tmpdir):
        src = tmpdir / "orig.txt"
        src.write_text("important data")
        result = copy_file.invoke({
            "source": str(src), "destination": str(tmpdir / "copy.txt")
        })
        assert "ok" in result
        assert (tmpdir / "copy.txt").read_text() == "important data"

    def test_copy_preserves_original(self, tmpdir):
        src = tmpdir / "orig.txt"
        src.write_text("data")
        copy_file.invoke({
            "source": str(src), "destination": str(tmpdir / "dup.txt")
        })
        assert src.exists()

    def test_source_directory_rejected(self, tmpdir):
        sub = tmpdir / "sub"
        sub.mkdir()
        result = copy_file.invoke({
            "source": str(sub), "destination": str(tmpdir / "copy_sub")
        })
        assert "error" in result


class TestDeleteFile:
    def test_delete_existing_file(self, tmpdir):
        f = tmpdir / "temp.txt"
        f.write_text("temp")
        result = delete_file.invoke({"path": str(f)})
        assert "ok" in result
        assert not f.exists()

    def test_delete_nonexistent(self):
        result = delete_file.invoke({"path": "/nonexistent/file.txt"})
        assert "error" in result

    def test_delete_directory_rejected(self, tmpdir):
        sub = tmpdir / "dir"
        sub.mkdir()
        result = delete_file.invoke({"path": str(sub)})
        assert "error" in result
