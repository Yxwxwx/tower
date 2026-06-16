"""测试安全校验模块。"""

import pytest
from tower.tools.safety import (
    resolve_safe_path,
    validate_bash_command,
    check_root_privilege,
    is_forbidden_dir,
    FORBIDDEN_DIRS,
    DANGEROUS_BASH_PATTERNS,
)


class TestResolveSafePath:
    def test_normal_file_in_cwd(self):
        p = resolve_safe_path("test.py")
        assert p.is_absolute()
        assert p.name == "test.py"

    def test_relative_path_resolves_to_cwd(self):
        p = resolve_safe_path("src/main.py")
        assert p.is_absolute()
        assert "src" in str(p)

    def test_absolute_path_outside_forbidden_is_allowed(self, tmpdir):
        p = resolve_safe_path(str(tmpdir / "data.txt"))
        # macOS 上 /var → /private/var symlink，Path.resolve() 会跟随
        assert p.resolve() == (tmpdir / "data.txt").resolve()

    def test_system_dir_etc_is_blocked(self):
        with pytest.raises(PermissionError, match="system directory"):
            resolve_safe_path("/etc/passwd")

    def test_system_dir_usr_is_blocked(self):
        with pytest.raises(PermissionError, match="system directory"):
            resolve_safe_path("/usr/bin/python3")

    def test_system_dir_System_is_blocked(self):
        with pytest.raises(PermissionError, match="system directory"):
            resolve_safe_path("/System/Library/test")

    def test_user_home_not_blocked(self):
        # 用户 home 目录应该允许访问
        import os
        home = os.path.expanduser("~")
        p = resolve_safe_path(f"{home}/Documents/test.txt")
        assert home in str(p)

    def test_write_op_checks_extra_dirs(self):
        import os
        ssh_dir = os.path.expanduser("~/.ssh/known_hosts")
        with pytest.raises(PermissionError, match="system directory"):
            resolve_safe_path(ssh_dir, write_op=True)


class TestValidateBash:
    def test_normal_command_passes(self):
        cmd = "ls -la"
        assert validate_bash_command(cmd) == cmd

    def test_sudo_is_blocked(self):
        with pytest.raises(ValueError, match="dangerous command"):
            validate_bash_command("sudo rm -rf /tmp/test")

    def test_fork_bomb_is_blocked(self):
        with pytest.raises(ValueError, match="dangerous command"):
            validate_bash_command(":(){ :|:& };:")

    def test_mkfs_is_blocked(self):
        with pytest.raises(ValueError, match="dangerous command"):
            validate_bash_command("mkfs.ext4 /dev/sda1")

    def test_chmod_777_root_is_blocked(self):
        with pytest.raises(ValueError, match="dangerous command"):
            validate_bash_command("chmod 777 /")


class TestIsForbiddenDir:
    def test_exact_match(self):
        assert is_forbidden_dir("/etc") == "/etc"

    def test_subdir_match(self):
        assert is_forbidden_dir("/etc/nginx/conf.d") == "/etc"

    def test_normal_dir_not_forbidden(self):
        assert is_forbidden_dir("/home/user/project") is None

    def test_partial_name_not_confused(self):
        # 部分匹配不触发误报
        assert is_forbidden_dir("/home/user/etc_backup") is None


class TestCheckRootPrivilege:
    def test_not_running_as_root(self):
        # 正常情况不应该以 root 运行
        assert check_root_privilege() is False


class TestForbiddenDirsCoverage:
    def test_all_forbidden_dirs_are_absolute(self):
        for d in FORBIDDEN_DIRS:
            assert d.startswith("/") or ":" in d, f"{d} 不是绝对路径"

    def test_all_dangerous_patterns_are_non_empty(self):
        for p in DANGEROUS_BASH_PATTERNS:
            assert len(p) > 0
