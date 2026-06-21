"""动作执行器测试"""
import os
import tempfile


from root.left.app.left.src.core_mod.action_executor import ActionExecutor, ActionResult


class TestActionResult:
    """ActionResult 测试"""

    def test_to_dict(self) -> None:
        result = ActionResult(success=True, action_type="shell", stdout="ok", returncode=0)
        d = result.to_dict()
        assert d["success"] is True
        assert d["action_type"] == "shell"
        assert d["stdout"] == "ok"


class TestActionExecutorShell:
    """Shell 动作测试"""

    def test_shell_dry_run(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        result = executor.execute({"action_type": "shell", "command": "echo hello"})
        assert result.success is True
        assert "[dry_run]" in result.stdout

    def test_shell_real_echo(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "shell", "command": "echo hello"})
        assert result.success is True
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_shell_blocked_command(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "shell", "command": "rm -rf / tmp"})
        assert result.success is False
        assert "Blocked" in result.error

    def test_shell_not_allowed_prefix(self) -> None:
        executor = ActionExecutor({"dry_run": False, "allowed_shell_prefixes": ["echo"]})
        result = executor.execute({"action_type": "shell", "command": "cat /etc/hosts"})
        assert result.success is False
        assert "not in allowed list" in result.error

    def test_shell_empty_command(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "shell", "command": ""})
        assert result.success is False


class TestActionExecutorHttp:
    """HTTP 动作测试"""

    def test_http_get_dry_run(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        result = executor.execute({"action_type": "http_get", "url": "http://localhost:8080/health"})
        assert result.success is True
        assert "[dry_run]" in result.stdout

    def test_http_get_blocked_host(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "http_get", "url": "http://example.com/"})
        assert result.success is False
        assert "not in allowed list" in result.error


class TestActionExecutorApproval:
    """审批门集成测试"""

    def test_destructive_auto_reject(self) -> None:
        executor = ActionExecutor({"dry_run": False, "approval_gate": {"mode": "auto_reject"}})
        result = executor.execute({"action_type": "file_delete", "path": "/tmp/x"})
        assert result.success is False
        assert "rejected" in result.error.lower()

    def test_destructive_auto_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.txt")
            with open(path, "w") as f:
                f.write("x")
            executor = ActionExecutor({"dry_run": False, "approval_gate": {"mode": "auto_pass"}})
            result = executor.execute({"action_type": "file_delete", "path": path})
            assert result.success is True


class TestActionExecutorFile:
    """文件动作测试"""

    def test_file_write_read_delete(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            write_result = executor.execute({"action_type": "file_write", "path": path, "content": "hello"})
            assert write_result.success is True

            read_result = executor.execute({"action_type": "file_read", "path": path})
            assert read_result.success is True
            assert read_result.data == "hello"

            delete_result = executor.execute({"action_type": "file_delete", "path": path})
            assert delete_result.success is True
            assert not os.path.exists(path)

    def test_file_blocked_path(self) -> None:
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "file_read", "path": "/etc/passwd"})
        assert result.success is False
        assert "blocked" in result.error.lower()


class TestActionExecutorNoop:
    """Noop 动作测试"""

    def test_noop(self) -> None:
        executor = ActionExecutor({})
        result = executor.execute({"action_type": "noop"})
        assert result.success is True

    def test_unknown_action(self) -> None:
        executor = ActionExecutor({})
        result = executor.execute({"action_type": "magic"})
        assert result.success is False
        assert "Unknown" in result.error


class TestActionExecutorUnifiedInterface:
    """统一接口 get_state/reset/health_check 测试"""

    def test_get_state(self) -> None:
        executor = ActionExecutor({"dry_run": True, "timeout": 10.0})
        state = executor.get_state()
        assert state["dry_run"] is True
        assert state["timeout"] == 10.0
        assert "allowed_shell_prefixes" in state

    def test_reset_does_not_raise(self) -> None:
        """reset() 应调用 approval_gate.reset() 且不抛异常"""
        executor = ActionExecutor({})
        # 先执行一些动作
        executor.execute({"action_type": "noop"})
        # reset 不应抛异常
        executor.reset()

    def test_reset_preserves_config(self) -> None:
        """reset() 应保留配置项"""
        executor = ActionExecutor({"dry_run": False, "timeout": 15.0})
        executor.reset()
        state = executor.get_state()
        assert state["dry_run"] is False
        assert state["timeout"] == 15.0

    def test_health_check_normal(self) -> None:
        executor = ActionExecutor({})
        assert executor.health_check() is True

    def test_health_check_empty_prefixes(self) -> None:
        """空 allowed_shell_prefixes 应判定为不健康"""
        executor = ActionExecutor({"allowed_shell_prefixes": []})
        assert executor.health_check() is False

    def test_health_check_invalid_timeout(self) -> None:
        """非正 timeout 应判定为不健康"""
        executor = ActionExecutor({"timeout": 0})
        assert executor.health_check() is False

    def test_health_check_negative_timeout(self) -> None:
        executor = ActionExecutor({"timeout": -1})
        assert executor.health_check() is False

    def test_health_check_after_reset(self) -> None:
        executor = ActionExecutor({})
        executor.reset()
        assert executor.health_check() is True


class TestActionExecutorHistory:
    """执行历史记录测试"""

    def test_history_empty_initially(self) -> None:
        executor = ActionExecutor({})
        assert executor.get_history() == []
        assert executor.get_history(limit=5) == []

    def test_history_records_each_execution(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "shell", "command": "echo hi"})
        history = executor.get_history()
        assert len(history) == 2
        assert history[0]["action_type"] == "noop"
        assert history[0]["success"] is True
        assert history[1]["action_type"] == "shell"
        assert history[1]["success"] is True

    def test_history_records_failure(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "unknown_action"})
        history = executor.get_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert "Unknown" in history[0]["error"]

    def test_history_records_approval_rejection(self) -> None:
        executor = ActionExecutor({"dry_run": False, "approval_gate": {"mode": "auto_reject"}})
        executor.execute({"action_type": "file_delete", "path": "/tmp/x"})
        history = executor.get_history()
        assert len(history) == 1
        assert history[0]["success"] is False

    def test_history_includes_timestamp_and_elapsed(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        record = executor.get_history()[0]
        assert "timestamp" in record
        assert "elapsed" in record
        assert isinstance(record["timestamp"], float)
        assert isinstance(record["elapsed"], float)
        assert record["elapsed"] >= 0.0

    def test_history_bounded_by_max_history(self) -> None:
        executor = ActionExecutor({"dry_run": True, "max_history": 3})
        for _ in range(5):
            executor.execute({"action_type": "noop"})
        history = executor.get_history()
        assert len(history) == 3
        # 应保留最近 3 条
        assert all(r["action_type"] == "noop" for r in history)

    def test_history_limit(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        for _ in range(5):
            executor.execute({"action_type": "noop"})
        assert len(executor.get_history(limit=2)) == 2
        assert len(executor.get_history(limit=0)) == 0
        assert len(executor.get_history(limit=10)) == 5
        assert len(executor.get_history(limit=-1)) == 5

    def test_clear_history(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        assert len(executor.get_history()) == 1
        executor.clear_history()
        assert executor.get_history() == []


class TestActionExecutorStatistics:
    """执行统计测试"""

    def test_statistics_empty(self) -> None:
        executor = ActionExecutor({})
        stats = executor.get_statistics()
        assert stats["total"] == 0
        assert stats["success"] == 0
        assert stats["failure"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["by_action_type"] == {}

    def test_statistics_all_success(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "shell", "command": "echo hi"})
        stats = executor.get_statistics()
        assert stats["total"] == 2
        assert stats["success"] == 2
        assert stats["failure"] == 0
        assert stats["success_rate"] == 1.0

    def test_statistics_mixed(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "unknown"})
        stats = executor.get_statistics()
        assert stats["total"] == 2
        assert stats["success"] == 1
        assert stats["failure"] == 1
        assert stats["success_rate"] == 0.5

    def test_statistics_by_action_type(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "shell", "command": "echo hi"})
        executor.execute({"action_type": "unknown"})
        stats = executor.get_statistics()
        by_type = stats["by_action_type"]
        assert by_type["noop"]["total"] == 2
        assert by_type["noop"]["success"] == 2
        assert by_type["noop"]["failure"] == 0
        assert by_type["shell"]["total"] == 1
        assert by_type["shell"]["success"] == 1
        assert by_type["unknown"]["total"] == 1
        assert by_type["unknown"]["failure"] == 1

    def test_get_state_includes_history_stats(self) -> None:
        executor = ActionExecutor({"dry_run": True, "max_history": 50})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "unknown"})
        state = executor.get_state()
        assert state["history_size"] == 2
        assert state["max_history"] == 50
        assert state["total_executions"] == 2
        assert state["success_count"] == 1
        assert state["failure_count"] == 1
        assert state["success_rate"] == 0.5

    def test_reset_clears_history_and_stats(self) -> None:
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        executor.execute({"action_type": "unknown"})
        assert executor.get_statistics()["total"] == 2
        executor.reset()
        assert executor.get_history() == []
        assert executor.get_statistics()["total"] == 0
        # 配置项应保留
        state = executor.get_state()
        assert state["history_size"] == 0
        assert state["total_executions"] == 0


class TestErrorHandlingHardened:
    """D1: 错误处理加固测试"""

    def test_health_check_logs_warning_empty_prefixes(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        executor = ActionExecutor({"allowed_shell_prefixes": []})
        with caplog.at_level(logging.WARNING):
            result = executor.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)

    def test_health_check_logs_warning_invalid_timeout(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        executor = ActionExecutor({"timeout": 0})
        with caplog.at_level(logging.WARNING):
            result = executor.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)

    def test_health_check_logs_warning_negative_max_history(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        executor = ActionExecutor({"max_history": -1})
        with caplog.at_level(logging.WARNING):
            result = executor.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)


class TestEdgeCases:
    """D2: 边界条件与异常输入测试"""

    def test_execute_empty_payload(self) -> None:
        """空 payload 默认 action_type=noop"""
        executor = ActionExecutor({"dry_run": True})
        result = executor.execute({})
        assert result.success is True
        assert result.action_type == "noop"

    def test_shell_whitespace_command(self) -> None:
        """纯空白命令失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "shell", "command": "   "})
        assert result.success is False

    def test_http_empty_url(self) -> None:
        """空 URL 失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "http_get", "url": ""})
        assert result.success is False

    def test_http_post_empty_url(self) -> None:
        """空 URL 失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "http_post", "url": ""})
        assert result.success is False

    def test_file_write_empty_path(self) -> None:
        """空路径失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "file_write", "path": "", "content": "x"})
        assert result.success is False

    def test_file_read_empty_path(self) -> None:
        """空路径失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "file_read", "path": ""})
        assert result.success is False

    def test_file_delete_empty_path(self) -> None:
        """空路径失败"""
        executor = ActionExecutor({"dry_run": False})
        result = executor.execute({"action_type": "file_delete", "path": ""})
        assert result.success is False

    def test_history_limit_exceeds_size(self) -> None:
        """limit 超过历史大小时返回全部"""
        executor = ActionExecutor({"dry_run": True})
        for _ in range(3):
            executor.execute({"action_type": "noop"})
        assert len(executor.get_history(limit=100)) == 3

    def test_statistics_all_failures(self) -> None:
        """全部失败的统计"""
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "unknown1"})
        executor.execute({"action_type": "unknown2"})
        stats = executor.get_statistics()
        assert stats["total"] == 2
        assert stats["success"] == 0
        assert stats["failure"] == 2
        assert stats["success_rate"] == 0.0

    def test_history_records_elapsed_positive(self) -> None:
        """elapsed 应为非负"""
        executor = ActionExecutor({"dry_run": True})
        executor.execute({"action_type": "noop"})
        record = executor.get_history()[0]
        assert record["elapsed"] >= 0.0

    def test_max_history_zero(self) -> None:
        """max_history=0 边界"""
        executor = ActionExecutor({"dry_run": True, "max_history": 0})
        executor.execute({"action_type": "noop"})
        # max_history=0 意味着不保留历史
        assert len(executor.get_history()) <= 0

    def test_reset_preserves_config_after_multiple_executions(self) -> None:
        """多次执行后 reset 仍保留配置"""
        executor = ActionExecutor({"dry_run": False, "timeout": 15.0})
        for _ in range(5):
            executor.execute({"action_type": "noop"})
        executor.reset()
        state = executor.get_state()
        assert state["dry_run"] is False
        assert state["timeout"] == 15.0
        assert state["history_size"] == 0


class TestStateConsistency:
    """D5: 状态一致性与幂等性测试"""

    def test_reset_restores_initial_state(self) -> None:
        """reset 后 history 清空、统计归零。"""
        executor = ActionExecutor({})
        for _ in range(3):
            executor.execute({"action_type": "noop"})
        assert len(executor.history) > 0
        stats = executor.get_statistics()
        assert stats["total"] > 0
        # reset
        executor.reset()
        assert len(executor.history) == 0
        stats_after = executor.get_statistics()
        assert stats_after["total"] == 0
        assert stats_after["success"] == 0
        assert stats_after["failure"] == 0

    def test_reset_state_equals_fresh_instance(self) -> None:
        """reset 后 get_state 等价于新构造实例。"""
        config = {"dry_run": True, "timeout": 10.0, "max_history": 50}
        mod1 = ActionExecutor(config)
        mod2 = ActionExecutor(config)
        mod1.execute({"action_type": "noop"})
        mod1.execute({"action_type": "shell", "command": "echo hi"})
        mod1.reset()
        assert mod1.get_state() == mod2.get_state()

    def test_reset_idempotent(self) -> None:
        """连续调用 reset 两次，状态一致。"""
        executor = ActionExecutor({})
        executor.execute({"action_type": "noop"})
        executor.reset()
        state1 = executor.get_state()
        executor.reset()
        state2 = executor.get_state()
        assert state1 == state2

    def test_health_check_idempotent(self) -> None:
        """连续调用 health_check 多次，结果一致且无副作用。"""
        executor = ActionExecutor({})
        executor.execute({"action_type": "noop"})
        state_before = executor.get_state()
        r1 = executor.health_check()
        r2 = executor.health_check()
        r3 = executor.health_check()
        state_after = executor.get_state()
        assert r1 == r2 == r3 is True
        assert state_before == state_after

    def test_get_state_idempotent(self) -> None:
        """连续调用 get_state 多次，结果一致且不改变内部状态。"""
        executor = ActionExecutor({})
        executor.execute({"action_type": "noop"})
        s1 = executor.get_state()
        s2 = executor.get_state()
        s3 = executor.get_state()
        assert s1 == s2 == s3
