"""决策模块单元测试"""
import pytest
from root.left.app.left.src.core_mod.decide import (
    ExplorationController, TaskSelector, ResourceBalancer, DecisionModule,
)
from root.left.app.left.src.core_mod.task import TaskPoolModule, Task
from root.left.app.left.src.utils_main.constants import TaskType, DriveType
from root.left.app.left.src.utils_main.exceptions import ValidationError, DecisionException


class TestExplorationController:
    """测试ExplorationController"""

    def test_normal_should_explore(self) -> None:
        ctrl = ExplorationController({})
        drive = {"scenario": "normal"}
        result = ctrl.should_explore(drive)
        assert isinstance(result, bool)

    def test_empty_drive(self) -> None:
        ctrl = ExplorationController({})
        result = ctrl.should_explore({})
        assert isinstance(result, bool)

    def test_boundary_emergency(self) -> None:
        ctrl = ExplorationController({})
        drive = {"scenario": "emergency"}
        assert ctrl.should_explore(drive) is False

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            ExplorationController(None)

    def test_record_choice(self) -> None:
        ctrl = ExplorationController({})
        ctrl.record_choice(True)
        assert ctrl.exploration_count == 1

    def test_current_ratio(self) -> None:
        ctrl = ExplorationController({"base_exploration_ratio": 0.3})
        assert ctrl._get_current_ratio() == 0.3


class TestTaskSelector:
    """测试TaskSelector"""

    def test_normal_select(self) -> None:
        sel = TaskSelector({})
        tasks = [Task(TaskType.SAFE, 0.5, {}), Task(TaskType.GAIN, 0.8, {})]
        drive = {"gaps": {TaskType.SAFE: 0.5, TaskType.GAIN: 0.3}}
        result = sel.select(tasks, drive)
        assert result is not None

    def test_empty_tasks(self) -> None:
        sel = TaskSelector({})
        assert sel.select([], {}) is None

    def test_boundary_single_task(self) -> None:
        sel = TaskSelector({})
        tasks = [Task(TaskType.SAFE, 0.5, {})]
        result = sel.select(tasks, {})
        assert result is not None

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            TaskSelector(None)


class TestResourceBalancer:
    """测试ResourceBalancer"""

    def test_normal_within_limits(self) -> None:
        bal = ResourceBalancer({"cpu_limit": 20.0})
        assert bal.check_limits(10.0, 10.0) is True

    def test_empty_config(self) -> None:
        bal = ResourceBalancer({})
        assert bal.check_limits(0.0, 0.0) is True

    def test_boundary_at_limit(self) -> None:
        bal = ResourceBalancer({"cpu_limit": 20.0})
        assert bal.check_limits(20.0, 20.0) is True

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            ResourceBalancer(None)

    def test_available_capacity(self) -> None:
        bal = ResourceBalancer({"cpu_limit": 20.0})
        cap = bal.get_available_capacity(10.0, 10.0)
        assert cap["cpu"] == 10.0


class TestDecisionModule:
    """测试DecisionModule"""

    def test_normal_decide(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        result = mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        assert "selected_task" in result

    def test_empty_pool(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        result = mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        assert result["selected_task"] is None

    def test_boundary_resource_exceeded(self) -> None:
        mod = DecisionModule({"resource_balancer": {"cpu_limit": 10.0}})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        with pytest.raises(DecisionException):
            mod.decide(pool_mod.pool, drive, [], 1, 20.0, 0.0)

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            DecisionModule(None)

    def test_should_pause(self) -> None:
        mod = DecisionModule({})
        drive = {"gaps": {DriveType.SAFE: 0.8}}
        assert mod.should_pause(drive) is True

    def test_should_not_pause(self) -> None:
        mod = DecisionModule({})
        drive = {"gaps": {DriveType.SAFE: 0.5}}
        assert mod.should_pause(drive) is False


class TestDecisionSummary:
    """决策摘要测试"""

    def test_summary_empty(self) -> None:
        mod = DecisionModule({})
        summary = mod.get_decision_summary()
        assert summary["total"] == 0
        assert summary["exploration"] == 0
        assert summary["exploitation"] == 0
        assert summary["exploration_ratio"] == 0.0
        assert summary["task_selected"] == 0
        assert summary["task_none"] == 0
        assert summary["task_selection_rate"] == 0.0
        assert summary["avg_available_cpu"] == 0.0
        assert summary["avg_available_memory"] == 0.0

    def test_summary_with_decisions(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        # 执行多次决策
        for i in range(5):
            mod.decide(pool_mod.pool, drive, [], i, 5.0, 5.0)
        summary = mod.get_decision_summary()
        assert summary["total"] == 5
        assert summary["exploration"] + summary["exploitation"] == 5
        assert 0.0 <= summary["exploration_ratio"] <= 1.0
        # 每次都有任务可选
        assert summary["task_selected"] == 5
        assert summary["task_none"] == 0
        assert summary["task_selection_rate"] == 1.0
        # 资源限制默认 20，使用 5，可用 15
        assert summary["avg_available_cpu"] == 15.0
        assert summary["avg_available_memory"] == 15.0

    def test_summary_task_none(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        # 空池子，每次都选不到任务
        for i in range(3):
            mod.decide(pool_mod.pool, drive, [], i, 0.0, 0.0)
        summary = mod.get_decision_summary()
        assert summary["total"] == 3
        assert summary["task_selected"] == 0
        assert summary["task_none"] == 3
        assert summary["task_selection_rate"] == 0.0

    def test_summary_exploration_ratio(self) -> None:
        mod = DecisionModule({})
        # 手动构造 decision_history 以控制探索比例
        mod.decision_history = [
            {"loop": 1, "is_exploration": True, "selected_task": {"id": 1}, "resource_capacity": {"cpu": 10.0, "memory": 10.0}},
            {"loop": 2, "is_exploration": False, "selected_task": {"id": 2}, "resource_capacity": {"cpu": 10.0, "memory": 10.0}},
            {"loop": 3, "is_exploration": True, "selected_task": None, "resource_capacity": {"cpu": 5.0, "memory": 5.0}},
        ]
        summary = mod.get_decision_summary()
        assert summary["total"] == 3
        assert summary["exploration"] == 2
        assert summary["exploitation"] == 1
        assert abs(summary["exploration_ratio"] - 2 / 3) < 1e-9
        assert summary["task_selected"] == 2
        assert summary["task_none"] == 1
        assert abs(summary["task_selection_rate"] - 2 / 3) < 1e-9
        # (10 + 10 + 5) / 3 = 25/3
        assert abs(summary["avg_available_cpu"] - 25 / 3) < 1e-9


class TestDecisionModuleUnifiedInterface:
    """统一接口增强测试"""

    def test_get_state_includes_summary_info(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod.decide(pool_mod.pool, drive, [], 1, 5.0, 5.0)
        state = mod.get_state()
        assert state["decision_history_len"] == 1
        assert "exploration_count" in state
        assert "exploitation_count" in state
        assert state["exploration_count"] + state["exploitation_count"] == 1
        assert "task_selection_rate" in state
        assert state["task_selection_rate"] == 1.0

    def test_reset_clears_history_and_counts(self) -> None:
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        assert len(mod.decision_history) == 1
        assert mod.exploration.exploration_count + mod.exploration.exploitation_count == 1
        mod.reset()
        assert len(mod.decision_history) == 0
        assert mod.exploration.exploration_count == 0
        assert mod.exploration.exploitation_count == 0

    def test_health_check_normal(self) -> None:
        mod = DecisionModule({})
        assert mod.health_check() is True

    def test_health_check_invalid_max_history(self) -> None:
        mod = DecisionModule({"max_history": -1})
        assert mod.health_check() is False


class TestErrorHandlingHardened:
    """D1: 错误处理加固测试"""

    def test_health_check_logs_warning(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        mod = DecisionModule({"max_history": -1})
        with caplog.at_level(logging.WARNING):
            result = mod.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)


class TestEdgeCases:
    """D2: 边界条件与异常输入测试"""

    def test_decide_empty_performance_history(self) -> None:
        """空 performance_history"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {}}
        result = mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        assert "selected_task" in result

    def test_decide_large_loop_count(self) -> None:
        """极大 loop_count"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        result = mod.decide(pool_mod.pool, drive, [], 999999, 0.0, 0.0)
        assert result["loop"] == 999999

    def test_decide_at_resource_limit(self) -> None:
        """恰好等于资源限制边界"""
        mod = DecisionModule({"resource_balancer": {"cpu_limit": 20.0, "memory_limit": 20.0}})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        # 恰好等于限制，不应抛异常
        result = mod.decide(pool_mod.pool, drive, [], 1, 20.0, 20.0)
        assert "selected_task" in result

    def test_should_pause_empty_gaps(self) -> None:
        """空 gaps 不暂停"""
        mod = DecisionModule({})
        assert mod.should_pause({}) is False

    def test_should_pause_missing_safe_key(self) -> None:
        """gaps 中无 SAFE 键不暂停"""
        mod = DecisionModule({})
        drive = {"gaps": {TaskType.GAIN: 0.9}}
        assert mod.should_pause(drive) is False

    def test_should_pause_boundary_07(self) -> None:
        """safe_gap 恰好 0.7 不暂停（>0.7 才暂停）"""
        mod = DecisionModule({})
        drive = {"gaps": {DriveType.SAFE: 0.7}}
        assert mod.should_pause(drive) is False

    def test_should_pause_above_07(self) -> None:
        """safe_gap > 0.7 暂停"""
        mod = DecisionModule({})
        drive = {"gaps": {DriveType.SAFE: 0.71}}
        assert mod.should_pause(drive) is True

    def test_exploration_emergency_always_false(self) -> None:
        """emergency 场景永不探索"""
        ctrl = ExplorationController({})
        for _ in range(10):
            assert ctrl.should_explore({"scenario": "emergency"}) is False

    def test_exploration_ratio_bounds(self) -> None:
        """探索比例在 min/max 之间调节"""
        ctrl = ExplorationController({
            "base_exploration_ratio": 0.3,
            "min_exploration_ratio": 0.1,
            "max_exploration_ratio": 0.5,
        })
        # 全部探索后 actual=1.0 > max=0.5，应降低 ratio
        for _ in range(10):
            ctrl.record_choice(True)
        ratio = ctrl._get_current_ratio()
        assert ratio < 0.3  # base - 0.1 = 0.2

    def test_task_selector_empty_gaps(self) -> None:
        """空 gaps 使用默认 gap=0.5"""
        sel = TaskSelector({})
        tasks = [Task(TaskType.SAFE, 0.5, {})]
        result = sel.select(tasks, {})
        assert result is not None

    def test_decision_history_bounded(self) -> None:
        """决策历史有界"""
        mod = DecisionModule({"max_history": 3})
        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        for i in range(10):
            mod.decide(pool_mod.pool, drive, [], i, 0.0, 0.0)
        assert len(mod.decision_history) == 3

    def test_summary_all_exploration(self) -> None:
        """全部探索的摘要"""
        mod = DecisionModule({})
        mod.decision_history = [
            {"loop": i, "is_exploration": True, "selected_task": None, "resource_capacity": {"cpu": 10.0, "memory": 10.0}}
            for i in range(5)
        ]
        summary = mod.get_decision_summary()
        assert summary["exploration"] == 5
        assert summary["exploitation"] == 0
        assert summary["exploration_ratio"] == 1.0

    def test_summary_all_exploitation(self) -> None:
        """全部利用的摘要"""
        mod = DecisionModule({})
        mod.decision_history = [
            {"loop": i, "is_exploration": False, "selected_task": {"id": i}, "resource_capacity": {"cpu": 10.0, "memory": 10.0}}
            for i in range(5)
        ]
        summary = mod.get_decision_summary()
        assert summary["exploration"] == 0
        assert summary["exploitation"] == 5
        assert summary["exploration_ratio"] == 0.0


class TestConfigValidation:
    """D4: 构造时配置验证测试"""

    def test_exploration_base_ratio_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be in \\[0, 1\\]"):
            ExplorationController({"base_exploration_ratio": 1.5})

    def test_exploration_negative_ratio_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be in \\[0, 1\\]"):
            ExplorationController({"base_exploration_ratio": -0.1})

    def test_exploration_min_gt_base_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_exploration_ratio.*>.*base_exploration_ratio"):
            ExplorationController({
                "base_exploration_ratio": 0.2,
                "min_exploration_ratio": 0.3,
            })

    def test_exploration_base_gt_max_raises(self) -> None:
        with pytest.raises(ValidationError, match="base_exploration_ratio.*>.*max_exploration_ratio"):
            ExplorationController({
                "base_exploration_ratio": 0.6,
                "max_exploration_ratio": 0.5,
            })

    def test_exploration_valid_boundary_zero(self) -> None:
        """比率为 0 是合法的"""
        ctrl = ExplorationController({
            "base_exploration_ratio": 0.0,
            "min_exploration_ratio": 0.0,
            "max_exploration_ratio": 0.5,
        })
        assert ctrl.base_ratio == 0.0

    def test_exploration_valid_boundary_one(self) -> None:
        """比率为 1 是合法的"""
        ctrl = ExplorationController({
            "base_exploration_ratio": 1.0,
            "min_exploration_ratio": 0.5,
            "max_exploration_ratio": 1.0,
        })
        assert ctrl.base_ratio == 1.0

    def test_resource_balancer_zero_cpu_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="cpu_limit must be positive"):
            ResourceBalancer({"cpu_limit": 0.0})

    def test_resource_balancer_negative_cpu_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="cpu_limit must be positive"):
            ResourceBalancer({"cpu_limit": -1.0})

    def test_resource_balancer_zero_memory_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="memory_limit must be positive"):
            ResourceBalancer({"memory_limit": 0.0})

    def test_resource_balancer_valid_custom_limits(self) -> None:
        """自定义合法限制"""
        bal = ResourceBalancer({"cpu_limit": 50.0, "memory_limit": 100.0})
        assert bal.cpu_limit == 50.0
        assert bal.memory_limit == 100.0


class TestStateConsistency:
    """D5: 状态一致性与幂等性测试"""

    def test_reset_restores_initial_state(self) -> None:
        """reset 后 decision_history 清空、exploration 计数归零。"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        # 执行若干次决策产生状态
        for i in range(5):
            mod.decide(pool_mod.pool, drive, [], i, 0.0, 0.0)
        assert len(mod.decision_history) > 0
        assert mod.exploration.exploration_count + mod.exploration.exploitation_count > 0
        # reset
        mod.reset()
        assert len(mod.decision_history) == 0
        assert mod.exploration.exploration_count == 0
        assert mod.exploration.exploitation_count == 0

    def test_reset_state_equals_fresh_instance(self) -> None:
        """reset 后 get_state 等价于新构造实例。"""
        config = {"max_history": 50}
        mod1 = DecisionModule(config)
        mod2 = DecisionModule(config)
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod1.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        mod1.reset()
        # reset 后 mod1 的状态字段应与 mod2 一致
        assert mod1.get_state() == mod2.get_state()

    def test_reset_idempotent(self) -> None:
        """连续调用 reset 两次，状态一致。"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        mod.reset()
        state1 = mod.get_state()
        mod.reset()
        state2 = mod.get_state()
        assert state1 == state2

    def test_health_check_idempotent(self) -> None:
        """连续调用 health_check 多次，结果一致且无副作用。"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        state_before = mod.get_state()
        r1 = mod.health_check()
        r2 = mod.health_check()
        r3 = mod.health_check()
        state_after = mod.get_state()
        assert r1 == r2 == r3 is True
        assert state_before == state_after

    def test_get_state_idempotent(self) -> None:
        """连续调用 get_state 多次，结果一致且不改变内部状态。"""
        mod = DecisionModule({})
        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {}))
        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        mod.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)
        s1 = mod.get_state()
        s2 = mod.get_state()
        s3 = mod.get_state()
        assert s1 == s2 == s3
