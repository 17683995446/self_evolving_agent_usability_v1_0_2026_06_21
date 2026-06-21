"""D3: 模块间集成测试

验证核心模块流水线的端到端正确性：
1. decide → action_executor → effect_verifier：决策→执行→效果验证
2. learn → verify：学习→安全验证
3. action_executor → effect_verifier：执行→效果采集
4. decide → action_executor：决策→执行历史对齐
"""
import pytest

from root.left.app.left.src.core_mod.action_executor import ActionExecutor
from root.left.app.left.src.core_mod.decide import DecisionModule
from root.left.app.left.src.core_mod.effect_verifier import EffectVerifier
from root.left.app.left.src.core_mod.learn import AdaptiveLearner
from root.left.app.left.src.core_mod.task import Task, TaskPoolModule
from root.left.app.left.src.core_mod.verify import VerificationModule
from root.left.app.left.src.utils_main.constants import TaskType


class TestDecideActionEffectPipeline:
    """decide → action_executor → effect_verifier 流水线测试"""

    def test_full_pipeline_dry_run(self) -> None:
        """完整流水线：决策→执行(dry_run)→效果采集"""
        # 初始化三个模块
        decider = DecisionModule({})
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({})

        # 准备任务池
        pool_mod = TaskPoolModule({})
        task = Task(TaskType.SAFE, 0.8, {"action_type": "shell", "command": "echo hello"})
        pool_mod.pool.add(task)

        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}

        # 1. 决策
        decision = decider.decide(pool_mod.pool, drive, [], 1, 5.0, 5.0)
        assert decision["selected_task"] is not None

        # 2. 执行
        payload = decision["selected_task"]["payload"]
        # 采集执行前指标
        verifier.capture_before("cpu", 50.0)
        result = executor.execute(payload)
        assert result.success is True
        # 采集执行后指标
        effect = verifier.capture_after("cpu", 45.0)
        assert effect["improved"] is True

    def test_pipeline_multiple_iterations(self) -> None:
        """多次迭代流水线"""
        decider = DecisionModule({})
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({"max_points": 20})

        pool_mod = TaskPoolModule({})
        for _ in range(5):
            pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {"action_type": "noop"}))

        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.3}}

        for i in range(5):
            decision = decider.decide(pool_mod.pool, drive, [], i, 5.0, 5.0)
            if decision["selected_task"]:
                payload = decision["selected_task"]["payload"]
                verifier.capture_before("latency", 100.0 + i)
                executor.execute(payload)
                verifier.capture_after("latency", 95.0 + i)

        # 验证执行历史
        assert len(executor.get_history()) == 5
        stats = executor.get_statistics()
        assert stats["success"] == 5

        # 验证效果采集
        details = verifier.get_detailed_summary()
        assert "latency" in details
        assert details["latency"]["count"] == 10  # 5 before + 5 after

    def test_pipeline_with_failed_action(self) -> None:
        """执行失败时效果仍被采集"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({})

        verifier.capture_before("error_rate", 0.1)
        result = executor.execute({"action_type": "unknown_action"})
        assert result.success is False
        effect = verifier.capture_after("error_rate", 0.15)
        assert effect["improved"] is False  # error_rate 上升 = 未改善

    def test_pipeline_decision_summary_after_execution(self) -> None:
        """决策摘要与执行统计对齐"""
        decider = DecisionModule({})
        executor = ActionExecutor({"dry_run": True})

        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {"action_type": "noop"}))
        drive = {"scenario": "normal", "gaps": {}}

        for i in range(3):
            decider.decide(pool_mod.pool, drive, [], i, 0.0, 0.0)
            executor.execute({"action_type": "noop"})

        decision_summary = decider.get_decision_summary()
        exec_stats = executor.get_statistics()

        assert decision_summary["total"] == 3
        assert exec_stats["total"] == 3
        assert exec_stats["success"] == 3


class TestLearnVerifyIntegration:
    """learn → verify 集成测试"""

    def test_learn_then_verify_health(self) -> None:
        """学习后验证模块健康"""
        learner = AdaptiveLearner({})
        verifier = VerificationModule({})

        # 注册安全不变式：权重必须有限
        def weights_finite(state):
            w = state.get("weights", [])
            if not w:
                return True
            return all(isinstance(x, (int, float)) for x in w)

        verifier.register_safety("weights_finite", weights_finite)

        # 学习
        for _ in range(10):
            learner.learn([1.0, 2.0], 5.0)

        # 验证健康
        assert learner.health_check() is True
        assert verifier.health_check() is True

        # 验证安全不变式
        state = {"weights": learner.sgd.get_weights()["weights"]}
        result = verifier.check_safety(state)
        assert result["safe"] is True

    def test_learn_verify_after_reset(self) -> None:
        """学习后 reset，验证状态归零"""
        learner = AdaptiveLearner({})
        verifier = VerificationModule({})

        learner.learn([1.0], 2.0)
        verifier.trace_decision("learn", {}, "done", "learning step")

        assert learner.sgd.n_samples == 1
        assert len(verifier.tracer.traces) == 1

        learner.reset()
        verifier.reset()

        assert learner.sgd.n_samples == 0
        assert len(verifier.tracer.traces) == 0
        assert learner.health_check() is True
        assert verifier.health_check() is True

    def test_learn_overfit_detection_with_verify(self) -> None:
        """过拟合检测 + 验证模块记录"""
        learner = AdaptiveLearner({"overfit": {"threshold": 0.01}})
        verifier = VerificationModule({})

        # 制造过拟合：训练误差低，验证误差高
        for _ in range(15):
            learner.learn([1.0], 2.0, validation_target=10.0)

        # 检测过拟合
        is_overfit = learner.check_overfitting()
        assert is_overfit is True

        # 记录到验证模块的 tracer
        verifier.trace_decision(
            "overfit_detection",
            {"is_overfitting": is_overfit, "gap": learner.overfit.get_gap()},
            "alert",
            "overfitting detected, gap exceeds threshold",
        )

        explanation = verifier.explain_decision()
        assert "overfit_detection" in explanation


class TestActionEffectIntegration:
    """action_executor → effect_verifier 集成测试"""

    def test_execute_and_measure_improvement(self) -> None:
        """执行动作后测量指标改善"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({})

        # 采集基线
        verifier.capture_before("cpu", 80.0)
        # 执行动作
        executor.execute({"action_type": "shell", "command": "echo optimize"})
        # 采集改善后指标
        effect = verifier.capture_after("cpu", 60.0)

        assert effect["delta"] == -20.0
        assert effect["improved"] is True  # cpu 下降 = 改善

    def test_execute_and_measure_regression(self) -> None:
        """执行动作后指标恶化"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({})

        verifier.capture_before("throughput", 100.0)
        executor.execute({"action_type": "noop"})
        effect = verifier.capture_after("throughput", 80.0)

        assert effect["delta"] == -20.0
        assert effect["improved"] is False  # throughput 下降 = 恶化

    def test_threshold_violation_after_execution(self) -> None:
        """执行后阈值违规检测"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({"thresholds": {"cpu": {"max": 70.0}}})

        verifier.capture_before("cpu", 60.0)
        executor.execute({"action_type": "noop"})
        verifier.capture_after("cpu", 80.0)  # 超过阈值

        violations = verifier.get_threshold_violations()
        assert len(violations) == 1
        assert violations[0]["metric"] == "cpu"
        assert violations[0]["value"] == 80.0

    def test_significance_after_multiple_executions(self) -> None:
        """多次执行后显著性检测"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({"min_samples_for_significance": 3})

        # 建立基线
        for val in [50.0, 50.0, 50.0]:
            verifier.capture_before("latency", val)
            executor.execute({"action_type": "noop"})

        # 异常值
        verifier.capture_before("latency", 50.0)
        executor.execute({"action_type": "noop"})
        assert verifier.is_significant("latency", 100.0) is True

    def test_trend_analysis_after_execution_series(self) -> None:
        """执行序列后趋势分析"""
        executor = ActionExecutor({"dry_run": True})
        verifier = EffectVerifier({})

        # 逐渐改善的趋势
        for cpu in [80.0, 75.0, 70.0, 65.0, 60.0]:
            verifier.capture_before("cpu", cpu)
            executor.execute({"action_type": "noop"})

        slope, direction = verifier.get_trend("cpu")
        assert slope < 0  # 下降趋势
        assert direction == "down"  # cpu 下降 = 改善趋势


class TestDecideActionAlignment:
    """decide → action_executor 对齐测试"""

    def test_decision_task_matches_executed_payload(self) -> None:
        """决策选中的任务 payload 与执行的 payload 一致"""
        decider = DecisionModule({})
        executor = ActionExecutor({"dry_run": True})

        pool_mod = TaskPoolModule({})
        expected_payload = {"action_type": "shell", "command": "echo test"}
        task = Task(TaskType.SAFE, 0.9, expected_payload)
        pool_mod.pool.add(task)

        drive = {"scenario": "normal", "gaps": {TaskType.SAFE: 0.5}}
        decision = decider.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)

        assert decision["selected_task"] is not None
        actual_payload = decision["selected_task"]["payload"]
        assert actual_payload == expected_payload

        result = executor.execute(actual_payload)
        assert result.success is True

    def test_empty_pool_decision_no_execution(self) -> None:
        """空任务池决策时不执行"""
        decider = DecisionModule({})
        executor = ActionExecutor({"dry_run": True})

        pool_mod = TaskPoolModule({})
        drive = {"scenario": "normal", "gaps": {}}
        decision = decider.decide(pool_mod.pool, drive, [], 1, 0.0, 0.0)

        assert decision["selected_task"] is None
        # 不执行任何动作
        assert len(executor.get_history()) == 0

    def test_resource_limit_prevents_execution(self) -> None:
        """资源超限时决策抛异常，不执行"""
        from root.left.app.left.src.utils_main.exceptions import DecisionException

        decider = DecisionModule({"resource_balancer": {"cpu_limit": 10.0}})
        executor = ActionExecutor({"dry_run": True})

        pool_mod = TaskPoolModule({})
        pool_mod.pool.add(Task(TaskType.SAFE, 0.5, {"action_type": "noop"}))
        drive = {"scenario": "normal", "gaps": {}}

        with pytest.raises(DecisionException):
            decider.decide(pool_mod.pool, drive, [], 1, 50.0, 0.0)

        # 执行器历史仍为空
        assert len(executor.get_history()) == 0
