"""效果验证器测试"""

import pytest

from root.left.app.left.src.core_mod.effect_verifier import EffectVerifier


class TestEffectVerifier:
    """效果验证器测试"""

    def test_capture_before_after(self) -> None:
        v = EffectVerifier({})
        v.capture_before("cpu", 80.0)
        result = v.capture_after("cpu", 60.0)
        assert result["before"] == 80.0
        assert result["after"] == 60.0
        assert result["improved"] is True

    def test_no_before(self) -> None:
        v = EffectVerifier({})
        result = v.capture_after("memory", 50.0)
        assert result["before"] is None
        assert result["improved"] is True

    def test_higher_better(self) -> None:
        v = EffectVerifier({})
        v.capture_before("accuracy", 0.7)
        result = v.capture_after("accuracy", 0.8)
        assert result["improved"] is True

    def test_summary(self) -> None:
        v = EffectVerifier({})
        v.capture_before("cpu", 50.0)
        summary = v.get_summary()
        assert "cpu" in summary["metrics"]
        assert summary["latest"]["cpu"] == 50.0

    def test_max_points(self) -> None:
        v = EffectVerifier({"max_points": 2})
        v.capture_before("x", 1.0)
        v.capture_before("x", 2.0)
        v.capture_before("x", 3.0)
        result = v.capture_after("x", 4.0)
        assert result["before"] == 2.0

    def test_capture_after_includes_new_fields(self) -> None:
        """capture_after 结果应包含新增的 significant/threshold_ok/trend 字段"""
        v = EffectVerifier({})
        v.capture_before("cpu", 80.0)
        result = v.capture_after("cpu", 60.0)
        assert "significant" in result
        assert "threshold_ok" in result
        assert "trend_slope" in result
        assert "trend_direction" in result


class TestThreshold:
    """阈值检查测试"""

    def test_set_threshold(self) -> None:
        v = EffectVerifier({})
        v.set_threshold("cpu", min_val=0.0, max_val=90.0)
        assert "cpu" in v.thresholds
        assert v.thresholds["cpu"]["min"] == 0.0
        assert v.thresholds["cpu"]["max"] == 90.0

    def test_set_threshold_partial(self) -> None:
        v = EffectVerifier({})
        v.set_threshold("cpu", max_val=90.0)
        assert "min" not in v.thresholds["cpu"]
        assert v.thresholds["cpu"]["max"] == 90.0

    def test_check_threshold_in_range(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"min": 0.0, "max": 90.0}}})
        assert v.check_threshold("cpu", 50.0) is True

    def test_check_threshold_below_min(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"min": 10.0, "max": 90.0}}})
        assert v.check_threshold("cpu", 5.0) is False

    def test_check_threshold_above_max(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"min": 10.0, "max": 90.0}}})
        assert v.check_threshold("cpu", 95.0) is False

    def test_check_threshold_no_config(self) -> None:
        v = EffectVerifier({})
        assert v.check_threshold("unknown", 50.0) is True

    def test_get_threshold_violations_empty(self) -> None:
        v = EffectVerifier({})
        assert v.get_threshold_violations() == []

    def test_get_threshold_violations_detected(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 90.0)
        violations = v.get_threshold_violations()
        assert len(violations) == 1
        assert violations[0]["metric"] == "cpu"
        assert violations[0]["value"] == 90.0
        assert violations[0]["max"] == 80.0

    def test_get_threshold_violations_no_violation(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        assert v.get_threshold_violations() == []


class TestSignificance:
    """统计显著性测试"""

    def test_insufficient_samples(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": 3})
        v.capture_before("x", 1.0)
        v.capture_before("x", 1.0)
        assert v.is_significant("x", 10.0) is False

    def test_significant_change(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": 3, "significance_sigma": 1.0})
        for val in [10.0, 10.0, 10.0, 10.0]:
            v.capture_before("x", val)
        # 历史值全为 10，新值 20 应显著
        assert v.is_significant("x", 20.0) is True

    def test_not_significant(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": 3, "significance_sigma": 3.0})
        for val in [10.0, 11.0, 10.0, 11.0]:
            v.capture_before("x", val)
        # 均值约 10.5，std 约 0.58，3*std 约 1.73，新值 11 在范围内
        assert v.is_significant("x", 11.0) is False

    def test_significant_with_variance(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": 3, "significance_sigma": 1.0})
        for val in [10.0, 10.0, 10.0, 10.0]:
            v.capture_before("x", val)
        # std=0，任何变化都显著
        assert v.is_significant("x", 10.0001) is True

    def test_not_significant_same_value(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": 3})
        for val in [10.0, 10.0, 10.0]:
            v.capture_before("x", val)
        # 相同值不显著
        assert v.is_significant("x", 10.0) is False


class TestTrend:
    """历史趋势测试"""

    def test_trend_unknown_insufficient(self) -> None:
        v = EffectVerifier({})
        v.capture_before("x", 1.0)
        slope, direction = v.get_trend("x")
        assert direction == "unknown"

    def test_trend_up(self) -> None:
        v = EffectVerifier({})
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            v.capture_before("x", val)
        slope, direction = v.get_trend("x")
        assert slope > 0
        assert direction == "up"

    def test_trend_down(self) -> None:
        v = EffectVerifier({})
        for val in [5.0, 4.0, 3.0, 2.0, 1.0]:
            v.capture_before("x", val)
        slope, direction = v.get_trend("x")
        assert slope < 0
        assert direction == "down"

    def test_trend_stable(self) -> None:
        v = EffectVerifier({})
        for val in [100.0, 100.0, 100.0, 100.0]:
            v.capture_before("x", val)
        slope, direction = v.get_trend("x")
        assert direction == "stable"

    def test_change_rate_empty(self) -> None:
        v = EffectVerifier({})
        assert v.get_change_rate("x") == 0.0

    def test_change_rate_normal(self) -> None:
        v = EffectVerifier({})
        v.capture_before("x", 100.0)
        v.capture_before("x", 100.0)
        v.capture_before("x", 110.0)
        # 前两个均值 100，最新 110，变化率 10%
        rate = v.get_change_rate("x")
        assert abs(rate - 10.0) < 0.01

    def test_change_rate_zero_mean(self) -> None:
        v = EffectVerifier({})
        v.capture_before("x", 0.0)
        v.capture_before("x", 0.0)
        v.capture_before("x", 0.0)
        assert v.get_change_rate("x") == 0.0


class TestDetailedSummary:
    """详细汇总测试"""

    def test_empty(self) -> None:
        v = EffectVerifier({})
        assert v.get_detailed_summary() == {}

    def test_with_metrics(self) -> None:
        v = EffectVerifier({})
        for val in [1.0, 2.0, 3.0]:
            v.capture_before("x", val)
        details = v.get_detailed_summary()
        assert "x" in details
        assert details["x"]["count"] == 3
        assert details["x"]["latest"] == 3.0
        assert "mean" in details["x"]
        assert "std" in details["x"]
        assert "trend_slope" in details["x"]
        assert "trend_direction" in details["x"]
        assert "change_rate_pct" in details["x"]
        assert "threshold_ok" in details["x"]


class TestStateAndReset:
    """状态与重置测试"""

    def test_get_state_includes_threshold_info(self) -> None:
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        state = v.get_state()
        assert state["threshold_count"] == 1
        assert "threshold_violations" in state

    def test_reset_restores_initial_state(self) -> None:
        """D5: reset 后 thresholds 应从 config 恢复，而非清空。"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        v.reset()
        assert len(v.metrics) == 0
        # thresholds 应从 config 恢复，保持与构造时一致
        assert len(v.thresholds) == 1
        assert "cpu" in v.thresholds

    def test_health_check(self) -> None:
        v = EffectVerifier({})
        assert v.health_check() is True

    def test_health_check_invalid(self) -> None:
        v = EffectVerifier({"max_points": 0})
        assert v.health_check() is False


class TestErrorHandlingHardened:
    """D1: 错误处理加固测试"""

    def test_capture_before_empty_name_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError, match="name must not be empty"):
            v.capture_before("", 1.0)

    def test_capture_before_invalid_value_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError):
            v.capture_before("x", "not_a_number")

    def test_capture_after_empty_name_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError, match="name must not be empty"):
            v.capture_after("  ", 1.0)

    def test_capture_after_invalid_value_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError):
            v.capture_after("x", None)

    def test_set_threshold_empty_name_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError, match="name must not be empty"):
            v.set_threshold("", min_val=0.0)

    def test_set_threshold_min_gt_max_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError, match="min_val.*max_val"):
            v.set_threshold("x", min_val=10.0, max_val=5.0)

    def test_set_threshold_invalid_min_raises(self) -> None:
        from root.left.app.left.src.utils_main.exceptions import ValidationError
        v = EffectVerifier({})
        with pytest.raises(ValidationError):
            v.set_threshold("x", min_val="invalid")

    def test_health_check_negative_sigma(self) -> None:
        v = EffectVerifier({"significance_sigma": -1.0})
        assert v.health_check() is False

    def test_health_check_negative_min_samples(self) -> None:
        v = EffectVerifier({"min_samples_for_significance": -1})
        assert v.health_check() is False

    def test_health_check_logs_warning(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        v = EffectVerifier({"max_points": 0})
        with caplog.at_level(logging.WARNING):
            result = v.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)


class TestEdgeCases:
    """D2: 边界条件与异常输入测试"""

    def test_capture_after_first_capture(self) -> None:
        """首次 capture_after 无 before 值"""
        v = EffectVerifier({})
        result = v.capture_after("cpu", 50.0)
        assert result["before"] is None
        assert result["delta"] == 0.0
        assert result["improved"] is True

    def test_capture_after_lower_is_better_keywords(self) -> None:
        """lower_is_better 关键词检测"""
        v = EffectVerifier({})
        v.capture_before("error_rate", 0.5)
        result = v.capture_after("error_rate", 0.3)
        assert result["improved"] is True

        v.capture_before("latency", 100.0)
        result = v.capture_after("latency", 80.0)
        assert result["improved"] is True

    def test_capture_after_higher_is_better_default(self) -> None:
        """默认 higher_is_better"""
        v = EffectVerifier({})
        v.capture_before("throughput", 100.0)
        result = v.capture_after("throughput", 120.0)
        assert result["improved"] is True

        v.capture_before("throughput", 120.0)
        result = v.capture_after("throughput", 100.0)
        assert result["improved"] is False

    def test_is_significant_exactly_min_samples(self) -> None:
        """恰好 min_samples 个样本时开始判断"""
        v = EffectVerifier({"min_samples_for_significance": 3})
        v.capture_before("x", 10.0)
        v.capture_before("x", 10.0)
        v.capture_before("x", 10.0)
        # 恰好 3 个样本，应能判断
        assert v.is_significant("x", 20.0) is True

    def test_is_significant_one_below_min(self) -> None:
        """比 min_samples 少一个时不判断"""
        v = EffectVerifier({"min_samples_for_significance": 3})
        v.capture_before("x", 10.0)
        v.capture_before("x", 10.0)
        assert v.is_significant("x", 100.0) is False

    def test_get_trend_exactly_two_samples(self) -> None:
        """恰好 2 个样本时能计算趋势"""
        v = EffectVerifier({})
        v.capture_before("x", 1.0)
        v.capture_before("x", 2.0)
        slope, direction = v.get_trend("x")
        assert slope > 0
        assert direction == "up"

    def test_get_trend_unknown_metric(self) -> None:
        """未知指标返回 unknown"""
        v = EffectVerifier({})
        slope, direction = v.get_trend("nonexistent")
        assert slope == 0.0
        assert direction == "unknown"

    def test_get_change_rate_inf_when_mean_zero(self) -> None:
        """均值为 0 且新值非 0 时返回 inf"""
        v = EffectVerifier({})
        v.capture_before("x", 0.0)
        v.capture_before("x", 0.0)
        v.capture_before("x", 5.0)
        rate = v.get_change_rate("x")
        assert rate == float('inf')

    def test_get_change_rate_unknown_metric(self) -> None:
        """未知指标返回 0"""
        v = EffectVerifier({})
        assert v.get_change_rate("nonexistent") == 0.0

    def test_threshold_violations_multiple(self) -> None:
        """多指标多违规"""
        v = EffectVerifier({"thresholds": {
            "cpu": {"max": 50.0},
            "memory": {"min": 20.0},
        }})
        v.capture_before("cpu", 80.0)
        v.capture_before("memory", 10.0)
        violations = v.get_threshold_violations()
        assert len(violations) == 2
        metrics = [v["metric"] for v in violations]
        assert "cpu" in metrics
        assert "memory" in metrics

    def test_detailed_summary_with_threshold_violation(self) -> None:
        """详细汇总含阈值违规信息"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 50.0}}})
        v.capture_before("cpu", 80.0)
        details = v.get_detailed_summary()
        assert details["cpu"]["threshold_ok"] is False

    def test_max_points_boundary_one(self) -> None:
        """max_points=1 边界"""
        v = EffectVerifier({"max_points": 1})
        v.capture_before("x", 1.0)
        v.capture_before("x", 2.0)
        # 只保留最新
        assert len(v.metrics["x"]) == 1
        assert v.metrics["x"][0] == 2.0

    def test_reset_preserves_config(self) -> None:
        """reset 后配置项保留"""
        v = EffectVerifier({"max_points": 5, "significance_sigma": 2.0})
        v.capture_before("x", 1.0)
        v.reset()
        assert v.max_points == 5
        assert v.significance_sigma == 2.0
        assert len(v.metrics) == 0


class TestStateConsistency:
    """D5: 状态一致性与幂等性测试"""

    def test_reset_restores_initial_state_with_config_thresholds(self) -> None:
        """reset 后状态等价于新构造实例（含 config 阈值）。"""
        config = {"thresholds": {"cpu": {"max": 80.0}, "mem": {"min": 10.0}}}
        v1 = EffectVerifier(config)
        v2 = EffectVerifier(config)
        # 对 v1 做若干操作
        v1.capture_before("cpu", 90.0)
        v1.capture_after("mem", 50.0)
        v1.set_threshold("latency", min_val=0.0, max_val=100.0)
        # reset 后应与 v2（未操作）状态一致
        v1.reset()
        assert v1.thresholds == v2.thresholds
        assert v1.metrics == v2.metrics

    def test_reset_idempotent(self) -> None:
        """连续调用 reset 两次，状态应一致。"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        v.reset()
        state_after_first = v.get_state()
        v.reset()
        state_after_second = v.get_state()
        assert state_after_first == state_after_second

    def test_health_check_idempotent(self) -> None:
        """连续调用 health_check 多次，结果应一致且无副作用。"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        state_before = v.get_state()
        result1 = v.health_check()
        result2 = v.health_check()
        result3 = v.health_check()
        state_after = v.get_state()
        assert result1 == result2 == result3
        assert state_before == state_after

    def test_get_state_idempotent(self) -> None:
        """连续调用 get_state 多次，结果应一致且不改变内部状态。"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.capture_before("cpu", 50.0)
        state1 = v.get_state()
        state2 = v.get_state()
        state3 = v.get_state()
        assert state1 == state2 == state3

    def test_reset_clears_runtime_added_thresholds(self) -> None:
        """reset 后运行时通过 set_threshold 添加的阈值被清除，config 阈值保留。"""
        v = EffectVerifier({"thresholds": {"cpu": {"max": 80.0}}})
        v.set_threshold("latency", max_val=100.0)
        assert len(v.thresholds) == 2
        v.reset()
        # 只有 config 中的 cpu 阈值保留
        assert len(v.thresholds) == 1
        assert "cpu" in v.thresholds
        assert "latency" not in v.thresholds
