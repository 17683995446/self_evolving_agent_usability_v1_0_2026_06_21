"""形式化验证与可解释性模块单元测试"""
import pytest

from root.left.app.left.src.core_mod.verify import (
    SafetyInvariant, LivenessProperty, DecisionTracer,
    FeatureAttributor, VerificationModule,
)
from root.left.app.left.src.utils_main.exceptions import ValidationError


class TestSafetyInvariant:
    """测试安全性不变式"""

    def test_normal_safe(self) -> None:
        inv = SafetyInvariant({})
        inv.register("cpu_ok", lambda s: s.get("cpu", 0) < 1.0)
        result = inv.check({"cpu": 0.5})
        assert result["safe"] is True
        assert result["checks"]["cpu_ok"] == "safe"

    def test_violated(self) -> None:
        inv = SafetyInvariant({})
        inv.register("cpu_ok", lambda s: s.get("cpu", 0) < 1.0)
        result = inv.check({"cpu": 1.5})
        assert result["safe"] is False
        assert result["checks"]["cpu_ok"] == "VIOLATED"
        assert len(inv.get_violations()) == 1

    def test_exception_in_check(self) -> None:
        inv = SafetyInvariant({})
        inv.register("bad", lambda s: s["missing"] > 0)
        result = inv.check({})
        assert result["safe"] is False

    def test_violations_bounded(self) -> None:
        inv = SafetyInvariant({"max_violations": 2})
        inv.register("v", lambda s: False)
        for _ in range(5):
            inv.check({})
        assert len(inv.get_violations()) == 2


class TestLivenessProperty:
    """测试活性属性"""

    def test_satisfied(self) -> None:
        lp = LivenessProperty({})
        lp.register("perf_improved", lambda s: s.get("perf", 0) > 0.5, 10)
        result = lp.check(1, {"perf": 0.6})
        assert result["alive"] is True
        assert result["checks"]["perf_improved"] == "satisfied"

    def test_pending(self) -> None:
        lp = LivenessProperty({})
        lp.register("perf_improved", lambda s: s.get("perf", 0) > 0.5, 10)
        result = lp.check(1, {"perf": 0.3})
        assert result["checks"]["perf_improved"] == "pending"

    def test_timeout(self) -> None:
        lp = LivenessProperty({})
        lp.register("perf_improved", lambda s: s.get("perf", 0) > 0.5, 2)
        lp.check(1, {"perf": 0.3})
        result = lp.check(5, {"perf": 0.3})
        assert result["alive"] is False
        assert result["checks"]["perf_improved"] == "TIMEOUT"

    def test_already_satisfied(self) -> None:
        lp = LivenessProperty({})
        lp.register("perf_improved", lambda s: s.get("perf", 0) > 0.5, 10)
        lp.check(1, {"perf": 0.6})
        result = lp.check(2, {"perf": 0.3})
        assert result["checks"]["perf_improved"] == "satisfied"


class TestDecisionTracer:
    """测试决策追踪"""

    def test_trace_and_explain(self) -> None:
        dt = DecisionTracer({})
        dt.trace("select_task", {"pool_size": 5}, "task_1", "highest priority")
        explain = dt.explain_last()
        assert "select_task" in explain
        assert "highest priority" in explain

    def test_trace_bounded(self) -> None:
        dt = DecisionTracer({"max_traces": 2})
        for i in range(5):
            dt.trace("step", {}, i, "reason")
        assert len(dt.get_all_traces()) == 2

    def test_empty_explain(self) -> None:
        dt = DecisionTracer({})
        assert dt.explain_last() == "No trace available"


class TestFeatureAttributor:
    """测试特征归因"""

    def test_gradient_method(self) -> None:
        fa = FeatureAttributor({"method": "gradient"})
        features = [1.0, 2.0, 3.0]
        names = ["a", "b", "c"]

        def predict_fn(x):
            return sum(x)

        attrs = fa.attribute(features, names, predict_fn)
        assert len(attrs) == 3
        assert abs(sum(attrs.values()) - 1.0) < 0.01

    def test_perturbation_method(self) -> None:
        fa = FeatureAttributor({"method": "perturbation"})
        features = [1.0, 2.0]
        names = ["a", "b"]

        def predict_fn(x):
            return x[0] * 2 + x[1] * 0.5

        attrs = fa.attribute(features, names, predict_fn)
        assert len(attrs) == 2
        assert attrs["a"] > attrs["b"]

    def test_unknown_method(self) -> None:
        fa = FeatureAttributor({"method": "unknown"})
        attrs = fa.attribute([1.0], ["a"], lambda x: x[0])
        assert attrs["a"] == 0.0


class TestVerificationModule:
    """测试验证模块主控"""

    def test_normal_init(self) -> None:
        mod = VerificationModule({})
        assert mod.safety is not None

    def test_safety_check(self) -> None:
        mod = VerificationModule({})
        mod.register_safety("cpu_ok", lambda s: s.get("cpu", 0) < 1.0)
        result = mod.check_safety({"cpu": 0.5})
        assert result["safe"] is True

    def test_liveness_check(self) -> None:
        mod = VerificationModule({})
        mod.register_liveness("improved", lambda s: s.get("p", 0) > 0.5, 10)
        result = mod.check_liveness(1, {"p": 0.6})
        assert result["alive"] is True

    def test_trace_and_explain(self) -> None:
        mod = VerificationModule({})
        mod.trace_decision("test", {}, "out", "reason")
        assert "test" in mod.explain_decision()

    def test_attribute(self) -> None:
        mod = VerificationModule({})
        attrs = mod.attribute_features([1.0, 2.0], ["a", "b"], lambda x: sum(x))
        assert len(attrs) == 2

    def test_verification_state(self) -> None:
        mod = VerificationModule({})
        state = mod.get_verification_state()
        assert "safety_violations" in state
        assert "trace_count" in state

    def test_get_state_unified(self) -> None:
        """统一接口 get_state()"""
        mod = VerificationModule({})
        mod.register_safety("cpu_ok", lambda s: s.get("cpu", 0) < 1.0)
        mod.trace_decision("test", {}, "out", "reason")
        state = mod.get_state()
        assert "safety_violations" in state
        assert "trace_count" in state
        assert "liveness_properties" in state
        assert state["trace_count"] == 1

    def test_reset_unified(self) -> None:
        """统一接口 reset() 应清空运行时状态"""
        mod = VerificationModule({})
        mod.register_safety("cpu_ok", lambda s: True)
        mod.trace_decision("test", {}, "out", "reason")
        assert len(mod.tracer.traces) == 1
        mod.reset()
        # 重置后子组件应为新实例
        assert mod.safety is not None
        assert mod.tracer is not None
        # 新实例的 traces 应为空
        assert len(mod.tracer.traces) == 0

    def test_health_check_unified(self) -> None:
        """统一接口 health_check()"""
        mod = VerificationModule({})
        assert mod.health_check() is True

    def test_health_check_after_reset(self) -> None:
        mod = VerificationModule({})
        mod.reset()
        assert mod.health_check() is True

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            VerificationModule(None)


class TestErrorHandlingHardened:
    """D1: 错误处理加固测试"""

    def test_safety_check_logs_exception(self, caplog) -> None:
        """SafetyInvariant.check 中函数抛异常时应记录 warning"""
        import logging
        mod = VerificationModule({})
        mod.register_safety("bad_check", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        with caplog.at_level(logging.WARNING):
            result = mod.check_safety({})
        assert result["safe"] is False
        assert "bad_check" in result["checks"]
        assert any("bad_check" in r.message and "raised" in r.message for r in caplog.records)

    def test_liveness_check_logs_exception(self, caplog) -> None:
        """LivenessProperty.check 中函数抛异常时应记录 warning"""
        import logging
        mod = VerificationModule({})
        mod.register_liveness("bad_prop", lambda s: (_ for _ in ()).throw(ValueError("err")), 10)
        with caplog.at_level(logging.WARNING):
            result = mod.check_liveness(1, {})
        assert result["alive"] is False
        assert any("bad_prop" in r.message and "raised" in r.message for r in caplog.records)

    def test_health_check_logs_warning_on_failure(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        mod = VerificationModule({})
        mod.safety = None  # 模拟不健康状态
        with caplog.at_level(logging.WARNING):
            result = mod.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)


class TestEdgeCases:
    """D2: 边界条件与异常输入测试"""

    def test_safety_check_no_invariants(self) -> None:
        """无不变式注册时 check 返回 safe=True"""
        mod = VerificationModule({})
        result = mod.check_safety({})
        assert result["safe"] is True
        assert result["checks"] == {}

    def test_safety_check_empty_state(self) -> None:
        """空 state 字典"""
        mod = VerificationModule({})
        mod.register_safety("always_safe", lambda s: True)
        result = mod.check_safety({})
        assert result["safe"] is True

    def test_safety_violations_bounded(self) -> None:
        """违规列表有界"""
        mod = VerificationModule({"safety": {"max_violations": 3}})
        mod.register_safety("always_fail", lambda s: False)
        for _ in range(10):
            mod.check_safety({})
        assert len(mod.safety.get_violations()) == 3

    def test_liveness_check_no_properties(self) -> None:
        """无活性属性注册时 check 返回 alive=True"""
        mod = VerificationModule({})
        result = mod.check_liveness(1, {})
        assert result["alive"] is True

    def test_liveness_reset_unknown_name(self) -> None:
        """reset 未知名称不报错"""
        mod = VerificationModule({})
        mod.liveness.reset("nonexistent", 1)  # 不应抛异常

    def test_liveness_timeout_boundary(self) -> None:
        """恰好超时边界"""
        mod = VerificationModule({})
        mod.register_liveness("never_satisfied", lambda s: False, 5)
        # loop 5 时还未超时（5 - 0 = 5，不大于 5）
        result = mod.check_liveness(5, {})
        assert result["checks"]["never_satisfied"] == "pending"
        # loop 6 时超时（6 - 0 = 6 > 5）
        result = mod.check_liveness(6, {})
        assert result["checks"]["never_satisfied"] == "TIMEOUT"
        assert result["alive"] is False

    def test_tracer_empty_get_trace(self) -> None:
        """空 traces 时 get_trace 返回 None"""
        mod = VerificationModule({})
        assert mod.tracer.get_trace() is None
        assert mod.tracer.get_trace(0) is None

    def test_tracer_explain_last_empty(self) -> None:
        """空 traces 时 explain_last 返回提示"""
        mod = VerificationModule({})
        assert "No trace" in mod.explain_decision()

    def test_tracer_bounded(self) -> None:
        """traces 有界"""
        mod = VerificationModule({"tracer": {"max_traces": 3}})
        for i in range(10):
            mod.trace_decision("point", {"i": i}, f"out{i}", "reason")
        assert len(mod.tracer.get_all_traces()) == 3

    def test_attributor_unknown_method(self) -> None:
        """未知归因方法返回全 0"""
        mod = VerificationModule({"attributor": {"method": "unknown"}})
        attrs = mod.attribute_features([1.0, 2.0], ["a", "b"], lambda x: sum(x))
        assert attrs == {"a": 0.0, "b": 0.0}

    def test_attributor_empty_features(self) -> None:
        """空特征列表"""
        mod = VerificationModule({})
        attrs = mod.attribute_features([], [], lambda x: 0.0)
        assert attrs == {}


class TestStateConsistency:
    """D5: 状态一致性与幂等性测试"""

    def test_reset_restores_initial_state(self) -> None:
        """reset 后 safety/liveness/tracer 均回到初始状态。"""
        mod = VerificationModule({})
        # 产生状态
        mod.register_safety("s1", lambda s: True)
        mod.check_safety({"x": 1})
        # 故意触发一次违规
        mod.register_safety("s2", lambda s: False)
        mod.check_safety({"x": 1})
        mod.register_liveness("l1", lambda s: True, timeout=10)
        mod.check_liveness(1, {"x": 1})
        mod.trace_decision("point", {"i": 1}, "out", "reason")
        assert len(mod.safety.get_violations()) > 0
        assert len(mod.tracer.traces) > 0
        assert len(mod.liveness.properties) > 0
        # reset
        mod.reset()
        assert len(mod.safety.get_violations()) == 0
        assert len(mod.safety.invariants) == 0
        assert len(mod.tracer.traces) == 0
        assert len(mod.liveness.properties) == 0

    def test_reset_state_equals_fresh_instance(self) -> None:
        """reset 后 get_state 等价于新构造实例。"""
        config = {"safety": {"max_violations": 50}, "tracer": {"max_traces": 100}}
        mod1 = VerificationModule(config)
        mod2 = VerificationModule(config)
        mod1.register_safety("s1", lambda s: True)
        mod1.check_safety({"x": 1})
        mod1.trace_decision("p", {}, "o", "r")
        mod1.reset()
        assert mod1.get_state() == mod2.get_state()

    def test_reset_idempotent(self) -> None:
        """连续调用 reset 两次，状态一致。"""
        mod = VerificationModule({})
        mod.register_safety("s1", lambda s: True)
        mod.check_safety({"x": 1})
        mod.trace_decision("p", {}, "o", "r")
        mod.reset()
        state1 = mod.get_state()
        mod.reset()
        state2 = mod.get_state()
        assert state1 == state2

    def test_health_check_idempotent(self) -> None:
        """连续调用 health_check 多次，结果一致且无副作用。"""
        mod = VerificationModule({})
        mod.register_safety("s1", lambda s: True)
        mod.check_safety({"x": 1})
        mod.trace_decision("p", {}, "o", "r")
        state_before = mod.get_state()
        r1 = mod.health_check()
        r2 = mod.health_check()
        r3 = mod.health_check()
        state_after = mod.get_state()
        assert r1 == r2 == r3 is True
        assert state_before == state_after

    def test_get_state_idempotent(self) -> None:
        """连续调用 get_state 多次，结果一致且不改变内部状态。"""
        mod = VerificationModule({})
        mod.register_safety("s1", lambda s: True)
        mod.check_safety({"x": 1})
        mod.trace_decision("p", {}, "o", "r")
        s1 = mod.get_state()
        s2 = mod.get_state()
        s3 = mod.get_state()
        assert s1 == s2 == s3
