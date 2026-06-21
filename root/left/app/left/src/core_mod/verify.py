"""形式化验证与可解释性模块"""
from typing import Any, Callable, Dict, Final, List, Optional


from root.left.app.left.src.utils_main.logger import logger
from root.left.app.left.src.utils_main.validators import Validator


class SafetyInvariant:
    """安全性不变式"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[SafetyInvariant.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.invariants: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.violations: List[Dict[str, Any]] = []
        self.max_violations: int = int(config.get("max_violations", 100))
        logger.info("SafetyInvariant initialized", "verify")

    def register(self, name: str, check_fn: Callable[[Dict[str, Any]], bool]) -> None:
        logger.debug(f"[SafetyInvariant.register] name={name}")
        self.invariants[name] = check_fn

    def check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("[SafetyInvariant.check] start")
        results = {}
        all_safe = True
        for name, fn in self.invariants.items():
            try:
                ok = fn(state)
                results[name] = "safe" if ok else "VIOLATED"
                if not ok:
                    all_safe = False
                    self._record_violation(name, state)
            except Exception as e:
                logger.warning(f"[SafetyInvariant.check] {name} raised: {e}", "verify")
                results[name] = f"error: {e}"
                all_safe = False
        result = {"safe": all_safe, "checks": results}
        logger.debug(f"[SafetyInvariant.check] safe={all_safe}")
        return result

    def _record_violation(self, name: str, state: Dict[str, Any]) -> None:
        self.violations.append({"invariant": name, "state": state})
        if len(self.violations) > self.max_violations:
            self.violations.pop(0)
        logger.warning(f"[SafetyInvariant._record_violation] {name} violated", "verify")

    def get_violations(self) -> List[Dict[str, Any]]:
        return list(self.violations)


class LivenessProperty:
    """活性属性检查器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[LivenessProperty.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.properties: Dict[str, Dict[str, Any]] = {}
        logger.info("LivenessProperty initialized", "verify")

    def register(
        self, name: str, condition_fn: Callable[[Dict[str, Any]], bool], timeout_loops: int
    ) -> None:
        logger.debug(f"[LivenessProperty.register] name={name} timeout={timeout_loops}")
        self.properties[name] = {
            "condition": condition_fn,
            "timeout": timeout_loops,
            "since_loop": 0,
            "satisfied": False,
        }

    def check(self, loop_count: int, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"[LivenessProperty.check] loop={loop_count}")
        results = {}
        all_alive = True
        for name, prop in self.properties.items():
            if prop["satisfied"]:
                results[name] = "satisfied"
                continue
            try:
                satisfied = prop["condition"](state)
                if satisfied:
                    prop["satisfied"] = True
                    results[name] = "satisfied"
                    logger.info(f"[LivenessProperty.check] {name} satisfied at loop {loop_count}", "verify")
                elif loop_count - prop["since_loop"] > prop["timeout"]:
                    results[name] = "TIMEOUT"
                    all_alive = False
                    logger.error(f"[LivenessProperty.check] {name} TIMEOUT", "verify")
                else:
                    results[name] = "pending"
            except Exception as e:
                logger.warning(f"[LivenessProperty.check] {name} raised: {e}", "verify")
                results[name] = f"error: {e}"
                all_alive = False
        result = {"alive": all_alive, "checks": results}
        logger.debug(f"[LivenessProperty.check] alive={all_alive}")
        return result

    def reset(self, name: str, loop_count: int) -> None:
        logger.debug(f"[LivenessProperty.reset] name={name}")
        if name in self.properties:
            self.properties[name]["since_loop"] = loop_count
            self.properties[name]["satisfied"] = False


class DecisionTracer:
    """决策路径追踪器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[DecisionTracer.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.traces: List[Dict[str, Any]] = []
        self.max_traces: int = int(config.get("max_traces", 200))
        logger.info("DecisionTracer initialized", "verify")

    def trace(self, decision_point: str, inputs: Dict[str, Any], output: Any, reasoning: str) -> None:
        logger.debug(f"[DecisionTracer.trace] point={decision_point}")
        self.traces.append({
            "decision_point": decision_point,
            "inputs": inputs,
            "output": output,
            "reasoning": reasoning,
        })
        if len(self.traces) > self.max_traces:
            self.traces.pop(0)

    def get_trace(self, index: int = -1) -> Optional[Dict[str, Any]]:
        if not self.traces:
            return None
        return self.traces[index]

    def get_all_traces(self) -> List[Dict[str, Any]]:
        return list(self.traces)

    def explain_last(self) -> str:
        trace = self.get_trace()
        if trace is None:
            return "No trace available"
        return (
            f"Decision: {trace['decision_point']}\n"
            f"Reasoning: {trace['reasoning']}\n"
            f"Output: {trace['output']}"
        )


class FeatureAttributor:
    """特征重要性归因器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[FeatureAttributor.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.method: str = str(config.get("method", "gradient"))
        logger.info(f"FeatureAttributor initialized method={self.method}", "verify")

    def attribute(
        self,
        features: List[float],
        feature_names: List[str],
        predict_fn: Callable[[List[float]], float]
    ) -> Dict[str, float]:
        logger.debug(f"[FeatureAttributor.attribute] n_features={len(features)}")
        if self.method == "gradient":
            return self._gradient_attr(features, feature_names, predict_fn)
        elif self.method == "perturbation":
            return self._perturbation_attr(features, feature_names, predict_fn)
        else:
            return {name: 0.0 for name in feature_names}

    def _gradient_attr(
        self,
        features: List[float],
        feature_names: List[str],
        predict_fn: Callable[[List[float]], float]
    ) -> Dict[str, float]:
        base_pred = predict_fn(features)
        attrs = {}
        eps = 0.01
        for i, name in enumerate(feature_names):
            perturbed = list(features)
            perturbed[i] += eps
            new_pred = predict_fn(perturbed)
            attrs[name] = abs(new_pred - base_pred) / eps
        total = sum(attrs.values()) + 1e-8
        for name in attrs:
            attrs[name] /= total
        logger.debug("[FeatureAttributor._gradient_attr] done")
        return attrs

    def _perturbation_attr(
        self,
        features: List[float],
        feature_names: List[str],
        predict_fn: Callable[[List[float]], float]
    ) -> Dict[str, float]:
        base_pred = predict_fn(features)
        attrs = {}
        for i, name in enumerate(feature_names):
            perturbed = list(features)
            perturbed[i] = 0.0
            new_pred = predict_fn(perturbed)
            attrs[name] = abs(base_pred - new_pred)
        total = sum(attrs.values()) + 1e-8
        for name in attrs:
            attrs[name] /= total
        logger.debug("[FeatureAttributor._perturbation_attr] done")
        return attrs


class VerificationModule:
    """验证模块主控"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[VerificationModule.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.safety = SafetyInvariant(config.get("safety", {}))
        logger.debug("[VerificationModule.__init__] safety created")
        self.liveness = LivenessProperty(config.get("liveness", {}))
        logger.debug("[VerificationModule.__init__] liveness created")
        self.tracer = DecisionTracer(config.get("tracer", {}))
        logger.debug("[VerificationModule.__init__] tracer created")
        self.attributor = FeatureAttributor(config.get("attributor", {}))
        logger.debug("[VerificationModule.__init__] attributor created")
        logger.info("VerificationModule initialized", "verify")

    def register_safety(self, name: str, check_fn: Callable[[Dict[str, Any]], bool]) -> None:
        self.safety.register(name, check_fn)

    def check_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.safety.check(state)

    def register_liveness(
        self, name: str, condition_fn: Callable[[Dict[str, Any]], bool], timeout: int
    ) -> None:
        self.liveness.register(name, condition_fn, timeout)

    def check_liveness(self, loop_count: int, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.liveness.check(loop_count, state)

    def trace_decision(self, point: str, inputs: Dict[str, Any], output: Any, reasoning: str) -> None:
        self.tracer.trace(point, inputs, output, reasoning)

    def explain_decision(self) -> str:
        return self.tracer.explain_last()

    def attribute_features(
        self,
        features: List[float],
        names: List[str],
        predict_fn: Callable[[List[float]], float]
    ) -> Dict[str, float]:
        return self.attributor.attribute(features, names, predict_fn)

    def get_verification_state(self) -> Dict[str, Any]:
        return {
            "safety_violations": len(self.safety.get_violations()),
            "trace_count": len(self.tracer.traces),
        }

    def get_state(self) -> Dict[str, Any]:
        """获取模块状态（统一接口）。"""
        return {
            "safety_violations": len(self.safety.get_violations()),
            "trace_count": len(self.tracer.traces),
            "liveness_properties": len(self.liveness.properties),
            "config": dict(self.config),
        }

    def reset(self) -> None:
        """重置模块到初始状态（统一接口）。"""
        logger.info("[VerificationModule.reset] resetting state", "verify")
        self.safety = SafetyInvariant(self.config.get("safety", {}))
        self.liveness = LivenessProperty(self.config.get("liveness", {}))
        self.tracer = DecisionTracer(self.config.get("tracer", {}))
        self.attributor = FeatureAttributor(self.config.get("attributor", {}))

    def health_check(self) -> bool:
        """健康检查（统一接口）。

        Returns:
            模块是否健康：子组件均存在且安全违规数未超阈值
        """
        try:
            if self.safety is None:
                logger.warning("health_check fail: safety is None", "verify")
                return False
            if self.liveness is None:
                logger.warning("health_check fail: liveness is None", "verify")
                return False
            if self.tracer is None:
                logger.warning("health_check fail: tracer is None", "verify")
                return False
            # 安全违规数超过 1000 视为异常
            if len(self.safety.get_violations()) > 1000:
                logger.warning("health_check fail: safety violations > 1000", "verify")
                return False
            return True
        except Exception as exc:
            logger.warning(f"health_check exception: {exc}", "verify")
            return False
