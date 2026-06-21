"""效果验证器

在执行动作前后采集关键指标，评估动作是否产生预期效果。

增强能力：
1. 指标阈值检查：可配置每个指标的上下限阈值
2. 统计显著性：基于历史样本均值/标准差判断变化是否显著
3. 历史趋势：计算指标的线性趋势斜率与变化率

错误处理策略：
- 公共方法验证输入类型与值，无效输入抛 ValidationError
- health_check 返回 False 时记录 warning 以便调试
"""
import math
from typing import Any, Dict, Final, List, Optional, Tuple

from root.left.app.left.src.utils_main.exceptions import ValidationError
from root.left.app.left.src.utils_main.logger import logger
from root.left.app.left.src.utils_main.validators import Validator


class EffectVerifier:
    """效果验证器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[EffectVerifier.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.metrics: Dict[str, List[float]] = {}
        self.max_points: int = int(config.get("max_points", 10))
        # 指标阈值配置：{"metric_name": {"min": 0.0, "max": 1.0}}
        self.thresholds: Dict[str, Dict[str, float]] = dict(config.get("thresholds", {}))
        # 显著性水平（标准差倍数），变化超过此倍数视为显著
        self.significance_sigma: float = float(config.get("significance_sigma", 1.0))
        # 最小样本数才进行显著性判断
        self.min_samples_for_significance: int = int(config.get("min_samples_for_significance", 3))
        logger.info(f"EffectVerifier initialized max_points={self.max_points}", "effect")

    def capture_before(self, name: str, value: float) -> None:
        Validator.require_type(name, str, "name")
        if not name.strip():
            raise ValidationError("name must not be empty", field="name")
        Validator.require_type(value, (int, float), "value")
        logger.debug(f"[EffectVerifier.capture_before] {name}={value}", "effect")
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(float(value))
        if len(self.metrics[name]) > self.max_points:
            self.metrics[name].pop(0)

    def capture_after(self, name: str, value: float) -> Dict[str, Any]:
        Validator.require_type(name, str, "name")
        if not name.strip():
            raise ValidationError("name must not be empty", field="name")
        Validator.require_type(value, (int, float), "value")
        value = float(value)
        logger.debug(f"[EffectVerifier.capture_after] {name}={value}", "effect")
        before = self._get_before(name)
        delta = value - before if before is not None else 0.0
        improved = self._evaluate(name, before, value)
        significant = self.is_significant(name, value)
        threshold_ok = self.check_threshold(name, value)
        trend = self.get_trend(name)
        result = {
            "metric": name,
            "before": before,
            "after": value,
            "delta": delta,
            "improved": improved,
            "significant": significant,
            "threshold_ok": threshold_ok,
            "trend_slope": trend[0],
            "trend_direction": trend[1],
        }
        self.capture_before(name, value)
        logger.info(f"[EffectVerifier.capture_after] result={result}", "effect")
        return result

    def _get_before(self, name: str) -> Optional[float]:
        series = self.metrics.get(name, [])
        if len(series) < 2:
            return series[0] if series else None
        return series[-2]

    def _evaluate(self, name: str, before: Optional[float], after: float) -> bool:
        if before is None:
            return True
        lower_is_better = any(keyword in name for keyword in ["error", "latency", "cpu", "memory", "disk"])
        if lower_is_better:
            return after < before
        return after > before

    # ------------------------------------------------------------------
    # 阈值检查
    # ------------------------------------------------------------------

    def set_threshold(self, name: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
        """设置指标阈值。

        Args:
            name: 指标名
            min_val: 最小值（None 表示不限）
            max_val: 最大值（None 表示不限）

        Raises:
            ValidationError: name 为空或 min_val > max_val
        """
        Validator.require_type(name, str, "name")
        if not name.strip():
            raise ValidationError("name must not be empty", field="name")
        if min_val is not None:
            Validator.require_type(min_val, (int, float), "min_val")
            min_val = float(min_val)
        if max_val is not None:
            Validator.require_type(max_val, (int, float), "max_val")
            max_val = float(max_val)
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValidationError(
                f"min_val {min_val} > max_val {max_val}",
                field="min_val",
            )
        if name not in self.thresholds:
            self.thresholds[name] = {}
        if min_val is not None:
            self.thresholds[name]["min"] = min_val
        if max_val is not None:
            self.thresholds[name]["max"] = max_val
        logger.debug(f"[EffectVerifier.set_threshold] {name}: {self.thresholds[name]}", "effect")

    def check_threshold(self, name: str, value: float) -> bool:
        """检查指标值是否在阈值范围内。

        Args:
            name: 指标名
            value: 当前值

        Returns:
            True 表示在范围内或未配置阈值
        """
        threshold = self.thresholds.get(name)
        if not threshold:
            return True
        min_val = threshold.get("min")
        max_val = threshold.get("max")
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    def get_threshold_violations(self) -> List[Dict[str, Any]]:
        """获取当前所有指标的阈值违规情况。

        Returns:
            违规列表，每项含 metric/value/min/max
        """
        violations: List[Dict[str, Any]] = []
        for name, values in self.metrics.items():
            if not values:
                continue
            latest = values[-1]
            if not self.check_threshold(name, latest):
                threshold = self.thresholds.get(name, {})
                violations.append({
                    "metric": name,
                    "value": latest,
                    "min": threshold.get("min"),
                    "max": threshold.get("max"),
                })
        return violations

    # ------------------------------------------------------------------
    # 统计显著性
    # ------------------------------------------------------------------

    def _compute_stats(self, values: List[float]) -> Tuple[float, float]:
        """计算均值与标准差。

        Args:
            values: 样本列表

        Returns:
            (mean, std)，空列表返回 (0.0, 0.0)
        """
        if not values:
            return 0.0, 0.0
        n = len(values)
        mean = sum(values) / n
        if n < 2:
            return mean, 0.0
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return mean, math.sqrt(variance)

    def is_significant(self, name: str, value: float) -> bool:
        """判断新值相对于历史样本是否显著变化。

        基于标准差倍数判断：|value - mean| > significance_sigma * std

        Args:
            name: 指标名
            value: 待检验值

        Returns:
            是否为显著变化
        """
        values = self.metrics.get(name, [])
        if len(values) < self.min_samples_for_significance:
            return False
        mean, std = self._compute_stats(values)
        if std == 0:
            # 历史值全部相同，任何变化都显著
            return abs(value - mean) > 1e-9
        return abs(value - mean) > self.significance_sigma * std

    # ------------------------------------------------------------------
    # 历史趋势
    # ------------------------------------------------------------------

    def get_trend(self, name: str) -> Tuple[float, str]:
        """计算指标的历史趋势（简单线性回归斜率）。

        Args:
            name: 指标名

        Returns:
            (slope, direction)：斜率与方向描述（"up"/"down"/"stable"/"unknown"）
        """
        values = self.metrics.get(name, [])
        if len(values) < 2:
            return 0.0, "unknown"
        n = len(values)
        # 简单线性回归：y = a + b*x，x = 0,1,...,n-1
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0, "stable"
        slope = numerator / denominator
        # 根据斜率相对均值的大小判断方向
        if y_mean != 0:
            relative_slope = abs(slope) / abs(y_mean)
        else:
            relative_slope = abs(slope)
        if relative_slope < 0.01:
            return slope, "stable"
        return slope, "up" if slope > 0 else "down"

    def get_change_rate(self, name: str) -> float:
        """计算指标最近变化率（最新值相对均值的偏离百分比）。

        Args:
            name: 指标名

        Returns:
            变化率（百分比），无历史时返回 0.0
        """
        values = self.metrics.get(name, [])
        if len(values) < 2:
            return 0.0
        mean, _ = self._compute_stats(values[:-1])  # 排除最新值
        if mean == 0:
            return 0.0 if values[-1] == 0 else float('inf')
        return (values[-1] - mean) / abs(mean) * 100.0

    # ------------------------------------------------------------------
    # 汇总与统一接口
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        return {
            "metrics": list(self.metrics.keys()),
            "latest": {name: values[-1] if values else None for name, values in self.metrics.items()},
        }

    def get_detailed_summary(self) -> Dict[str, Any]:
        """获取详细汇总，含统计信息与趋势。"""
        details: Dict[str, Any] = {}
        for name, values in self.metrics.items():
            mean, std = self._compute_stats(values)
            slope, direction = self.get_trend(name)
            change_rate = self.get_change_rate(name)
            threshold_ok = self.check_threshold(name, values[-1]) if values else True
            details[name] = {
                "count": len(values),
                "latest": values[-1] if values else None,
                "mean": mean,
                "std": std,
                "trend_slope": slope,
                "trend_direction": direction,
                "change_rate_pct": change_rate,
                "threshold_ok": threshold_ok,
            }
        return details

    def get_state(self) -> Dict[str, Any]:
        """获取验证器状态"""
        return {
            "max_points": self.max_points,
            "metric_count": len(self.metrics),
            "summary": self.get_summary(),
            "threshold_count": len(self.thresholds),
            "threshold_violations": self.get_threshold_violations(),
        }

    def reset(self) -> None:
        """重置验证器到初始状态。

        恢复 metrics 为空，thresholds 从 config 重新加载，
        确保 reset 后状态等价于新构造实例。
        """
        logger.debug("[EffectVerifier.reset] called", "effect")
        self.metrics.clear()
        # 从 config 重新加载阈值，保持与构造时一致
        self.thresholds = dict(self.config.get("thresholds", {}))

    def health_check(self) -> bool:
        """健康检查"""
        logger.debug("[EffectVerifier.health_check] called", "effect")
        if self.max_points <= 0:
            logger.warning(f"health_check fail: max_points={self.max_points} <= 0", "effect")
            return False
        if self.significance_sigma < 0:
            logger.warning(f"health_check fail: significance_sigma={self.significance_sigma} < 0", "effect")
            return False
        if self.min_samples_for_significance < 0:
            logger.warning(
                f"health_check fail: min_samples_for_significance={self.min_samples_for_significance} < 0",
                "effect",
            )
            return False
        return True
