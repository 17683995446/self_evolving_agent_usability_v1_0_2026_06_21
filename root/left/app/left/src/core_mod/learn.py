"""自适应学习算法升级模块

错误处理策略：
- 公共方法验证输入类型与值，无效输入抛 ValidationError
- 计算异常捕获并记录日志，避免静默失败
- health_check 返回 False 时记录 warning 以便调试
"""
from typing import Any, Dict, Final, List, Optional

import numpy as np

from root.left.app.left.src.utils_main.exceptions import ValidationError
from root.left.app.left.src.utils_main.logger import logger
from root.left.app.left.src.utils_main.validators import Validator


class IncrementalSGDRegressor:
    """增量SGD回归器（在线学习）"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[IncrementalSGDRegressor.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.learning_rate: float = float(config.get("learning_rate", 0.01))
        self.regularization: float = float(config.get("regularization", 0.001))
        self.n_features: int = int(config.get("n_features", 5))
        # 构造时校验：学习率必须为正，正则化必须非负
        if self.learning_rate <= 0:
            raise ValidationError(
                f"learning_rate must be positive, got {self.learning_rate}",
                field="learning_rate",
            )
        if self.regularization < 0:
            raise ValidationError(
                f"regularization must be non-negative, got {self.regularization}",
                field="regularization",
            )
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.n_samples: int = 0
        logger.info(f"IncrementalSGD initialized lr={self.learning_rate}", "learn")

    def partial_fit(self, x: List[float], y: float) -> None:
        logger.debug(f"[IncrementalSGDRegressor.partial_fit] n_samples={self.n_samples}")
        Validator.require_type(x, list, "x")
        if not x:
            raise ValidationError("x must not be empty", field="x")
        Validator.require_type(y, (int, float), "y")
        x_arr = np.array(x, dtype=np.float64)
        if self.weights is None:
            self.weights = np.zeros(len(x_arr), dtype=np.float64)
        elif len(x_arr) != len(self.weights):
            raise ValidationError(
                f"x length {len(x_arr)} != weights length {len(self.weights)}",
                field="x",
            )
        pred = float(np.dot(self.weights, x_arr) + self.bias)
        error = pred - y
        grad_w = error * x_arr + self.regularization * self.weights
        grad_b = error
        max_grad = 100.0
        grad_norm = np.linalg.norm(grad_w)
        if grad_norm > max_grad:
            grad_w = grad_w * (max_grad / grad_norm)
            logger.debug("[IncrementalSGDRegressor.partial_fit] gradient clipped")
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        self.n_samples += 1
        logger.debug(f"[IncrementalSGDRegressor.partial_fit] error={error:.4f}")

    def predict(self, x: List[float]) -> float:
        Validator.require_type(x, list, "x")
        if not x:
            return 0.0
        if self.weights is None:
            return 0.0
        x_arr = np.array(x, dtype=np.float64)
        if len(x_arr) != len(self.weights):
            raise ValidationError(
                f"x length {len(x_arr)} != weights length {len(self.weights)}",
                field="x",
            )
        return float(np.dot(self.weights, x_arr) + self.bias)

    def get_weights(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias,
            "n_samples": self.n_samples,
        }


class AutoFeatureEngineer:
    """自动特征工程"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[AutoFeatureEngineer.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.max_degree: int = int(config.get("max_degree", 2))
        self.include_interactions: bool = bool(config.get("include_interactions", True))
        # 构造时校验：max_degree 必须 >= 1
        if self.max_degree < 1:
            raise ValidationError(
                f"max_degree must be >= 1, got {self.max_degree}",
                field="max_degree",
            )
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.n_samples: int = 0
        logger.info(f"AutoFeatureEngineer initialized degree={self.max_degree}", "learn")

    def fit_transform(self, x: List[float]) -> List[float]:
        logger.debug("[AutoFeatureEngineer.fit_transform] start")
        Validator.require_type(x, list, "x")
        if not x:
            raise ValidationError("x must not be empty", field="x")
        x_arr = np.array(x, dtype=np.float64)
        features = [1.0]
        features.extend(x_arr.tolist())
        for d in range(2, self.max_degree + 1):
            features.extend((x_arr ** d).tolist())
        if self.include_interactions and len(x_arr) > 1:
            for i in range(len(x_arr)):
                for j in range(i + 1, len(x_arr)):
                    features.append(float(x_arr[i] * x_arr[j]))
        result = self._standardize(features)
        logger.debug(f"[AutoFeatureEngineer.fit_transform] in={len(x)} out={len(result)}")
        return result

    def _standardize(self, features: List[float]) -> List[float]:
        arr = np.array(features, dtype=np.float64)
        if self.scaler_mean is None:
            self.scaler_mean = np.zeros_like(arr)
            self.scaler_std = np.ones_like(arr)
        std = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        result = (arr - self.scaler_mean) / std
        return result.tolist()

    def update_scaler(self, features: List[float]) -> None:
        """在线更新标准化参数（Welford 算法）。

        Args:
            features: 单个样本的特征向量

        Raises:
            ValidationError: features 为空或非列表
        """
        Validator.require_type(features, list, "features")
        if not features:
            raise ValidationError("features must not be empty", field="features")
        arr = np.array(features, dtype=np.float64)
        self.n_samples += 1
        if self.scaler_mean is None:
            self.scaler_mean = arr.copy()
            self.scaler_std = np.zeros_like(arr)
        else:
            if len(arr) != len(self.scaler_mean):
                raise ValidationError(
                    f"features length {len(arr)} != scaler length {len(self.scaler_mean)}",
                    field="features",
                )
            delta = arr - self.scaler_mean
            self.scaler_mean += delta / self.n_samples
            # 累加 M2（平方差之和），后续 get_scaler_state 时转为标准差
            self.scaler_std += delta * (arr - self.scaler_mean)

    def get_scaler_state(self) -> Dict[str, Any]:
        """获取标准化参数状态。

        Returns:
            含 mean/std/n_samples 的字典；std 由 M2 计算得出
        """
        if self.scaler_mean is None or self.n_samples == 0:
            return {"mean": [], "std": [], "n_samples": 0}
        if self.n_samples > 1:
            variance = self.scaler_std / (self.n_samples - 1)
            std = np.sqrt(np.maximum(variance, 0.0))
        else:
            std = np.zeros_like(self.scaler_mean)
        return {
            "mean": self.scaler_mean.tolist(),
            "std": std.tolist(),
            "n_samples": self.n_samples,
        }


class ModelSelector:
    """模型选择器（AIC/BIC）"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[ModelSelector.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.criterion: str = str(config.get("criterion", "aic"))
        self.complexity_penalty: float = float(config.get("complexity_penalty", 2.0))
        self.models: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ModelSelector initialized criterion={self.criterion}", "learn")

    def evaluate(self, model_name: str, sse: float, n_params: int, n_samples: int) -> float:
        logger.debug(f"[ModelSelector.evaluate] model={model_name} sse={sse:.4f}")
        if n_samples <= n_params:
            return float('inf')
        # 防止 sse <= 0 导致 log 域异常
        safe_sse = max(sse, 1e-10)
        if self.criterion == "aic":
            score = n_samples * np.log(safe_sse / n_samples) + self.complexity_penalty * n_params
        elif self.criterion == "bic":
            score = n_samples * np.log(safe_sse / n_samples) + self.complexity_penalty * n_params * np.log(n_samples)
        else:
            score = safe_sse / n_samples
        logger.debug(f"[ModelSelector.evaluate] score={score:.4f}")
        return float(score)

    def select_best(self, candidates: List[Dict[str, Any]]) -> Optional[str]:
        logger.debug(f"[ModelSelector.select_best] candidates={len(candidates)}")
        if not candidates:
            return None
        best = min(candidates, key=lambda c: c.get("score", float('inf')))
        logger.info(f"[ModelSelector.select_best] best={best.get('name')}", "learn")
        return best.get("name")


class OverfitDetector:
    """过拟合检测器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[OverfitDetector.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.threshold: float = float(config.get("threshold", 0.2))
        self.train_errors: List[float] = []
        self.val_errors: List[float] = []
        logger.info(f"OverfitDetector initialized threshold={self.threshold}", "learn")

    def add_errors(self, train_error: float, val_error: float) -> None:
        logger.debug(f"[OverfitDetector.add_errors] train={train_error:.4f} val={val_error:.4f}")
        self.train_errors.append(train_error)
        self.val_errors.append(val_error)
        if len(self.train_errors) > 100:
            self.train_errors.pop(0)
            self.val_errors.pop(0)

    def is_overfitting(self) -> bool:
        if len(self.train_errors) < 10:
            return False
        recent_train = float(np.mean(self.train_errors[-10:]))
        recent_val = float(np.mean(self.val_errors[-10:]))
        gap = recent_val - recent_train
        result = bool(gap > self.threshold * max(recent_train, 1e-8))
        logger.debug(f"[OverfitDetector.is_overfitting] gap={gap:.4f} result={result}")
        return result

    def get_gap(self) -> float:
        if not self.train_errors or not self.val_errors:
            return 0.0
        return float(np.mean(self.val_errors[-10:]) - np.mean(self.train_errors[-10:]))


class AdaptiveLearner:
    """自适应学习器主控"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[AdaptiveLearner.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.sgd = IncrementalSGDRegressor(config.get("sgd", {}))
        logger.debug("[AdaptiveLearner.__init__] sgd created")
        self.featurizer = AutoFeatureEngineer(config.get("featurizer", {}))
        logger.debug("[AdaptiveLearner.__init__] featurizer created")
        self.selector = ModelSelector(config.get("selector", {}))
        logger.debug("[AdaptiveLearner.__init__] selector created")
        self.overfit = OverfitDetector(config.get("overfit", {}))
        logger.debug("[AdaptiveLearner.__init__] overfit created")
        self.history: List[Dict[str, Any]] = []
        self.max_history: int = int(config.get("max_history", 500))
        logger.info("AdaptiveLearner initialized", "learn")

    def learn(self, features: List[float], target: float, validation_target: Optional[float] = None) -> Dict[str, Any]:
        logger.debug(f"[AdaptiveLearner.learn] features={len(features)}")
        Validator.require_type(features, list, "features")
        if not features:
            raise ValidationError("features must not be empty", field="features")
        Validator.require_type(target, (int, float), "target")
        enriched = self.featurizer.fit_transform(features)
        pred_before = self.sgd.predict(enriched)
        self.sgd.partial_fit(enriched, target)
        pred_after = self.sgd.predict(enriched)
        train_error = abs(pred_after - target)
        val_error = abs(pred_after - validation_target) if validation_target is not None else train_error
        self.overfit.add_errors(train_error, val_error)
        record = {
            "target": target,
            "pred_before": pred_before,
            "pred_after": pred_after,
            "train_error": train_error,
            "val_error": val_error,
        }
        self._add_history(record)
        logger.debug(f"[AdaptiveLearner.learn] train_err={train_error:.4f} val_err={val_error:.4f}")
        return record

    def predict(self, features: List[float]) -> float:
        Validator.require_type(features, list, "features")
        if not features:
            return 0.0
        enriched = self.featurizer.fit_transform(features)
        return self.sgd.predict(enriched)

    def check_overfitting(self) -> bool:
        result = self.overfit.is_overfitting()
        if result:
            logger.warning("[AdaptiveLearner.check_overfitting] OVERFITTING DETECTED", "learn")
        return result

    def get_learning_state(self) -> Dict[str, Any]:
        return {
            "sgd": self.sgd.get_weights(),
            "overfit_gap": self.overfit.get_gap(),
            "history_size": len(self.history),
            "is_overfitting": self.overfit.is_overfitting(),
        }

    def get_state(self) -> Dict[str, Any]:
        """获取模块状态（统一接口）。"""
        return {
            "sgd": self.sgd.get_weights(),
            "scaler": self.featurizer.get_scaler_state(),
            "overfit_gap": self.overfit.get_gap(),
            "history_size": len(self.history),
            "is_overfitting": self.overfit.is_overfitting(),
            "n_samples": self.sgd.n_samples,
        }

    def reset(self) -> None:
        """重置模块到初始状态（统一接口）。"""
        logger.info("[AdaptiveLearner.reset] resetting state", "learn")
        self.sgd = IncrementalSGDRegressor(self.config.get("sgd", {}))
        self.featurizer = AutoFeatureEngineer(self.config.get("featurizer", {}))
        self.selector = ModelSelector(self.config.get("selector", {}))
        self.overfit = OverfitDetector(self.config.get("overfit", {}))
        self.history.clear()

    def health_check(self) -> bool:
        """健康检查（统一接口）。

        Returns:
            模块是否健康：SGD 权重有限且未检测到数值异常
        """
        try:
            if self.sgd.n_samples < 0:
                logger.warning("health_check fail: n_samples < 0", "learn")
                return False
            if self.sgd.weights is not None:
                if not np.all(np.isfinite(self.sgd.weights)):
                    logger.warning("health_check fail: weights contain non-finite values", "learn")
                    return False
                if not np.all(np.isfinite(self.sgd.bias)):
                    logger.warning("health_check fail: bias is non-finite", "learn")
                    return False
            return True
        except Exception as exc:
            logger.warning(f"health_check exception: {exc}", "learn")
            return False

    def _add_history(self, record: Dict[str, Any]) -> None:
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            logger.debug("[AdaptiveLearner._add_history] trimmed oldest record")
