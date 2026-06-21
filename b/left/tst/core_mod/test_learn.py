"""自适应学习模块单元测试"""
import pytest

from root.left.app.left.src.core_mod.learn import (
    IncrementalSGDRegressor, AutoFeatureEngineer,
    ModelSelector, OverfitDetector, AdaptiveLearner,
)
from root.left.app.left.src.utils_main.exceptions import ValidationError


class TestIncrementalSGDRegressor:
    """测试增量SGD"""

    def test_normal_fit_predict(self) -> None:
        sgd = IncrementalSGDRegressor({"learning_rate": 0.1})
        for _ in range(100):
            sgd.partial_fit([1.0, 2.0], 5.0)
        pred = sgd.predict([1.0, 2.0])
        assert abs(pred - 5.0) < 1.0

    def test_weights_converge(self) -> None:
        sgd = IncrementalSGDRegressor({"learning_rate": 0.001, "regularization": 0.01})
        for i in range(1000):
            x = [float(i % 10), float((i + 1) % 10)]
            y = 2.0 * x[0] + 3.0 * x[1] + 1.0
            sgd.partial_fit(x, y)
        pred = sgd.predict([5.0, 5.0])
        assert abs(pred - 26.0) < 3.0

    def test_uninitialized_predict(self) -> None:
        sgd = IncrementalSGDRegressor({})
        assert sgd.predict([1.0, 2.0]) == 0.0

    def test_get_weights(self) -> None:
        sgd = IncrementalSGDRegressor({})
        sgd.partial_fit([1.0], 2.0)
        state = sgd.get_weights()
        assert state["n_samples"] == 1
        assert len(state["weights"]) == 1

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            IncrementalSGDRegressor(None)


class TestAutoFeatureEngineer:
    """测试自动特征工程"""

    def test_normal_transform(self) -> None:
        fe = AutoFeatureEngineer({"max_degree": 2})
        result = fe.fit_transform([1.0, 2.0])
        assert len(result) > 2

    def test_polynomial_features(self) -> None:
        fe = AutoFeatureEngineer({"max_degree": 2, "include_interactions": False})
        result = fe.fit_transform([2.0])
        assert 4.0 in result

    def test_interaction_features(self) -> None:
        fe = AutoFeatureEngineer({"max_degree": 2, "include_interactions": True})
        result = fe.fit_transform([2.0, 3.0])
        assert 6.0 in result

    def test_standardize(self) -> None:
        fe = AutoFeatureEngineer({})
        result = fe.fit_transform([1.0, 2.0])
        assert all(isinstance(v, float) for v in result)

    def test_update_scaler_no_raise(self) -> None:
        """update_scaler 不应抛出 AttributeError（n_samples 未初始化 bug 回归）"""
        fe = AutoFeatureEngineer({})
        # 修复前：此调用会抛 AttributeError: 'AutoFeatureEngineer' object has no attribute 'n_samples'
        fe.update_scaler([1.0, 2.0, 3.0])
        fe.update_scaler([2.0, 3.0, 4.0])
        fe.update_scaler([3.0, 4.0, 5.0])

    def test_update_scaler_increments_n_samples(self) -> None:
        fe = AutoFeatureEngineer({})
        assert fe.n_samples == 0
        fe.update_scaler([1.0, 2.0])
        assert fe.n_samples == 1
        fe.update_scaler([2.0, 3.0])
        assert fe.n_samples == 2

    def test_get_scaler_state_empty(self) -> None:
        fe = AutoFeatureEngineer({})
        state = fe.get_scaler_state()
        assert state["n_samples"] == 0
        assert state["mean"] == []
        assert state["std"] == []

    def test_get_scaler_state_single_sample(self) -> None:
        fe = AutoFeatureEngineer({})
        fe.update_scaler([1.0, 2.0, 3.0])
        state = fe.get_scaler_state()
        assert state["n_samples"] == 1
        assert state["mean"] == [1.0, 2.0, 3.0]
        # 单样本时方差为 0
        assert state["std"] == [0.0, 0.0, 0.0]

    def test_get_scaler_state_multiple_samples(self) -> None:
        fe = AutoFeatureEngineer({})
        fe.update_scaler([1.0, 2.0])
        fe.update_scaler([3.0, 4.0])
        fe.update_scaler([5.0, 6.0])
        state = fe.get_scaler_state()
        assert state["n_samples"] == 3
        # 均值 = (1+3+5)/3 = 3.0, (2+4+6)/3 = 4.0
        assert abs(state["mean"][0] - 3.0) < 1e-9
        assert abs(state["mean"][1] - 4.0) < 1e-9
        # 标准差应为正数
        assert state["std"][0] > 0
        assert state["std"][1] > 0

    def test_get_scaler_state_std_correctness(self) -> None:
        """验证标准差计算正确性"""
        fe = AutoFeatureEngineer({})
        # 样本 [1, 2, 3, 4, 5]，均值 3，方差 2.5，标准差 ≈ 1.5811
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            fe.update_scaler([v])
        state = fe.get_scaler_state()
        assert abs(state["mean"][0] - 3.0) < 1e-9
        expected_std = (2.5 ** 0.5)
        assert abs(state["std"][0] - expected_std) < 1e-6


class TestModelSelector:
    """测试模型选择器"""

    def test_aic_selection(self) -> None:
        ms = ModelSelector({"criterion": "aic"})
        score = ms.evaluate("m1", sse=10.0, n_params=3, n_samples=100)
        assert isinstance(score, float)
        assert score != float('inf')

    def test_bic_selection(self) -> None:
        ms = ModelSelector({"criterion": "bic"})
        score = ms.evaluate("m1", sse=10.0, n_params=3, n_samples=100)
        assert isinstance(score, float)
        assert score != float('inf')

    def test_select_best(self) -> None:
        ms = ModelSelector({})
        candidates = [
            {"name": "m1", "score": 10.0},
            {"name": "m2", "score": 5.0},
        ]
        assert ms.select_best(candidates) == "m2"

    def test_select_best_empty(self) -> None:
        ms = ModelSelector({})
        assert ms.select_best([]) is None

    def test_invalid_n_samples(self) -> None:
        ms = ModelSelector({})
        score = ms.evaluate("m1", sse=10.0, n_params=10, n_samples=5)
        assert score == float('inf')


class TestOverfitDetector:
    """测试过拟合检测器"""

    def test_normal_no_overfit(self) -> None:
        od = OverfitDetector({"threshold": 0.5})
        for _ in range(20):
            od.add_errors(0.1, 0.12)
        assert od.is_overfitting() is False

    def test_overfit_detected(self) -> None:
        od = OverfitDetector({"threshold": 0.1})
        for _ in range(20):
            od.add_errors(0.05, 0.2)
        assert od.is_overfitting() is True

    def test_not_enough_data(self) -> None:
        od = OverfitDetector({})
        od.add_errors(0.1, 1.0)
        assert od.is_overfitting() is False

    def test_get_gap(self) -> None:
        od = OverfitDetector({})
        for _ in range(20):
            od.add_errors(0.1, 0.2)
        gap = od.get_gap()
        assert abs(gap - 0.1) < 0.01


class TestAdaptiveLearner:
    """测试自适应学习器"""

    def test_normal_learn(self) -> None:
        learner = AdaptiveLearner({})
        record = learner.learn([1.0, 2.0], 5.0)
        assert "train_error" in record

    def test_predict_after_learn(self) -> None:
        learner = AdaptiveLearner({"sgd": {"learning_rate": 0.01}})
        for _ in range(200):
            learner.learn([1.0, 2.0], 5.0)
        pred = learner.predict([1.0, 2.0])
        assert abs(pred - 5.0) < 1.0

    def test_overfit_detection(self) -> None:
        learner = AdaptiveLearner({"overfit": {"threshold": 0.01}})
        for _ in range(20):
            learner.learn([1.0], 2.0, validation_target=5.0)
        assert learner.check_overfitting() is True

    def test_learning_state(self) -> None:
        learner = AdaptiveLearner({})
        learner.learn([1.0], 2.0)
        state = learner.get_learning_state()
        assert "sgd" in state
        assert "history_size" in state

    def test_get_state_unified(self) -> None:
        """统一接口 get_state()"""
        learner = AdaptiveLearner({})
        learner.learn([1.0, 2.0], 5.0)
        state = learner.get_state()
        assert "sgd" in state
        assert "scaler" in state
        assert "overfit_gap" in state
        assert "history_size" in state
        assert "is_overfitting" in state
        assert "n_samples" in state
        assert state["n_samples"] == 1

    def test_reset_unified(self) -> None:
        """统一接口 reset() 应清空运行时状态"""
        learner = AdaptiveLearner({})
        for _ in range(10):
            learner.learn([1.0], 2.0)
        assert learner.sgd.n_samples == 10
        assert len(learner.history) == 10
        learner.reset()
        # 重置后状态归零
        assert learner.sgd.n_samples == 0
        assert len(learner.history) == 0
        assert learner.sgd.weights is None

    def test_health_check_unified(self) -> None:
        """统一接口 health_check()"""
        learner = AdaptiveLearner({})
        assert learner.health_check() is True
        # 学习后仍健康
        learner.learn([1.0], 2.0)
        assert learner.health_check() is True

    def test_health_check_after_reset(self) -> None:
        learner = AdaptiveLearner({})
        learner.learn([1.0], 2.0)
        learner.reset()
        assert learner.health_check() is True

    def test_history_bounded(self) -> None:
        learner = AdaptiveLearner({"max_history": 5})
        for _ in range(10):
            learner.learn([1.0], 2.0)
        assert len(learner.history) <= 5

    def test_invalid_format_none(self) -> None:
        with pytest.raises(ValidationError):
            AdaptiveLearner(None)


class TestErrorHandlingHardened:
    """D1: 错误处理加固测试"""

    def test_partial_fit_empty_x_raises(self) -> None:
        sgd = IncrementalSGDRegressor({})
        with pytest.raises(ValidationError, match="x must not be empty"):
            sgd.partial_fit([], 1.0)

    def test_partial_fit_invalid_y_raises(self) -> None:
        sgd = IncrementalSGDRegressor({})
        with pytest.raises(ValidationError):
            sgd.partial_fit([1.0], "not_a_number")

    def test_partial_fit_dimension_mismatch_raises(self) -> None:
        sgd = IncrementalSGDRegressor({})
        sgd.partial_fit([1.0, 2.0], 1.0)
        with pytest.raises(ValidationError, match="length"):
            sgd.partial_fit([1.0], 1.0)

    def test_predict_dimension_mismatch_raises(self) -> None:
        sgd = IncrementalSGDRegressor({})
        sgd.partial_fit([1.0, 2.0], 1.0)
        with pytest.raises(ValidationError, match="length"):
            sgd.predict([1.0])

    def test_predict_empty_returns_zero(self) -> None:
        sgd = IncrementalSGDRegressor({})
        assert sgd.predict([]) == 0.0

    def test_fit_transform_empty_raises(self) -> None:
        fe = AutoFeatureEngineer({})
        with pytest.raises(ValidationError, match="x must not be empty"):
            fe.fit_transform([])

    def test_update_scaler_empty_raises(self) -> None:
        fe = AutoFeatureEngineer({})
        with pytest.raises(ValidationError, match="features must not be empty"):
            fe.update_scaler([])

    def test_update_scaler_dimension_mismatch_raises(self) -> None:
        fe = AutoFeatureEngineer({})
        fe.update_scaler([1.0, 2.0])
        with pytest.raises(ValidationError, match="length"):
            fe.update_scaler([1.0])

    def test_learn_empty_features_raises(self) -> None:
        learner = AdaptiveLearner({})
        with pytest.raises(ValidationError, match="features must not be empty"):
            learner.learn([], 1.0)

    def test_learn_invalid_target_raises(self) -> None:
        learner = AdaptiveLearner({})
        with pytest.raises(ValidationError):
            learner.learn([1.0], "invalid")

    def test_learner_predict_empty_returns_zero(self) -> None:
        learner = AdaptiveLearner({})
        assert learner.predict([]) == 0.0

    def test_model_selector_evaluate_zero_sse(self) -> None:
        """sse=0 不应导致 log(0) 异常"""
        selector = ModelSelector({"criterion": "aic"})
        score = selector.evaluate("model", 0.0, 2, 10)
        assert isinstance(score, float)
        assert not (score != score)  # not NaN

    def test_model_selector_evaluate_negative_sse(self) -> None:
        """sse<0 不应导致 log(negative) 异常"""
        selector = ModelSelector({"criterion": "bic"})
        score = selector.evaluate("model", -1.0, 2, 10)
        assert isinstance(score, float)
        assert not (score != score)  # not NaN

    def test_health_check_logs_warning_on_failure(self, caplog) -> None:
        """health_check 失败时应记录 warning"""
        import logging
        learner = AdaptiveLearner({})
        # 模拟不健康状态
        learner.sgd.n_samples = -1
        with caplog.at_level(logging.WARNING):
            result = learner.health_check()
        assert result is False
        assert any("health_check fail" in r.message for r in caplog.records)


class TestEdgeCases:
    """D2: 边界条件与异常输入测试"""

    def test_partial_fit_single_element(self) -> None:
        """单个元素的 partial_fit"""
        sgd = IncrementalSGDRegressor({})
        sgd.partial_fit([1.0], 2.0)
        assert sgd.n_samples == 1
        assert len(sgd.weights) == 1

    def test_partial_fit_large_values(self) -> None:
        """极大值不导致溢出"""
        sgd = IncrementalSGDRegressor({"learning_rate": 0.001})
        sgd.partial_fit([1e10], 1e10)
        assert sgd.n_samples == 1
        import math
        assert math.isfinite(sgd.bias)

    def test_partial_fit_negative_values(self) -> None:
        """负值正常处理"""
        sgd = IncrementalSGDRegressor({})
        sgd.partial_fit([-1.0, -2.0], -5.0)
        pred = sgd.predict([-1.0, -2.0])
        assert isinstance(pred, float)

    def test_predict_no_training_returns_zero(self) -> None:
        """未训练时 predict 返回 0"""
        sgd = IncrementalSGDRegressor({})
        assert sgd.predict([1.0, 2.0]) == 0.0

    def test_fit_transform_single_element(self) -> None:
        """单个元素的 fit_transform"""
        fe = AutoFeatureEngineer({})
        result = fe.fit_transform([5.0])
        assert len(result) >= 2  # bias + feature

    def test_update_scaler_many_samples(self) -> None:
        """大量样本更新 scaler 不出错"""
        fe = AutoFeatureEngineer({})
        for i in range(100):
            fe.update_scaler([float(i), float(i * 2)])
        state = fe.get_scaler_state()
        assert state["n_samples"] == 100
        assert len(state["mean"]) == 2
        assert len(state["std"]) == 2

    def test_model_selector_n_samples_equals_n_params(self) -> None:
        """n_samples == n_params 边界条件返回 inf"""
        selector = ModelSelector({})
        score = selector.evaluate("model", 1.0, 5, 5)
        assert score == float('inf')

    def test_model_selector_unknown_criterion(self) -> None:
        """未知 criterion 回退到 sse/n_samples"""
        selector = ModelSelector({"criterion": "unknown"})
        score = selector.evaluate("model", 10.0, 2, 10)
        assert score == 1.0

    def test_overfit_detector_boundary_ten_errors(self) -> None:
        """恰好 10 个误差时开始检测"""
        detector = OverfitDetector({"threshold": 0.01})
        for _ in range(10):
            detector.add_errors(0.1, 0.5)
        assert detector.is_overfitting() is True

    def test_overfit_detector_nine_errors(self) -> None:
        """9 个误差时不检测"""
        detector = OverfitDetector({"threshold": 0.01})
        for _ in range(9):
            detector.add_errors(0.1, 0.5)
        assert detector.is_overfitting() is False

    def test_overfit_detector_empty_get_gap(self) -> None:
        """空误差列表 get_gap 返回 0"""
        detector = OverfitDetector({})
        assert detector.get_gap() == 0.0

    def test_select_best_empty_candidates(self) -> None:
        """空候选列表返回 None"""
        selector = ModelSelector({})
        assert selector.select_best([]) is None

    def test_select_best_with_inf_score(self) -> None:
        """含 inf 分数的候选不选中"""
        selector = ModelSelector({})
        candidates = [
            {"name": "a", "score": float('inf')},
            {"name": "b", "score": 1.0},
        ]
        assert selector.select_best(candidates) == "b"

    def test_learner_learn_with_validation_target(self) -> None:
        """带验证目标的 learn"""
        learner = AdaptiveLearner({})
        record = learner.learn([1.0], 2.0, validation_target=3.0)
        assert "val_error" in record
        assert record["val_error"] != record["train_error"]

    def test_learner_learn_without_validation_target(self) -> None:
        """不带验证目标时 val_error == train_error"""
        learner = AdaptiveLearner({})
        record = learner.learn([1.0], 2.0)
        assert record["val_error"] == record["train_error"]

    def test_learner_predict_before_learn(self) -> None:
        """未学习时 predict 返回 0"""
        learner = AdaptiveLearner({})
        assert learner.predict([1.0]) == 0.0


class TestConfigValidation:
    """D4: 构造时配置验证测试"""

    def test_sgd_zero_learning_rate_raises(self) -> None:
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            IncrementalSGDRegressor({"learning_rate": 0.0})

    def test_sgd_negative_learning_rate_raises(self) -> None:
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            IncrementalSGDRegressor({"learning_rate": -0.1})

    def test_sgd_negative_regularization_raises(self) -> None:
        with pytest.raises(ValidationError, match="regularization must be non-negative"):
            IncrementalSGDRegressor({"regularization": -0.001})

    def test_sgd_valid_zero_regularization(self) -> None:
        """正则化为 0 是合法的"""
        sgd = IncrementalSGDRegressor({"regularization": 0.0})
        assert sgd.regularization == 0.0

    def test_featurizer_zero_max_degree_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_degree must be >= 1"):
            AutoFeatureEngineer({"max_degree": 0})

    def test_featurizer_negative_max_degree_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_degree must be >= 1"):
            AutoFeatureEngineer({"max_degree": -1})

    def test_featurizer_valid_degree_one(self) -> None:
        """max_degree=1 是合法的"""
        fe = AutoFeatureEngineer({"max_degree": 1})
        assert fe.max_degree == 1


class TestStateConsistency:
    """D5: 状态一致性与幂等性测试"""

    def test_reset_restores_initial_state(self) -> None:
        """reset 后 sgd/featurizer/overfit/history 均回到初始状态。"""
        learner = AdaptiveLearner({})
        # 产生状态
        learner.learn([1.0, 2.0, 3.0], 1.0, validation_target=0.5)
        learner.learn([2.0, 3.0, 4.0], 0.5, validation_target=0.3)
        assert len(learner.history) > 0
        assert learner.sgd.n_samples > 0
        # reset
        learner.reset()
        assert len(learner.history) == 0
        assert learner.sgd.n_samples == 0
        assert learner.overfit.get_gap() == 0.0

    def test_reset_state_equals_fresh_instance(self) -> None:
        """reset 后 get_state 等价于新构造实例。"""
        config = {"sgd": {"learning_rate": 0.1}}
        mod1 = AdaptiveLearner(config)
        mod2 = AdaptiveLearner(config)
        mod1.learn([1.0, 2.0], 1.0)
        mod1.reset()
        assert mod1.get_state() == mod2.get_state()

    def test_reset_idempotent(self) -> None:
        """连续调用 reset 两次，状态一致。"""
        learner = AdaptiveLearner({})
        learner.learn([1.0, 2.0], 1.0)
        learner.reset()
        state1 = learner.get_state()
        learner.reset()
        state2 = learner.get_state()
        assert state1 == state2

    def test_health_check_idempotent(self) -> None:
        """连续调用 health_check 多次，结果一致且无副作用。"""
        learner = AdaptiveLearner({})
        learner.learn([1.0, 2.0], 1.0)
        state_before = learner.get_state()
        r1 = learner.health_check()
        r2 = learner.health_check()
        r3 = learner.health_check()
        state_after = learner.get_state()
        assert r1 == r2 == r3 is True
        assert state_before == state_after

    def test_get_state_idempotent(self) -> None:
        """连续调用 get_state 多次，结果一致且不改变内部状态。"""
        learner = AdaptiveLearner({})
        learner.learn([1.0, 2.0], 1.0)
        s1 = learner.get_state()
        s2 = learner.get_state()
        s3 = learner.get_state()
        assert s1 == s2 == s3
