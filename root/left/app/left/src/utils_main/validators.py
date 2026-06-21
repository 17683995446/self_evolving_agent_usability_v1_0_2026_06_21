"""严格输入验证器"""
from typing import Any, Dict, Type

from root.left.app.left.src.utils_main.exceptions import ValidationError, ConfigurationError


class Validator:
    """基础验证器"""

    @staticmethod
    def require_type(value: Any, expected_type: Type, name: str) -> None:
        if not isinstance(value, expected_type):
            # 支持元组类型（如 (int, float)），生成可读类型名
            if isinstance(expected_type, tuple):
                type_name = " or ".join(t.__name__ for t in expected_type)
            else:
                type_name = expected_type.__name__
            raise ValidationError(
                f"{name} must be {type_name}",
                field=name,
                expected_range=type_name,
                actual_value=value
            )

    @staticmethod
    def require_non_empty(value: Any, name: str) -> None:
        if not value:
            raise ValidationError(
                f"{name} must be non-empty",
                field=name,
                actual_value=value
            )

    @staticmethod
    def require_range(value: float, min_v: float, max_v: float, name: str) -> None:
        if not (min_v <= value <= max_v):
            raise ValidationError(
                f"{name} must be in [{min_v}, {max_v}]",
                field=name,
                expected_range=f"[{min_v}, {max_v}]",
                actual_value=value
            )

    @staticmethod
    def require_positive(value: float, name: str) -> None:
        if value <= 0:
            raise ValidationError(
                f"{name} must be positive",
                field=name,
                expected_range="> 0",
                actual_value=value
            )

    @staticmethod
    def require_non_negative(value: float, name: str) -> None:
        if value < 0:
            raise ValidationError(
                f"{name} must be non-negative",
                field=name,
                expected_range=">= 0",
                actual_value=value
            )


class ConfigValidator:
    """配置验证器"""

    REQUIRED_PHASE1_KEYS = {"sensory", "drive", "prediction", "adjustment"}
    REQUIRED_PHASE2_KEYS = {"task_pool", "decision", "memory"}
    REQUIRED_PHASE3_KEYS = {"evolution"}
    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        Validator.require_type(config, dict, "config")
        for key in ["phase1", "phase2", "phase3"]:
            if key not in config:
                raise ConfigurationError(
                    f"Missing required config section: {key}",
                    config_key=key
                )
            Validator.require_type(config[key], dict, key)
        cls._validate_phase1(config["phase1"])
        cls._validate_phase2(config["phase2"])
        cls._validate_phase3(config["phase3"])
        cls._validate_global(config)

    @classmethod
    def _validate_phase1(cls, config: Dict[str, Any]) -> None:
        missing = cls.REQUIRED_PHASE1_KEYS - set(config.keys())
        if missing:
            raise ConfigurationError(
                f"Phase1 missing keys: {missing}",
                config_key="phase1"
            )
        for key in cls.REQUIRED_PHASE1_KEYS:
            Validator.require_type(config[key], dict, f"phase1.{key}")

    @classmethod
    def _validate_phase2(cls, config: Dict[str, Any]) -> None:
        missing = cls.REQUIRED_PHASE2_KEYS - set(config.keys())
        if missing:
            raise ConfigurationError(
                f"Phase2 missing keys: {missing}",
                config_key="phase2"
            )
        for key in cls.REQUIRED_PHASE2_KEYS:
            Validator.require_type(config[key], dict, f"phase2.{key}")

    @classmethod
    def _validate_phase3(cls, config: Dict[str, Any]) -> None:
        missing = cls.REQUIRED_PHASE3_KEYS - set(config.keys())
        if missing:
            raise ConfigurationError(
                f"Phase3 missing keys: {missing}",
                config_key="phase3"
            )
        Validator.require_type(config["evolution"], dict, "phase3.evolution")

    @classmethod
    def _validate_global(cls, config: Dict[str, Any]) -> None:
        if "loop_interval" in config:
            value = config["loop_interval"]
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigurationError(
                    "loop_interval must be a non-negative number",
                    config_key="loop_interval",
                    actual_value=value
                )
        if "max_loops" in config:
            value = config["max_loops"]
            if not isinstance(value, int) or value < 0:
                raise ConfigurationError(
                    "max_loops must be a non-negative integer",
                    config_key="max_loops",
                    actual_value=value
                )
        if "log_level" in config:
            value = config["log_level"]
            if not isinstance(value, str) or value.upper() not in cls.VALID_LOG_LEVELS:
                raise ConfigurationError(
                    f"log_level must be one of {cls.VALID_LOG_LEVELS}",
                    config_key="log_level",
                    actual_value=value
                )
