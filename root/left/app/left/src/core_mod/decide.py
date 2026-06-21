"""决策模块

精简日志策略：仅保留有信息量的日志（关键决策点与异常），移除逐行跟踪日志。
新增决策摘要：聚合 decision_history 提供探索/利用比例、任务选中率、资源利用率。

配置验证策略：概率与限制类配置在构造时校验，快速失败。
"""
import random
from typing import Any, Dict, Final, List, Optional

from root.left.app.left.src.utils_main.constants import DriveType, TaskStatus
from root.left.app.left.src.utils_main.exceptions import DecisionException, ValidationError
from root.left.app.left.src.utils_main.logger import logger
from root.left.app.left.src.utils_main.validators import Validator


class ExplorationController:
    """探索控制器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.base_ratio: float = float(config.get("base_exploration_ratio", 0.3))
        self.min_ratio: float = float(config.get("min_exploration_ratio", 0.1))
        self.max_ratio: float = float(config.get("max_exploration_ratio", 0.5))
        # 构造时校验：概率必须在 [0, 1] 且 min <= base <= max
        for name, val in [("base_exploration_ratio", self.base_ratio),
                          ("min_exploration_ratio", self.min_ratio),
                          ("max_exploration_ratio", self.max_ratio)]:
            if not (0.0 <= val <= 1.0):
                raise ValidationError(
                    f"{name} must be in [0, 1], got {val}",
                    field=name,
                )
        if self.min_ratio > self.base_ratio:
            raise ValidationError(
                f"min_exploration_ratio {self.min_ratio} > base_exploration_ratio {self.base_ratio}",
                field="min_exploration_ratio",
            )
        if self.base_ratio > self.max_ratio:
            raise ValidationError(
                f"base_exploration_ratio {self.base_ratio} > max_exploration_ratio {self.max_ratio}",
                field="base_exploration_ratio",
            )
        self.exploration_count: int = 0
        self.exploitation_count: int = 0
        logger.debug(
            f"ExplorationController init base={self.base_ratio} min={self.min_ratio} max={self.max_ratio}",
            "decision",
        )

    def should_explore(self, drive_state: Dict[str, Any]) -> bool:
        scenario = drive_state.get("scenario", "normal")
        if scenario == "emergency":
            logger.debug("should_explore emergency -> False", "decision")
            return False
        current_ratio = self._get_current_ratio()
        result = random.random() < current_ratio
        logger.debug(f"should_explore scenario={scenario} ratio={current_ratio:.3f} -> {result}", "decision")
        return result

    def _get_current_ratio(self) -> float:
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return self.base_ratio
        actual = self.exploration_count / total
        if actual < self.min_ratio:
            return self.base_ratio + 0.1
        if actual > self.max_ratio:
            return self.base_ratio - 0.1
        return self.base_ratio

    def record_choice(self, is_exploration: bool) -> None:
        if is_exploration:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1


class TaskSelector:
    """任务选择器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config

    def select(self, tasks: List[Any], drive_state: Dict[str, Any]) -> Optional[Any]:
        if not tasks:
            return None
        gaps = drive_state.get("gaps", {})
        priority_order = sorted(
            tasks,
            key=lambda t: self._task_priority(t, gaps),
            reverse=True
        )
        return priority_order[0]

    def _task_priority(self, task: Any, gaps: Dict[Any, float]) -> float:
        task_type = getattr(task, "task_type", None)
        base_priority = getattr(task, "priority", 0.5)
        gap = gaps.get(task_type, 0.5)
        return base_priority * (1 + gap)


class ResourceBalancer:
    """资源平衡器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.cpu_limit: float = float(config.get("cpu_limit", 20.0))
        self.memory_limit: float = float(config.get("memory_limit", 20.0))
        # 构造时校验：资源限制必须为正
        if self.cpu_limit <= 0:
            raise ValidationError(
                f"cpu_limit must be positive, got {self.cpu_limit}",
                field="cpu_limit",
            )
        if self.memory_limit <= 0:
            raise ValidationError(
                f"memory_limit must be positive, got {self.memory_limit}",
                field="memory_limit",
            )

    def check_limits(self, agent_cpu: float, agent_memory: float) -> bool:
        return agent_cpu <= self.cpu_limit and agent_memory <= self.memory_limit

    def get_available_capacity(self, agent_cpu: float, agent_memory: float) -> Dict[str, float]:
        return {
            "cpu": max(0.0, self.cpu_limit - agent_cpu),
            "memory": max(0.0, self.memory_limit - agent_memory),
        }


class DecisionModule:
    """决策模块主控"""

    def __init__(self, config: Dict[str, Any]) -> None:
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.exploration = ExplorationController(config.get("exploration", {}))
        self.task_selector = TaskSelector(config.get("task_selector", {}))
        self.resource_balancer = ResourceBalancer(config.get("resource_balancer", {}))
        self.decision_history: List[Dict[str, Any]] = []
        self.max_history: int = int(config.get("max_history", 100))
        logger.info(
            f"DecisionModule init max_history={self.max_history} "
            f"cpu_limit={self.resource_balancer.cpu_limit} "
            f"mem_limit={self.resource_balancer.memory_limit}",
            "decision",
        )

    def decide(
        self,
        task_pool: Any,
        drive_state: Dict[str, Any],
        performance_history: List[float],
        loop_count: int,
        agent_cpu: float,
        agent_memory: float
    ) -> Dict[str, Any]:
        Validator.require_type(drive_state, dict, "drive_state")
        Validator.require_type(performance_history, list, "performance_history")
        Validator.require_type(loop_count, int, "loop_count")
        self._check_resources(agent_cpu, agent_memory)
        is_explore = self.exploration.should_explore(drive_state)
        self.exploration.record_choice(is_explore)
        selected_task = self._select_task(task_pool, drive_state, is_explore)
        result = self._build_decision(loop_count, is_explore, selected_task, agent_cpu, agent_memory)
        logger.info(
            f"decide loop={loop_count} explore={is_explore} "
            f"task_selected={selected_task is not None} cpu={agent_cpu} mem={agent_memory}",
            "decision",
        )
        return result

    def _check_resources(self, agent_cpu: float, agent_memory: float) -> None:
        if not self.resource_balancer.check_limits(agent_cpu, agent_memory):
            logger.error(
                f"resource limits exceeded cpu={agent_cpu}>{self.resource_balancer.cpu_limit} "
                f"or mem={agent_memory}>{self.resource_balancer.memory_limit}",
                "decision",
            )
            raise DecisionException("Resource limits exceeded", task_count=0)

    def _select_task(
        self, task_pool: Any, drive_state: Dict[str, Any], is_explore: bool
    ) -> Optional[Any]:
        pending_tasks = task_pool.get_by_status(TaskStatus.PENDING)
        if not pending_tasks:
            return None
        if is_explore:
            return random.choice(pending_tasks)
        return self.task_selector.select(pending_tasks, drive_state)

    def _build_decision(
        self, loop_count: int, is_explore: bool,
        selected_task: Optional[Any], agent_cpu: float, agent_memory: float
    ) -> Dict[str, Any]:
        decision = {
            "loop": loop_count,
            "is_exploration": is_explore,
            "selected_task": selected_task.to_dict() if selected_task else None,
            "resource_capacity": self.resource_balancer.get_available_capacity(agent_cpu, agent_memory),
        }
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
        return decision

    def should_pause(self, drive_state: Dict[str, Any]) -> bool:
        gaps = drive_state.get("gaps", {})
        safe_gap = gaps.get(DriveType.SAFE, 0)
        return safe_gap > 0.7

    # ------------------------------------------------------------------
    # 决策摘要
    # ------------------------------------------------------------------

    def get_decision_summary(self) -> Dict[str, Any]:
        """生成决策历史摘要。

        Returns:
            摘要字典：total/exploration/exploitation/exploration_ratio/
            task_selected/task_none/task_selection_rate/avg_available_cpu/avg_available_memory
        """
        total = len(self.decision_history)
        if total == 0:
            return {
                "total": 0,
                "exploration": 0,
                "exploitation": 0,
                "exploration_ratio": 0.0,
                "task_selected": 0,
                "task_none": 0,
                "task_selection_rate": 0.0,
                "avg_available_cpu": 0.0,
                "avg_available_memory": 0.0,
            }
        exploration = sum(1 for d in self.decision_history if d["is_exploration"])
        exploitation = total - exploration
        task_selected = sum(1 for d in self.decision_history if d["selected_task"] is not None)
        task_none = total - task_selected
        cpu_values = [d["resource_capacity"]["cpu"] for d in self.decision_history]
        mem_values = [d["resource_capacity"]["memory"] for d in self.decision_history]
        return {
            "total": total,
            "exploration": exploration,
            "exploitation": exploitation,
            "exploration_ratio": exploration / total,
            "task_selected": task_selected,
            "task_none": task_none,
            "task_selection_rate": task_selected / total,
            "avg_available_cpu": sum(cpu_values) / total,
            "avg_available_memory": sum(mem_values) / total,
        }

    def get_state(self) -> Dict[str, Any]:
        """获取决策模块状态"""
        summary = self.get_decision_summary()
        return {
            "decision_history_len": len(self.decision_history),
            "max_history": self.max_history,
            "has_exploration": hasattr(self.exploration, "should_explore"),
            "has_task_selector": hasattr(self.task_selector, "select"),
            "has_resource_balancer": hasattr(self.resource_balancer, "check_limits"),
            "exploration_count": self.exploration.exploration_count,
            "exploitation_count": self.exploration.exploitation_count,
            "task_selection_rate": summary["task_selection_rate"],
        }

    def reset(self) -> None:
        """重置决策模块"""
        logger.debug("DecisionModule reset", "decision")
        self.decision_history.clear()
        self.exploration.exploration_count = 0
        self.exploration.exploitation_count = 0

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if self.exploration is None:
                logger.warning("health_check fail: exploration is None", "decision")
                return False
            if self.task_selector is None:
                logger.warning("health_check fail: task_selector is None", "decision")
                return False
            if self.resource_balancer is None:
                logger.warning("health_check fail: resource_balancer is None", "decision")
                return False
            if self.max_history < 0:
                logger.warning(f"health_check fail: max_history={self.max_history} < 0", "decision")
                return False
            return True
        except Exception as exc:
            logger.warning(f"health_check exception: {exc}", "decision")
            return False
