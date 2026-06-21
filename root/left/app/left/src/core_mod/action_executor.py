"""真实动作执行器

将任务负载转换为可执行动作，支持 shell、HTTP、文件操作，并内置安全护栏。

增强能力：
1. 执行历史记录：每次 execute() 记录一条摘要，支持查询与统计
2. 成功率统计：按 action_type 与全局维度计算成功/失败计数与比率
"""
import shlex
import subprocess
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Final, List, Optional

from root.left.app.left.src.core_mod.approval_gate import ApprovalGate
from root.left.app.left.src.utils_main.exceptions import SafetyBoundaryException, ValidationError
from root.left.app.left.src.utils_main.logger import logger
from root.left.app.left.src.utils_main.validators import Validator


class ActionResult:
    """动作执行结果"""

    def __init__(
        self,
        success: bool,
        action_type: str,
        stdout: str = "",
        stderr: str = "",
        returncode: Optional[int] = None,
        data: Optional[Any] = None,
        error: str = "",
    ) -> None:
        self.success: bool = success
        self.action_type: str = action_type
        self.stdout: str = stdout
        self.stderr: str = stderr
        self.returncode: Optional[int] = returncode
        self.data: Any = data
        self.error: str = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_type": self.action_type,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "data": self.data,
            "error": self.error,
        }


class ActionExecutor:
    """动作执行器"""

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.debug("[ActionExecutor.__init__] start")
        Validator.require_type(config, dict, "config")
        self.config: Final[Dict[str, Any]] = config
        self.dry_run: bool = bool(config.get("dry_run", True))
        self.timeout: float = float(config.get("timeout", 30.0))
        self.allowed_shell_prefixes: List[str] = list(
            config.get("allowed_shell_prefixes", ["echo", "cat", "ls", "pwd", "df", "du", "ps", "curl", "python"])
        )
        self.blocked_shell_patterns: List[str] = list(
            config.get("blocked_shell_patterns", ["rm -rf /", "mkfs", ":(){ :|:& };:", "> /dev/sda"])
        )
        self.allowed_http_hosts: List[str] = list(config.get("allowed_http_hosts", ["localhost", "127.0.0.1"]))
        self.blocked_file_paths: List[str] = list(config.get("blocked_file_paths", ["/etc/passwd", "/etc/shadow"]))
        self.approval_gate = ApprovalGate(config.get("approval_gate", {"mode": "auto_pass"}))
        # 执行历史记录配置
        self.max_history: int = int(config.get("max_history", 100))
        self.history: List[Dict[str, Any]] = []
        logger.info(f"ActionExecutor initialized dry_run={self.dry_run} timeout={self.timeout}", "action")

    def execute(self, payload: Dict[str, Any]) -> ActionResult:
        logger.debug(f"[ActionExecutor.execute] payload={payload}")
        Validator.require_type(payload, dict, "payload")
        action_type = payload.get("action_type", "noop")
        logger.info(f"[ActionExecutor.execute] action_type={action_type}", "action")
        start_ts = time.time()
        try:
            self.approval_gate.check(action_type, payload)
        except SafetyBoundaryException as exc:
            logger.warning(f"[ActionExecutor.execute] approval rejected: {exc}", "action")
            result = ActionResult(success=False, action_type=action_type, error=str(exc))
            self._record_history(result, start_ts, payload)
            return result
        if action_type == "shell":
            result = self._execute_shell(payload)
        elif action_type == "http_get":
            result = self._execute_http_get(payload)
        elif action_type == "http_post":
            result = self._execute_http_post(payload)
        elif action_type == "file_write":
            result = self._execute_file_write(payload)
        elif action_type == "file_read":
            result = self._execute_file_read(payload)
        elif action_type == "file_delete":
            result = self._execute_file_delete(payload)
        elif action_type == "noop":
            result = ActionResult(success=True, action_type="noop", stdout="noop executed")
        else:
            result = ActionResult(
                success=False,
                action_type=action_type,
                error=f"Unknown action_type: {action_type}",
            )
        self._record_history(result, start_ts, payload)
        return result

    def _execute_shell(self, payload: Dict[str, Any]) -> ActionResult:
        command = payload.get("command", "")
        logger.info(f"[ActionExecutor._execute_shell] command={command}", "action")
        if not isinstance(command, str) or not command.strip():
            return ActionResult(success=False, action_type="shell", error="Empty command")
        try:
            self._validate_shell_command(command)
        except SafetyBoundaryException as exc:
            logger.warning(f"[ActionExecutor._execute_shell] blocked: {exc}", "action")
            return ActionResult(success=False, action_type="shell", error=str(exc))
        if self.dry_run:
            logger.info(f"[ActionExecutor._execute_shell] dry_run: {command}", "action")
            return ActionResult(success=True, action_type="shell", stdout=f"[dry_run] {command}")
        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return ActionResult(
                success=result.returncode == 0,
                action_type="shell",
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, action_type="shell", error=f"Timeout after {self.timeout}s")
        except Exception as exc:
            return ActionResult(success=False, action_type="shell", error=str(exc))

    def _validate_shell_command(self, command: str) -> None:
        for pattern in self.blocked_shell_patterns:
            if pattern in command:
                raise SafetyBoundaryException(
                    "Blocked shell pattern detected",
                    boundary="shell_command",
                    attempted_action=command,
                )
        parts = shlex.split(command)
        if not parts:
            raise ValidationError("Empty shell command", field="command")
        first = parts[0]
        if self.allowed_shell_prefixes and first not in self.allowed_shell_prefixes:
            raise SafetyBoundaryException(
                f"Shell command not in allowed list: {first}",
                boundary="shell_command",
                attempted_action=command,
            )

    def _execute_http_get(self, payload: Dict[str, Any]) -> ActionResult:
        url = payload.get("url", "")
        logger.info(f"[ActionExecutor._execute_http_get] url={url}", "action")
        if not isinstance(url, str) or not url.strip():
            return ActionResult(success=False, action_type="http_get", error="Empty URL")
        try:
            self._validate_http_url(url)
        except SafetyBoundaryException as exc:
            return ActionResult(success=False, action_type="http_get", error=str(exc))
        if self.dry_run:
            return ActionResult(success=True, action_type="http_get", stdout=f"[dry_run] GET {url}")
        try:
            with urllib.request.urlopen(url, timeout=int(self.timeout)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return ActionResult(success=True, action_type="http_get", stdout=body, returncode=resp.status, data=body[:1000])
        except urllib.error.HTTPError as exc:
            return ActionResult(success=False, action_type="http_get", error=f"HTTP {exc.code}")
        except Exception as exc:
            return ActionResult(success=False, action_type="http_get", error=str(exc))

    def _execute_http_post(self, payload: Dict[str, Any]) -> ActionResult:
        url = payload.get("url", "")
        body = payload.get("body", "")
        logger.info(f"[ActionExecutor._execute_http_post] url={url}", "action")
        if not isinstance(url, str) or not url.strip():
            return ActionResult(success=False, action_type="http_post", error="Empty URL")
        try:
            self._validate_http_url(url)
        except SafetyBoundaryException as exc:
            return ActionResult(success=False, action_type="http_post", error=str(exc))
        if self.dry_run:
            return ActionResult(success=True, action_type="http_post", stdout=f"[dry_run] POST {url}")
        try:
            data_bytes = body.encode("utf-8") if isinstance(body, str) else str(body).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=int(self.timeout)) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return ActionResult(success=True, action_type="http_post", stdout=response_body, returncode=resp.status)
        except Exception as exc:
            return ActionResult(success=False, action_type="http_post", error=str(exc))

    def _validate_http_url(self, url: str) -> None:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if self.allowed_http_hosts and host not in self.allowed_http_hosts:
            raise SafetyBoundaryException(
                f"HTTP host not in allowed list: {host}",
                boundary="http_host",
                attempted_action=url,
            )

    def _execute_file_write(self, payload: Dict[str, Any]) -> ActionResult:
        path = payload.get("path", "")
        content = payload.get("content", "")
        logger.info(f"[ActionExecutor._execute_file_write] path={path}", "action")
        try:
            self._validate_file_path(path)
        except (SafetyBoundaryException, ValidationError) as exc:
            return ActionResult(success=False, action_type="file_write", error=str(exc))
        if self.dry_run:
            return ActionResult(success=True, action_type="file_write", stdout=f"[dry_run] write {path}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ActionResult(success=True, action_type="file_write", stdout=f"Wrote {path}")
        except Exception as exc:
            return ActionResult(success=False, action_type="file_write", error=str(exc))

    def _execute_file_read(self, payload: Dict[str, Any]) -> ActionResult:
        path = payload.get("path", "")
        logger.info(f"[ActionExecutor._execute_file_read] path={path}", "action")
        try:
            self._validate_file_path(path)
        except (SafetyBoundaryException, ValidationError) as exc:
            return ActionResult(success=False, action_type="file_read", error=str(exc))
        if self.dry_run:
            return ActionResult(success=True, action_type="file_read", stdout=f"[dry_run] read {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ActionResult(success=True, action_type="file_read", stdout=content[:1000], data=content[:1000])
        except Exception as exc:
            return ActionResult(success=False, action_type="file_read", error=str(exc))

    def _execute_file_delete(self, payload: Dict[str, Any]) -> ActionResult:
        path = payload.get("path", "")
        logger.info(f"[ActionExecutor._execute_file_delete] path={path}", "action")
        try:
            self._validate_file_path(path)
        except (SafetyBoundaryException, ValidationError) as exc:
            return ActionResult(success=False, action_type="file_delete", error=str(exc))
        if self.dry_run:
            return ActionResult(success=True, action_type="file_delete", stdout=f"[dry_run] delete {path}")
        try:
            import os
            os.remove(path)
            return ActionResult(success=True, action_type="file_delete", stdout=f"Deleted {path}")
        except Exception as exc:
            return ActionResult(success=False, action_type="file_delete", error=str(exc))

    def get_state(self) -> Dict[str, Any]:
        """获取执行器状态"""
        stats = self.get_statistics()
        return {
            "dry_run": self.dry_run,
            "timeout": self.timeout,
            "allowed_shell_prefixes": self.allowed_shell_prefixes,
            "blocked_shell_patterns": self.blocked_shell_patterns,
            "allowed_http_hosts": self.allowed_http_hosts,
            "blocked_file_paths": self.blocked_file_paths,
            "history_size": len(self.history),
            "max_history": self.max_history,
            "total_executions": stats["total"],
            "success_count": stats["success"],
            "failure_count": stats["failure"],
            "success_rate": stats["success_rate"],
        }

    def reset(self) -> None:
        """重置执行器到初始状态（统一接口）。

        重置审批门状态与执行历史，保留配置项不变。
        """
        logger.info("[ActionExecutor.reset] resetting runtime state", "action")
        self.approval_gate.reset()
        self.history.clear()

    def health_check(self) -> bool:
        """健康检查（统一接口）。

        Returns:
            执行器是否健康：配置项非空且审批门可用
        """
        logger.debug("[ActionExecutor.health_check] called", "action")
        try:
            if not self.allowed_shell_prefixes:
                logger.warning("health_check fail: allowed_shell_prefixes is empty", "action")
                return False
            if self.timeout <= 0:
                logger.warning(f"health_check fail: timeout={self.timeout} <= 0", "action")
                return False
            if self.approval_gate is None:
                logger.warning("health_check fail: approval_gate is None", "action")
                return False
            if self.max_history < 0:
                logger.warning(f"health_check fail: max_history={self.max_history} < 0", "action")
                return False
            return True
        except Exception as exc:
            logger.warning(f"health_check exception: {exc}", "action")
            return False

    # ------------------------------------------------------------------
    # 执行历史与统计
    # ------------------------------------------------------------------

    def _record_history(self, result: ActionResult, start_ts: float, payload: Dict[str, Any]) -> None:
        """记录一次执行到历史。

        Args:
            result: 执行结果
            start_ts: 执行开始时间戳
            payload: 原始负载（仅记录 action_type，不记录敏感参数）
        """
        elapsed = time.time() - start_ts
        record: Dict[str, Any] = {
            "timestamp": start_ts,
            "action_type": result.action_type,
            "success": result.success,
            "returncode": result.returncode,
            "error": result.error,
            "elapsed": elapsed,
        }
        self.history.append(record)
        # 有界列表：超出上限时丢弃最旧记录
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取执行历史。

        Args:
            limit: 返回最近 N 条记录，None 表示全部

        Returns:
            历史记录列表（按时间升序）
        """
        if limit is None or limit < 0:
            return list(self.history)
        if limit == 0:
            return []
        return list(self.history[-limit:])

    def get_statistics(self) -> Dict[str, Any]:
        """计算执行统计信息。

        Returns:
            统计字典：total/success/failure/success_rate/by_action_type
        """
        total = len(self.history)
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "failure": 0,
                "success_rate": 0.0,
                "by_action_type": {},
            }
        success = sum(1 for r in self.history if r["success"])
        failure = total - success
        by_type: Dict[str, Dict[str, int]] = {}
        for r in self.history:
            at = r["action_type"]
            if at not in by_type:
                by_type[at] = {"total": 0, "success": 0, "failure": 0}
            by_type[at]["total"] += 1
            if r["success"]:
                by_type[at]["success"] += 1
            else:
                by_type[at]["failure"] += 1
        return {
            "total": total,
            "success": success,
            "failure": failure,
            "success_rate": success / total,
            "by_action_type": by_type,
        }

    def clear_history(self) -> None:
        """仅清空历史记录，不影响审批门状态。"""
        logger.debug("[ActionExecutor.clear_history] called", "action")
        self.history.clear()

    def _validate_file_path(self, path: str) -> None:
        if not isinstance(path, str) or not path.strip():
            raise ValidationError("Empty file path", field="path")
        for blocked in self.blocked_file_paths:
            if path.startswith(blocked) or path == blocked:
                raise SafetyBoundaryException(
                    f"File path blocked: {path}",
                    boundary="file_path",
                    attempted_action=path,
                )
