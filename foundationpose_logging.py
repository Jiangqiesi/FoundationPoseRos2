"""
服务端文件日志工具 — 将 ROS2 节点的 get_logger() 输出同时写入日志文件。

每次启动生成以时间戳命名的日志文件，感知和控制分别存放在不同目录中。

用法:
    from foundationpose_logging import setup_file_logging
    # 在 Node.__init__ 末尾调用:
    setup_file_logging(self, "logs/perception")   # → logs/perception/2026-03-18_14-30-00.log
    setup_file_logging(self, "logs/realman")       # → logs/realman/2026-03-18_14-30-00.log
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any


def _make_log_path(log_dir: str) -> str:
    """根据当前时间戳在指定目录下生成日志文件路径。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(log_dir, f"{timestamp}.log")


def _make_file_logger(
    name: str,
    log_path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """创建带 RotatingFileHandler 的 Python logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return logger

    handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class _DualLogger:
    """代理对象：同时调用 ROS2 logger 和 Python file logger。

    对外接口与 rclpy RcutilsLogger 一致（info/warn/error/debug/fatal），
    所有 self.get_logger().xxx(...) 调用无需修改。
    """

    def __init__(self, ros_logger: Any, file_logger: logging.Logger) -> None:
        self._ros = ros_logger
        self._file = file_logger

    def debug(self, message: str, **kwargs: Any) -> None:
        self._ros.debug(message, **kwargs)
        self._file.debug(message)

    def info(self, message: str, **kwargs: Any) -> None:
        self._ros.info(message, **kwargs)
        self._file.info(message)

    def warn(self, message: str, **kwargs: Any) -> None:
        self._ros.warn(message, **kwargs)
        self._file.warning(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._ros.warning(message, **kwargs)
        self._file.warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        self._ros.error(message, **kwargs)
        self._file.error(message)

    def fatal(self, message: str, **kwargs: Any) -> None:
        self._ros.fatal(message, **kwargs)
        self._file.critical(message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ros, name)


def setup_file_logging(
    node: Any,
    log_dir: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """为 ROS2 节点启用文件日志。

    替换 node.get_logger() 返回的对象为 _DualLogger，
    使所有后续 self.get_logger().info/warn/error 同时写入文件。

    每次调用会在 log_dir 下生成以时间戳命名的新日志文件。

    Args:
        node: rclpy.node.Node 实例
        log_dir: 日志目录路径，如 "logs/perception"
        max_bytes: 单个日志文件最大字节数，默认 10MB
        backup_count: 保留的旧日志文件数量，默认 5
    """
    log_path = _make_log_path(log_dir)

    ros_logger = node.get_logger()
    logger_name = f"file_logger.{log_dir}"
    file_logger = _make_file_logger(
        logger_name, log_path, max_bytes=max_bytes, backup_count=backup_count
    )

    dual = _DualLogger(ros_logger, file_logger)

    # 猴子补丁 get_logger() 返回 dual logger
    node.get_logger = lambda: dual

    start_msg = f"===== 日志开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====="
    file_logger.info(start_msg)
    file_logger.info(f"日志文件: {log_path}")
