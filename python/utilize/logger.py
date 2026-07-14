# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# [vLLM Project]
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/logger.py
import datetime
import logging
import logging.handlers
import os
import sys
from collections.abc import Hashable
from functools import lru_cache, partial
from logging import Logger
from types import MethodType
from typing import Optional, cast


# 日志配置（可从 YAML 读取）
KSANA_LOG_DIR = "log/"
KSANA_LOG_FILE = "log/ksana_python.log"
KSANA_LOG_MAX_SIZE = 536870912  # 512MB
KSANA_LOG_BACKUP_COUNT = 2


def _parse_file_size(size_str) -> int:
    """
    解析人类可读的文件大小格式，如 "100mb", "1gb", "512KB"
    支持: B, KB, MB, GB (大小写不敏感)
    如果是纯数字，则直接作为字节数
    """
    if size_str is None:
        return 536870912  # 默认 512MB
    
    if isinstance(size_str, (int, float)):
        return int(size_str)
    
    size_str = str(size_str).strip().upper()
    if not size_str:
        return 536870912
    
    # 提取数字和单位
    import re
    match = re.match(r'^([\d.]+)\s*([A-Z]*)$', size_str)
    if not match:
        return 536870912
    
    value = float(match.group(1))
    unit = match.group(2)
    
    multipliers = {
        '': 1,
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    return int(value * multipliers.get(unit, 1))

KSANA_LOGGING_PREFIX = "[KSANA] "
_FORMAT = (f"{KSANA_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "[%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _get_log_level_from_env() -> str:
    """
    Get log level from KLLM_LOG_LEVEL environment variable.
    
    Supports C++ style log levels: DEBUG, INFO, WARNING, ERROR, FATAL
    Also supports comma-separated values like "INFO,ATTENTION,DEBUG"
    Returns the most verbose level found.
    """
    env_level = os.environ.get("KLLM_LOG_LEVEL", "INFO")
    
    # 解析逗号分隔的级别
    levels = [l.strip().upper() for l in env_level.split(",")]
    
    # C++ 到 Python 日志级别映射
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "FATAL": logging.CRITICAL,
        # 特殊级别（如 ATTENTION, SCHEDULER 等）视为 DEBUG
        "ATTENTION": logging.DEBUG,
        "COMMUNICATION": logging.DEBUG,
        "MOE": logging.DEBUG,
        "MODEL": logging.DEBUG,
        "SCHEDULER": logging.DEBUG,
        "MAIN": logging.DEBUG,
        "MULTI_BATCH": logging.DEBUG,
    }
    
    # 找到最详细的级别（数值最小）
    min_level = logging.INFO  # 默认 INFO
    for level in levels:
        if level in level_map:
            min_level = min(min_level, level_map[level])
    
    # 转换回字符串
    level_to_name = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "FATAL",
    }
    return level_to_name.get(min_level, "INFO")


# 从环境变量获取日志级别
KSANA_LOGGING_LEVEL = _get_log_level_from_env()

# 延迟初始化标志
_LOGGING_INITIALIZED = False


@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print info message only once."""
    logger.info(msg, *args, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print warning message only once."""
    logger.warning(msg, *args, stacklevel=2)


@lru_cache
def _print_error_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print error message only once."""
    logger.error(msg, *args, stacklevel=2)


class _KsanaLogger(Logger):
    """Enhanced logger with additional methods."""

    def info_once(self, msg: str, *args: Hashable) -> None:
        _print_info_once(self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        _print_warning_once(self, msg, *args)
    
    def error_once(self, msg: str, *args: Hashable) -> None:
        _print_error_once(self, msg, *args)


def _configure_logging() -> None:
    """Configure logging with current settings."""
    log_level = getattr(logging, KSANA_LOGGING_LEVEL, logging.INFO)
    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(KSANA_LOG_FILE), exist_ok=True)
    
    # 配置的 logger 名称列表
    logger_names = ['ksana', 'ksana_llm', 'utilize', 'multimodal', 'openaiapi']
    
    for logger_name in logger_names:
        log = logging.getLogger(logger_name)
        log.setLevel(log_level)
        log.propagate = False
        
        # 清除已有的 handlers
        log.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        log.addHandler(console_handler)
        
        # File handler (RotatingFileHandler)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=KSANA_LOG_FILE,
            maxBytes=KSANA_LOG_MAX_SIZE,
            backupCount=KSANA_LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        log.addHandler(file_handler)


def init_logger(name: str) -> _KsanaLogger:
    """
    Initialize a logger with enhanced functionality.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        An enhanced logger instance
    """
    logger = logging.getLogger(name)

    # Patch additional methods onto the logger instance
    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
        "error_once": _print_error_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_KsanaLogger, logger)


def update_log_config(log_dir: str = None,
                      max_file_size: int = None,
                      max_backup_count: int = None) -> None:
    """
    Update logging configuration from YAML settings.
    
    This should be called after parsing the YAML config file.
    
    Args:
        log_dir: Log directory path
        max_file_size: Maximum log file size in bytes
        max_backup_count: Maximum number of backup files
    """
    global KSANA_LOG_DIR, KSANA_LOG_FILE, KSANA_LOG_MAX_SIZE, KSANA_LOG_BACKUP_COUNT
    global _LOGGING_INITIALIZED
    
    if log_dir is not None:
        KSANA_LOG_DIR = log_dir
        if not KSANA_LOG_DIR.endswith('/'):
            KSANA_LOG_DIR += '/'
        KSANA_LOG_FILE = KSANA_LOG_DIR + "ksana_python.log"
    
    if max_file_size is not None:
        KSANA_LOG_MAX_SIZE = max_file_size
    
    if max_backup_count is not None:
        KSANA_LOG_BACKUP_COUNT = max_backup_count
    
    # 配置 logger
    _configure_logging()
    _LOGGING_INITIALIZED = True


def update_log_config_from_yaml(yaml_config: dict) -> None:
    """
    Update logging configuration from parsed YAML config.
    
    Args:
        yaml_config: Parsed YAML configuration dictionary
    """
    try:
        setting = yaml_config.get('setting', {})
        log_config = setting.get('log', {})
        
        # 支持人类可读格式，如 "512mb", "1gb"
        max_file_size = log_config.get('max_file_size')
        if max_file_size is not None:
            max_file_size = _parse_file_size(max_file_size)
        
        update_log_config(
            log_dir=log_config.get('log_dir'),
            max_file_size=max_file_size,
            max_backup_count=log_config.get('max_backup_count')
        )
    except Exception as e:   # pylint: disable=broad-except
        logger.warning(f"Failed to parse log config from YAML: {e}")


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    """Trace function calls for debugging."""
    if event in ['call', 'return']:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == 'call':
                    f.write(f"{ts} Call to {func_name} in {filename}:{lineno} "
                            f"from {last_func_name} in {last_filename}:{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from {func_name} in {filename}:{lineno} "
                            f"to {last_func_name} in {last_filename}:{last_lineno}\n")
        except NameError:
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str, root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    
    Args:
        log_file_path: The path to the log file.
        root_dir: The root directory of the code to trace.
    """
    logger.warning(
        "KSANA_TRACE_FUNCTION is enabled. It will record every "
        "function executed by Python. This will slow down the code.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))


# Create a logger for this module
logger = init_logger(__name__)


def get_logger(name: str) -> _KsanaLogger:
    """Get a logger instance."""
    return init_logger(name)
