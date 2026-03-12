"""
=============================================================================
utils.py — Shared utilities: timing profiler, decorators
=============================================================================

TIMING DECORATOR:
  Wraps any function (sync or async) and logs how long it took to a
  rotating log file at logs/timing.log.

  Useful for profiling which parts of the pipeline are slow without
  adding manual time.time() calls everywhere.

USAGE:
  from src.utils import timing_decorator

  @timing_decorator
  def my_slow_function():
      ...

  # Output in logs/timing.log:
  # 2024-01-15 12:34:56 [INFO] predictor.SentimentPredictor.predict_dl took 24.3120 seconds

CONFIGURATION (via .env):
  TIMING_LOG_FILE        — override log file path (default: logs/timing.log)
  TIMING_LOG_MIN_SECONDS — skip logging calls faster than this (default: 0.005s)
                           Set to 0 to log everything. Set higher (e.g. 1.0) to
                           only log the slow calls.
"""

import functools
import inspect
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

_LOGGER_NAME             = 'timing_profiler'
_LOG_FILE_ENV_VAR        = 'TIMING_LOG_FILE'
_DEFAULT_LOG_PATH        = Path('logs') / 'timing.log'
_MIN_LOG_SECONDS_ENV_VAR = 'TIMING_LOG_MIN_SECONDS'
_DEFAULT_MIN_LOG_SECONDS = 0.005   # skip anything faster than 5ms


def _safe_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


_MIN_LOG_SECONDS = _safe_float_env(_MIN_LOG_SECONDS_ENV_VAR, _DEFAULT_MIN_LOG_SECONDS)

_logger_lock   = Lock()
_timing_logger = None


def _resolve_log_path() -> Path:
    env_path = os.getenv(_LOG_FILE_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    # backend/src/utils.py → repo root is parents[2]
    return Path(__file__).resolve().parents[2] / _DEFAULT_LOG_PATH


def _get_timing_logger() -> logging.Logger:
    global _timing_logger
    if _timing_logger is not None:
        return _timing_logger

    with _logger_lock:
        if _timing_logger is not None:
            return _timing_logger

        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.propagate = False   # don't double-log to root logger

        if not logger.handlers:
            log_path = _resolve_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(
                filename   = log_path,
                maxBytes   = 20 * 1024 * 1024,   # 20 MB per file
                backupCount= 3,
                encoding   = 'utf-8',
            )
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt = '%Y-%m-%d %H:%M:%S',
            ))
            logger.addHandler(handler)

        _timing_logger = logger
        return _timing_logger


def _log_timing(func, elapsed_seconds: float) -> None:
    if elapsed_seconds < _MIN_LOG_SECONDS:
        return
    _get_timing_logger().info(
        '%s.%s took %.4f seconds',
        func.__module__, func.__qualname__, elapsed_seconds,
    )


def timing_decorator(func):
    """
    Decorator that logs how long a function takes to logs/timing.log.
    Works on both regular (sync) and async functions transparently.

    Example:
        @timing_decorator
        def predict_dl(self, texts):
            ...
        # Logs: predictor.SentimentPredictor.predict_dl took 24.3120 seconds
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                _log_timing(func, time.perf_counter() - start)
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _log_timing(func, time.perf_counter() - start)
    return wrapper
