"""Structured logging for a single pipeline run.

Adds two file handlers alongside the existing Rich console output:

* ``run.log`` — standard timestamped text log (``ts | level | stage | msg``).
* ``events.jsonl`` — one JSON object per line for machine consumption.

Call :func:`setup_run_logging` once at orchestrator init and use
:func:`log_event` at stage boundaries / key decisions.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOGGER_NAME = "auto_heuristics"
_EVENTS_ATTR = "_events_jsonl_path"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


class _JsonlHandler(logging.Handler):
    """Append a JSON record per log call to a ``.jsonl`` file.

    Only records that carry ``extra={"event": ...}`` are written; plain log
    messages skip this handler to keep the event stream clean.
    """

    def __init__(self, path: str):
        super().__init__(level=logging.DEBUG)
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "event", None)
        if event is None:
            return
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "event": event,
            "message": record.getMessage() or None,
        }
        data = getattr(record, "data", None)
        if data:
            payload["data"] = data
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=_json_default) + "\n")
        except OSError:
            # Logging must never raise.
            pass


class _StageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", "-") or "-"
        record.stage_name = stage
        return super().format(record)


def setup_run_logging(log_dir: str) -> logging.Logger:
    """Configure the project logger for one run.

    Idempotent: calling again for the same ``log_dir`` is safe but will not
    duplicate handlers on the same logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    text_path = os.path.join(log_dir, "run.log")
    events_path = os.path.join(log_dir, "events.jsonl")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Drop any handlers from a previous run in the same process.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    text_handler = logging.FileHandler(text_path, encoding="utf-8")
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(
        _StageFormatter("%(asctime)s | %(levelname)s | %(stage_name)s | %(message)s")
    )
    logger.addHandler(text_handler)

    jsonl_handler = _JsonlHandler(events_path)
    logger.addHandler(jsonl_handler)

    setattr(logger, _EVENTS_ATTR, events_path)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(
    stage: str,
    event: str,
    *,
    level: int = logging.INFO,
    message: Optional[str] = None,
    **data: Any,
) -> None:
    """Emit a structured event.

    Goes to ``run.log`` as a human-readable line and to ``events.jsonl`` as a
    JSON object carrying ``data``. Never raises — safe to call before
    :func:`setup_run_logging` (becomes a no-op).
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return
    extra = {"stage": stage, "event": event, "data": data or None}
    logger.log(level, message or event, extra=extra)
