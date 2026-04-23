"""Regenerate ``docs/schemas/*.json`` from the pydantic models in :mod:`agent.schemas`.

Usage::

    python scripts/export_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.schemas import (
    EvaluationResult,
    HistoryEntry,
    RunMetadata,
    RunSummary,
)


OUT_DIR = Path("docs/schemas")

MODELS = {
    "run_metadata.schema.json": RunMetadata,
    "run_summary.schema.json": RunSummary,
    "evaluation_result.schema.json": EvaluationResult,
    "history_entry.schema.json": HistoryEntry,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        path = OUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model.model_json_schema(), f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
