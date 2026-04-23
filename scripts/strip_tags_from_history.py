"""One-shot cleanup: remove the deprecated `tags` field from every entry in experiment_history.json."""
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "experiment_history.json"


def strip_tags(path: Path) -> tuple[int, int]:
    with path.open("r") as f:
        data = json.load(f)

    history_stripped = 0
    for entry in data.get("history", []):
        if "tags" in entry:
            entry.pop("tags")
            history_stripped += 1

    archive_stripped = 0
    for entry in data.get("archive", []):
        if "tags" in entry:
            entry.pop("tags")
            archive_stripped += 1

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    return history_stripped, archive_stripped


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    h, a = strip_tags(path)
    print(f"Stripped tags from {h} history entries and {a} archive entries in {path}")
