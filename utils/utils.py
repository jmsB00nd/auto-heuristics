import os
import json
from datetime import datetime

from rich.console import Console


console = Console()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_log(stage_dir: str, filename: str, content: str):
    """Save a text log file inside the given stage directory."""
    _ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def _save_json(stage_dir: str, filename: str, data):
    """Save a JSON log file inside the given stage directory."""
    _ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def file_to_string(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read()