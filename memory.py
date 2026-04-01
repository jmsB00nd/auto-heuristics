import json
import os
from typing import List, Dict, Optional

class MemoryManager:
    """Manages persistent history of evaluated ideas and reflections."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.history: List[Dict] = []
        self.reflections: List[str] = []
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.reflections = data.get('reflections', [])
            except json.JSONDecodeError:
                self.history = []
                self.reflections = []

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({'history': self.history, 'reflections': self.reflections}, f, indent=4)

    def add_idea(self, name: str, description: str, code: str, mean_swaps: float, mean_depth: float, error: Optional[str] = None):
        self.history.append({
            "name": name,
            "description": description,
            "code": code,
            "mean_swaps": mean_swaps,
            "mean_depth": mean_depth,
            "error": error
        })
        self.save()

    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
        self.save()

    def get_top_k(self, k: int = 3) -> List[Dict]:
        """Returns the top K successful ideas sorted by mean_swaps."""
        valid = [x for x in self.history if x['error'] is None]
        return sorted(valid, key=lambda x: x['mean_swaps'])[:k]

    def get_worst_k(self, k: int = 1) -> List[Dict]:
        """Returns the worst K successful ideas (to learn from mistakes)."""
        valid = [x for x in self.history if x['error'] is None]
        return sorted(valid, key=lambda x: x['mean_swaps'], reverse=True)[:k]

    def get_latest_reflection(self) -> str:
        return self.reflections[-1] if self.reflections else ""