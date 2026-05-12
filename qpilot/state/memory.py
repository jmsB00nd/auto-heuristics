import json
import os
import random
from typing import List, Dict, Optional

class MemoryManager:
    """Manages persistent history of evaluated ideas and the cross-run
    Hypothesis-Driven Knowledge Graph."""
    def __init__(self, filepath: str, active_limit: int = 20):
        self.filepath = filepath
        self.history: List[Dict] = []
        self.knowledge_graph: Optional[Dict] = None
        self.archive: List[Dict] = []
        self.active_limit = active_limit
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.knowledge_graph = data.get('knowledge_graph')  # legacy 'reflections' key silently dropped
                    self.archive = data.get('archive', [])
            except json.JSONDecodeError:
                self.history = []
                self.knowledge_graph = None
                self.archive = []

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({
                'history': self.history,
                'knowledge_graph': self.knowledge_graph,
                'archive': self.archive
            }, f, indent=2)

    def add_idea(self, name: str, description: str, code: str, mean_swaps: float, mean_depth: float, error: Optional[str] = None, run_id: Optional[str] = None):
        self.history.append({
            "name": name,
            "description": description,
            "code": code,
            "mean_swaps": mean_swaps,
            "mean_depth": mean_depth,
            "error": error,
            "run_id": run_id,
        })
        self._maybe_archive()
        self.save()

    def set_knowledge_graph(self, kg_dict: Optional[Dict]):
        self.knowledge_graph = kg_dict
        self.save()

    def get_knowledge_graph(self) -> Optional[Dict]:
        return self.knowledge_graph

    def get_top_k(self, k: int = 3) -> List[Dict]:
        """Returns the top K successful ideas sorted by mean_swaps."""
        valid = [x for x in self.history if x['error'] is None]
        return sorted(valid, key=lambda x: x['mean_swaps'])[:k]

    def get_worst_k(self, k: int = 1) -> List[Dict]:
        """Returns the worst K successful ideas (to learn from mistakes)."""
        valid = [x for x in self.history if x['error'] is None]
        return sorted(valid, key=lambda x: x['mean_swaps'], reverse=True)[:k]

    def get_diverse_parents(self, k: int = 2) -> List[Dict]:
        """Select k parents from top performers, preferring diversity."""
        valid = [x for x in self.history if x['error'] is None]
        if len(valid) <= k:
            return sorted(valid, key=lambda x: x['mean_swaps'])[:k]

        ranked = sorted(valid, key=lambda x: x['mean_swaps'])

        selected = [ranked[0]]
        remaining = ranked[1:]

        for _ in range(k - 1):
            pool_size = max(4, len(remaining) // 2)
            pool = remaining[:pool_size]
            candidates = random.sample(pool, min(4, len(pool)))

            best_candidate = None
            best_combined_score = float('inf')
            for c in candidates:
                c_words = set(c.get('description', '').lower().split())
                max_jaccard = 0.0
                for s in selected:
                    s_words = set(s.get('description', '').lower().split())
                    union = c_words | s_words
                    if union:
                        max_jaccard = max(max_jaccard, len(c_words & s_words) / len(union))
                combined = c['mean_swaps'] * (0.5 + 0.5 * max_jaccard)
                if combined < best_combined_score:
                    best_combined_score = combined
                    best_candidate = c

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _maybe_archive(self):
        """Move old, low-performing entries to archive (without code)."""
        if len(self.history) <= self.active_limit:
            return

        valid = [x for x in self.history if x['error'] is None]

        top_k = sorted(valid, key=lambda x: x['mean_swaps'])[:self.active_limit // 2]
        top_k_names = {x['name'] for x in top_k}

        recent = self.history[-(self.active_limit // 2):]
        recent_names = {x['name'] for x in recent}

        keep_names = top_k_names | recent_names

        new_history = []
        for entry in self.history:
            if entry['name'] in keep_names:
                new_history.append(entry)
            else:
                self.archive.append({
                    "name": entry['name'],
                    "description": entry['description'],
                    "mean_swaps": entry.get('mean_swaps', float('inf')),
                    "mean_depth": entry.get('mean_depth', 0),
                    "error": entry.get('error'),
                })

        self.history = new_history

    def get_all_summarized(self) -> str:
        """Returns a budget-aware summary of past ideas."""
        if not self.history and not self.archive:
            return "No past ideas in memory."

        summary_parts = []

        top = self.get_top_k(5)
        if top:
            summary_parts.append("=== TOP PERFORMERS ===")
            for i, idea in enumerate(top):
                summary_parts.append(
                    f"#{i+1}: {idea['name']} — {idea['mean_swaps']:.2f} swaps\n"
                    f"  Strategy: {idea['description'][:200]}"
                )

        recent = self.history[-10:]
        if recent:
            summary_parts.append("\n=== RECENT EXPERIMENTS ===")
            for idea in recent:
                status = f"{idea['mean_swaps']:.2f} swaps" if not idea.get('error') else f"FAILED: {str(idea['error'])[:80]}"
                summary_parts.append(f"- {idea['name']}: {status}")

        # to update 
        if self.archive:
            archive_valid = [x for x in self.archive if x.get('error') is None]
            archive_failed = [x for x in self.archive if x.get('error') is not None]
            summary_parts.append(f"\n=== ARCHIVE ({len(self.archive)} older experiments) ===")
            summary_parts.append(f"Successful: {len(archive_valid)}, Failed: {len(archive_failed)}")
            if archive_valid:
                scores = [x['mean_swaps'] for x in archive_valid]
                summary_parts.append(f"Score range: {min(scores):.2f} - {max(scores):.2f}")

        return "\n".join(summary_parts)
