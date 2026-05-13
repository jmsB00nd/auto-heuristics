"""Hypothesis-Driven Knowledge Graph (HD-KG) for the evolution loop.

Replaces the unstructured short/long reflection blob with a graph of
algorithmic Traits and testable Hypotheses whose confidence is updated
empirically from offspring-vs-parent score deltas.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional


HypothesisStatus = Literal["open", "confident", "falsified", "exhausted"]


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "trait"


@dataclass
class Trait:
    id: str
    label: str
    exemplars: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    id: str
    statement: str
    related_trait_ids: List[str]
    created_iter: int
    confidence: float = 0.5
    status: HypothesisStatus = "open"
    trials: int = 0
    wins: int = 0
    last_confidence_increase_at_trial: int = 0


class KnowledgeGraph:
    """Structured hypothesis store. The whole graph is JSON-serializable."""

    def __init__(
        self,
        pop_size: int,
        alpha: float = 0.2,
        confidence_threshold: float = 0.75,
        open_sample_prob: float = 0.3,
    ):
        self.pop_size = pop_size
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.open_sample_prob = open_sample_prob
        self.traits: Dict[str, Trait] = {}
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._trait_by_label: Dict[str, str] = {}
        self._next_trait_n = 0
        self._next_hyp_n = 0

    # ---------- registration ----------

    def register_traits(self, traits: List[Dict]) -> List[Trait]:
        """Register traits (dedup by lowercased label). Each input dict needs
        ``label``; ``exemplars`` is optional."""
        out: List[Trait] = []
        for t in traits:
            label = (t.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()
            if key in self._trait_by_label:
                trait = self.traits[self._trait_by_label[key]]
                # Merge new exemplars in
                for ex in t.get("exemplars") or []:
                    if ex not in trait.exemplars:
                        trait.exemplars.append(ex)
            else:
                tid = f"t{self._next_trait_n}_{_slugify(label)[:24]}"
                self._next_trait_n += 1
                trait = Trait(id=tid, label=label, exemplars=list(t.get("exemplars") or []))
                self.traits[tid] = trait
                self._trait_by_label[key] = tid
            out.append(trait)
        return out

    def register_hypothesis(
        self,
        statement: str,
        related_trait_labels: Optional[List[str]],
        iter_idx: int,
    ) -> Optional[Hypothesis]:
        statement = (statement or "").strip()
        if not statement:
            return None
        # Skip exact-duplicate statements (case-insensitive).
        for h in self.hypotheses.values():
            if h.statement.lower() == statement.lower():
                return h
        related_ids: List[str] = []
        for lbl in related_trait_labels or []:
            tid = self._trait_by_label.get((lbl or "").strip().lower())
            if tid:
                related_ids.append(tid)
        hid = f"h{self._next_hyp_n}"
        self._next_hyp_n += 1
        hyp = Hypothesis(
            id=hid,
            statement=statement,
            related_trait_ids=related_ids,
            created_iter=iter_idx,
        )
        self.hypotheses[hid] = hyp
        return hyp

    # ---------- updates ----------

    def update_after_trial(
        self,
        hypothesis_id: Optional[str],
        parent_obj: float,
        offspring_obj: float,
        best_so_far_at_iter_start: float,
        success: bool,
    ) -> None:
        """Apply one empirical observation to a hypothesis. Cold-start children
        with ``hypothesis_id=None`` are silently ignored."""
        if hypothesis_id is None:
            return
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp is None:
            return

        # Update confidence + counters based on outcome class.
        if not success:
            hyp.confidence = 0.1
            hyp.status = "falsified"
        elif parent_obj != offspring_obj:
            denom = (
                best_so_far_at_iter_start
                if best_so_far_at_iter_start not in (0, float("inf"))
                else max(parent_obj, 1e-9)
            )
            delta = (parent_obj - offspring_obj) / denom
            c_old = hyp.confidence
            c_new = max(0.0, min(1.0, c_old + self.alpha * delta))
            if offspring_obj < parent_obj:
                hyp.wins += 1
            if c_new > c_old:
                hyp.last_confidence_increase_at_trial = hyp.trials + 1
            hyp.confidence = c_new
        # else: tied delta — no confidence change, no win.

        hyp.trials += 1

        # Promote open → confident (skipped if already falsified above).
        if hyp.status == "open" and hyp.confidence >= self.confidence_threshold:
            hyp.status = "confident"

        # Exhaust confident hypothesis once it stops moving.
        if (
            hyp.status == "confident"
            and hyp.trials >= 2 * self.pop_size
            and (hyp.trials - hyp.last_confidence_increase_at_trial) >= self.pop_size
        ):
            hyp.status = "exhausted"

    # ---------- sampling ----------

    def sample_for_crossover(self, rng: random.Random) -> Optional[Hypothesis]:
        """Mixed exploit/explore sampling. Returns ``None`` only if both pools
        are empty (cold-start case)."""
        confident = [h for h in self.hypotheses.values() if h.status == "confident"]
        open_h = [h for h in self.hypotheses.values() if h.status == "open"]

        roll = rng.random()
        prefer_open = roll < self.open_sample_prob

        if prefer_open and open_h:
            weights = [1.0 / (h.trials + 1) for h in open_h]
            return rng.choices(open_h, weights=weights, k=1)[0]
        if not prefer_open and confident:
            weights = [max(h.confidence, 1e-3) for h in confident]
            return rng.choices(confident, weights=weights, k=1)[0]

        # Fallback when the preferred pool is empty
        if open_h:
            weights = [1.0 / (h.trials + 1) for h in open_h]
            return rng.choices(open_h, weights=weights, k=1)[0]
        if confident:
            weights = [max(h.confidence, 1e-3) for h in confident]
            return rng.choices(confident, weights=weights, k=1)[0]
        return None

    def sample_for_mutation(self) -> Optional[Hypothesis]:
        active = [h for h in self.hypotheses.values() if h.status in ("open", "confident")]
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)

    # ---------- queries ----------

    def banned_statements(self) -> List[str]:
        return [
            h.statement
            for h in self.hypotheses.values()
            if h.status in ("falsified", "exhausted")
        ]

    def active_statements(self) -> List[str]:
        return [
            h.statement
            for h in self.hypotheses.values()
            if h.status in ("open", "confident")
        ]

    def all_explored_statements(self) -> List[str]:
        """Every direction the agent has already considered, regardless of
        outcome (open / confident / falsified / exhausted). Used by the
        re-ideation phase to push the LLM toward genuinely novel directions."""
        return [h.statement for h in self.hypotheses.values()]

    # ---------- persistence ----------

    def to_dict(self) -> Dict:
        return {
            "pop_size": self.pop_size,
            "alpha": self.alpha,
            "confidence_threshold": self.confidence_threshold,
            "open_sample_prob": self.open_sample_prob,
            "next_trait_n": self._next_trait_n,
            "next_hyp_n": self._next_hyp_n,
            "traits": {tid: asdict(t) for tid, t in self.traits.items()},
            "hypotheses": {hid: asdict(h) for hid, h in self.hypotheses.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict, pop_size: Optional[int] = None) -> "KnowledgeGraph":
        kg = cls(
            pop_size=pop_size if pop_size is not None else data.get("pop_size", 10),
            alpha=data.get("alpha", 0.2),
            confidence_threshold=data.get("confidence_threshold", 0.75),
            open_sample_prob=data.get("open_sample_prob", 0.3),
        )
        kg._next_trait_n = data.get("next_trait_n", 0)
        kg._next_hyp_n = data.get("next_hyp_n", 0)
        for tid, td in (data.get("traits") or {}).items():
            kg.traits[tid] = Trait(**td)
            kg._trait_by_label[td["label"].lower()] = tid
        for hid, hd in (data.get("hypotheses") or {}).items():
            kg.hypotheses[hid] = Hypothesis(**hd)
        return kg
