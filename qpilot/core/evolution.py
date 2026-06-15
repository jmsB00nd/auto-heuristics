from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

from ..state.budget import BudgetTracker
from ..config.settings import OrchestratorConfig
from .evaluator import CodeEvaluator
from ..prompting.idea_parser import IdeaParser
from ..state.knowledge_graph import Hypothesis, KnowledgeGraph
from .llm_client import LLMClient
from ..state.memory import MemoryManager
from ..prompting.prompt_manager import PromptManager
from utils.utils import save_log, save_json
from utils.logging_setup import log_event

console = Console()


EvalCallback = Callable[[str, str, Dict, Optional[Dict]], None]


def target_func(problem: str) -> str:
    return "init_mapping" if problem == "mapping" else "qlosure_poly_heuristic"


def extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    bare = re.search(r"\{.*\}", text, re.DOTALL)
    if bare:
        try:
            return json.loads(bare.group(0))
        except json.JSONDecodeError:
            pass
    return None


COLD_HYPOTHESIS_FALLBACK = "(no hypothesis yet — use your own judgment to combine the parents)"


class EvolutionLoop:
    """Owns one evolutionary run with HD-KG hypothesis-driven reflection."""

    def __init__(
        self,
        config: OrchestratorConfig,
        prompts: PromptManager,
        llm: LLMClient,
        evaluator: CodeEvaluator,
        memory: MemoryManager,
        run_id: str,
        evolution_dir: str,
        kg: KnowledgeGraph,
        budget: BudgetTracker,
        on_eval: Optional[EvalCallback] = None,
        start_iteration: int = 0,
    ):
        self.config = config
        self.prompts = prompts
        self.llm = llm
        self.evaluator = evaluator
        self.memory = memory
        self.run_id = run_id
        self.evolution_dir = evolution_dir
        self.kg = kg
        self.budget = budget
        self._on_eval = on_eval

        # Ablation switches (default True = full qpilot).
        self.use_kg = getattr(config, "use_kg", True)
        self.use_crossover = getattr(config, "use_crossover", True)
        self.use_mutation = getattr(config, "use_mutation", True)

        self.population: List[Dict] = []
        self.elitist: Optional[Dict] = None
        self.best_obj_overall: float = float("inf")
        self.best_individual_overall: Optional[Dict] = None
        # Continues numbering across outer evolution cycles so iter_* dirs and
        # crossover/mutation child names (…_c{iter}_{idx}) never collide.
        self.iteration: int = start_iteration

        seed = getattr(config, "seed", None)
        self._rng = random.Random(seed)

        os.makedirs(self.evolution_dir, exist_ok=True)
        self._target = target_func(config.problem)


    def seed(self, initial_population: List[Dict]) -> None:
        self.population = [self.normalize_individual(ind, origin="initial") for ind in initial_population]
        self.update_elite()
        console.print(
            f"[cyan]Evolution seeded with {len(self.population)} individuals "
            f"({sum(1 for p in self.population if p['exec_success'])} successful).[/cyan]"
        )

    @staticmethod
    def parse_strategy_meta(response: str) -> Tuple[str, str]:
        if not response:
            return "", ""
        name_m = re.search(r'^\s*NAME\s*:\s*(.+)$', response, re.MULTILINE | re.IGNORECASE)
        desc_m = re.search(
            r'^\s*DESCRIPTION\s*:\s*(.+?)(?=\n\s*(?:[A-Z_]{3,}\s*:|```)|\Z)',
            response,
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )
        raw_name = name_m.group(1).strip().splitlines()[0].strip() if name_m else ""
        cleaned = raw_name.strip("`*_ ")
        slug = re.sub(r'[^A-Za-z0-9_\-]+', '_', cleaned).strip('_').lower()
        desc = desc_m.group(1).strip() if desc_m else ""
        desc = re.sub(r'\s+', ' ', desc)
        return slug, desc

    @staticmethod
    def normalize_individual(ind: Dict, origin: str) -> Dict:
        mean_swaps = ind.get("mean_swaps", float("inf"))
        error = ind.get("error")
        exec_success = error is None and mean_swaps not in (None, float("inf"))
        return {
            "name": ind.get("name", "unnamed"),
            "description": ind.get("description", ""),
            "code": ind.get("code", ""),
            "mean_swaps": mean_swaps if mean_swaps is not None else float("inf"),
            "mean_depth": ind.get("mean_depth", 0) or 0,
            "error": error,
            "exec_success": exec_success,
            "origin": origin,
            "hypothesis_id": ind.get("hypothesis_id"),
        }

    def update_elite(self) -> None:
        successful = [p for p in self.population if p["exec_success"]]
        if not successful:
            return
        best = min(successful, key=lambda p: p["mean_swaps"])
        if self.elitist is None or best["mean_swaps"] < self.elitist["mean_swaps"]:
            self.elitist = best
        if best["mean_swaps"] < self.best_obj_overall:
            self.best_obj_overall = best["mean_swaps"]
            self.best_individual_overall = best
            log_event(
                "evolution",
                "new_best_overall",
                iteration=self.iteration,
                mean_swaps=best["mean_swaps"],
                name=best["name"],
            )


    def random_select(self, pool: List[Dict]) -> Optional[List[Dict]]:
        pool = [p for p in pool if p["exec_success"]]
        if len(pool) < 2:
            return None
        selected: List[Dict] = []
        target_size = 2 * self.config.crossover_count
        trial = 0
        while len(selected) < target_size:
            trial += 1
            a, b = self._rng.sample(pool, 2)
            if a["mean_swaps"] != b["mean_swaps"]:
                selected.extend([a, b])
            if trial > 1000:
                if len(selected) >= 2:
                    break
                return None
        return selected[:target_size]


    def split_top_bottom(self) -> Tuple[List[Dict], List[Dict]]:
        successful = [p for p in self.population if p["exec_success"]]
        if not successful:
            return [], []
        ranked = sorted(successful, key=lambda p: p["mean_swaps"])
        k = max(1, min(len(ranked) // 2, self.config.pop_size // 2 or 1))
        top = ranked[:k]
        bottom = ranked[-k:] if len(ranked) > k else []
        return top, bottom

    def format_heuristics_block(self, individuals: List[Dict], group_label: str) -> str:
        chunks = []
        for ind in individuals:
            chunks.append(
                f"--- [{group_label}] {ind['name']} (mean_swaps={ind['mean_swaps']:.2f}) ---\n"
                f"{ind['code']}"
            )
        return "\n\n".join(chunks)

    def extract_traits(self, top: List[Dict], bottom: List[Dict], iter_dir: str) -> List[str]:
        if not top and not bottom:
            return []
        block = "\n\n".join(filter(None, [
            self.format_heuristics_block(top, "TOP"),
            self.format_heuristics_block(bottom, "BOTTOM"),
        ]))
        user = self.prompts.trait_extraction_prompt.format(heuristics_block=block)
        prompt = f"{self.prompts.system_reflector}\n\n{user}"

        with self.llm.stage(f"evolution/iter_{self.iteration}/trait_extraction"):
            response, _ = self.llm.query(prompt, reset_conversation=True)
        save_log(iter_dir, "trait_extraction_prompt.txt", prompt, quiet=True)
        save_log(iter_dir, "trait_extraction_response.txt", response or "", quiet=True)

        data = extract_json(response or "")
        if not data or not isinstance(data.get("traits"), list):
            log_event("evolution", "trait_extraction_failed", iteration=self.iteration)
            return []
        registered = self.kg.register_traits(data["traits"])
        save_json(iter_dir, "traits_registered.json", [
            {"id": t.id, "label": t.label, "exemplars": t.exemplars} for t in registered
        ], quiet=True)
        return [t.label for t in registered]


    def partition_traits_by_group(
        self, top: List[Dict], bottom: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        top_names = {p["name"] for p in top}
        bottom_names = {p["name"] for p in bottom}
        top_labels: List[str] = []
        bottom_labels: List[str] = []
        for trait in self.kg.traits.values():
            top_count = sum(1 for ex in trait.exemplars if ex in top_names)
            bot_count = sum(1 for ex in trait.exemplars if ex in bottom_names)
            if top_count > bot_count:
                top_labels.append(trait.label)
            elif bot_count > top_count:
                bottom_labels.append(trait.label)
        return top_labels, bottom_labels

    def generate_hypotheses(
        self, top: List[Dict], bottom: List[Dict], iter_dir: str
    ) -> List[Hypothesis]:
        top_labels, bottom_labels = self.partition_traits_by_group(top, bottom)
        if not top_labels and not bottom_labels:
            return []

        active = self.kg.active_statements()
        user = self.prompts.hypothesis_generation_prompt.format(
            top_traits_block="\n".join(f"- {l}" for l in top_labels) or "(none identified)",
            bottom_traits_block="\n".join(f"- {l}" for l in bottom_labels) or "(none identified)",
            active_hypotheses_block="\n".join(f"- {s}" for s in active) or "(none yet)",
        )
        prompt = f"{self.prompts.system_reflector}\n\n{user}"

        with self.llm.stage(f"evolution/iter_{self.iteration}/hypothesis_generation"):
            response, _ = self.llm.query(prompt, reset_conversation=True)
        save_log(iter_dir, "hypothesis_generation_prompt.txt", prompt, quiet=True)
        save_log(iter_dir, "hypothesis_generation_response.txt", response or "", quiet=True)

        data = extract_json(response or "")
        if not data or not isinstance(data.get("hypotheses"), list):
            log_event("evolution", "hypothesis_generation_failed", iteration=self.iteration)
            return []

        registered: List[Hypothesis] = []
        for h in data["hypotheses"]:
            statement = h.get("statement", "") if isinstance(h, dict) else ""
            related = h.get("related_traits") if isinstance(h, dict) else None
            hyp = self.kg.register_hypothesis(statement, related, self.iteration)
            if hyp is not None:
                registered.append(hyp)
        save_json(iter_dir, "hypotheses_registered.json", [
            {"id": h.id, "statement": h.statement, "related_trait_ids": h.related_trait_ids}
            for h in registered
        ], quiet=True)
        return registered

    @staticmethod
    def render_hypothesis(hyp: Optional[Hypothesis]) -> str:
        if hyp is None:
            return COLD_HYPOTHESIS_FALLBACK
        return f"[{hyp.id} | confidence={hyp.confidence:.2f} | status={hyp.status}] {hyp.statement}"


    def crossover(self, selected: List[Dict], iter_dir: str) -> List[Tuple[Dict, Dict, Dict]]:
        """Returns list of (worse, better, child) — child carries hypothesis_id."""
        pairs: List[Tuple[Dict, Dict]] = []
        for i in range(0, len(selected), 2):
            a, b = selected[i], selected[i + 1]
            if a["mean_swaps"] <= b["mean_swaps"]:
                better, worse = a, b
            else:
                better, worse = b, a
            pairs.append((worse, better))

        prompts = []
        sampled_hyps: List[Optional[Hypothesis]] = []
        for worse, better in pairs:
            hyp = self.kg.sample_for_crossover(self._rng) if self.use_kg else None
            sampled_hyps.append(hyp)
            user = self.prompts.crossover_prompt.format(
                variables=self.prompts.variables,
                worse_code=worse["code"],
                better_code=better["code"],
                hypothesis=self.render_hypothesis(hyp),
            )
            prompts.append(f"{self.prompts.system_generator}\n\n{user}")

        console.print(Rule(f"Crossover ({len(prompts)} children)", style="dim magenta"))
        responses = self._parallel_llm(
            prompts,
            stage=f"evolution/iter_{self.iteration}/crossover",
            label="crossover children",
        )
        children_dir = os.path.join(iter_dir, "crossover")
        os.makedirs(children_dir, exist_ok=True)

        triples: List[Tuple[Dict, Dict, Dict]] = []
        for idx, ((worse, better), hyp, prompt, (response, call_snapshot)) in enumerate(
            zip(pairs, sampled_hyps, prompts, responses)
        ):
            child_dir = os.path.join(children_dir, f"child_{idx}")
            os.makedirs(child_dir, exist_ok=True)
            save_log(child_dir, "prompt.txt", prompt, quiet=True)
            save_log(child_dir, "raw_response.txt", response or "", quiet=True)
            if hyp is not None:
                save_json(child_dir, "hypothesis.json", {
                    "id": hyp.id, "statement": hyp.statement,
                    "confidence": hyp.confidence, "status": hyp.status,
                }, quiet=True)

            code = IdeaParser.extract_code(response or "", self._target)
            slug, desc = self.parse_strategy_meta(response or "")
            suffix = f"c{self.iteration}_{idx}"
            name = f"{slug}_{suffix}" if slug else f"crossover_iter{self.iteration}_{idx}"
            description = desc or f"Crossover child (iter {self.iteration}, idx {idx})"
            hypothesis_id = hyp.id if hyp is not None else None
            if not code:
                child = {
                    "name": name,
                    "description": description,
                    "code": "",
                    "mean_swaps": float("inf"),
                    "mean_depth": 0,
                    "error": "Extraction Failed",
                    "exec_success": False,
                    "origin": f"crossover_iter{self.iteration}",
                    "hypothesis_id": hypothesis_id,
                    "_call_snapshot": call_snapshot,
                }
            else:
                with open(os.path.join(child_dir, "heuristic.py"), "w") as f:
                    f.write(code)
                child = {
                    "name": name,
                    "description": description,
                    "code": code,
                    "mean_swaps": float("inf"),
                    "mean_depth": 0,
                    "error": None,
                    "exec_success": False,
                    "origin": f"crossover_iter{self.iteration}",
                    "hypothesis_id": hypothesis_id,
                    "_child_dir": child_dir,
                    "_call_snapshot": call_snapshot,
                }
            triples.append((worse, better, child))
        return triples


    def mutate(self, n_mutants: int, iter_dir: str) -> List[Tuple[Dict, Dict]]:
        """Returns list of (parent_elite, mutant) — mutant carries hypothesis_id."""
        if self.elitist is None or not self.elitist.get("code"):
            return []
        hyp = self.kg.sample_for_mutation() if self.use_kg else None
        prompt = f"{self.prompts.system_generator}\n\n" + self.prompts.refinement_prompt.format(
            variables=self.prompts.variables,
            hypothesis=self.render_hypothesis(hyp),
            elitist_code=self.elitist["code"],
        )
        prompts = [prompt] * n_mutants
        console.print(Rule(f"Mutation ({n_mutants} variants of {self.elitist['name']})", style="dim yellow"))
        responses = self._parallel_llm(
            prompts,
            stage=f"evolution/iter_{self.iteration}/mutation",
            label="mutants",
        )
        mutants_dir = os.path.join(iter_dir, "mutation")
        os.makedirs(mutants_dir, exist_ok=True)

        pairs: List[Tuple[Dict, Dict]] = []
        for idx, (response, call_snapshot) in enumerate(responses):
            child_dir = os.path.join(mutants_dir, f"mutant_{idx}")
            os.makedirs(child_dir, exist_ok=True)
            save_log(child_dir, "prompt.txt", prompt, quiet=True)
            save_log(child_dir, "raw_response.txt", response or "", quiet=True)
            if hyp is not None:
                save_json(child_dir, "hypothesis.json", {
                    "id": hyp.id, "statement": hyp.statement,
                    "confidence": hyp.confidence, "status": hyp.status,
                }, quiet=True)

            code = IdeaParser.extract_code(response or "", self._target)
            slug, desc = self.parse_strategy_meta(response or "")
            suffix = f"m{self.iteration}_{idx}"
            name = f"{slug}_{suffix}" if slug else f"mutation_iter{self.iteration}_{idx}"
            description = desc or f"Mutation of {self.elitist['name']} (iter {self.iteration}, idx {idx})"
            hypothesis_id = hyp.id if hyp is not None else None
            if not code:
                mutant = {
                    "name": name,
                    "description": description,
                    "code": "",
                    "mean_swaps": float("inf"),
                    "mean_depth": 0,
                    "error": "Extraction Failed",
                    "exec_success": False,
                    "origin": f"mutation_iter{self.iteration}",
                    "hypothesis_id": hypothesis_id,
                    "_call_snapshot": call_snapshot,
                }
            else:
                with open(os.path.join(child_dir, "heuristic.py"), "w") as f:
                    f.write(code)
                mutant = {
                    "name": name,
                    "description": description,
                    "code": code,
                    "mean_swaps": float("inf"),
                    "mean_depth": 0,
                    "error": None,
                    "exec_success": False,
                    "origin": f"mutation_iter{self.iteration}",
                    "hypothesis_id": hypothesis_id,
                    "_child_dir": child_dir,
                    "_call_snapshot": call_snapshot,
                }
            pairs.append((self.elitist, mutant))
        return pairs


    def _evaluate(self, individuals: List[Dict], phase: str) -> List[Dict]:
        for ind in individuals:
            if ind.get("exec_success") is True and ind.get("mean_swaps") != float("inf"):
                continue
            if ind.get("error") == "Extraction Failed":
                self.budget.increment()
                self._record(phase, ind)
                continue
            stats = self.evaluator.evaluate(ind["code"])
            self.budget.increment()
            ind["mean_swaps"] = stats.get("mean_swaps", float("inf"))
            ind["mean_depth"] = stats.get("mean_depth", 0) or 0
            ind["error"] = stats.get("error")
            ind["exec_success"] = ind["error"] is None and ind["mean_swaps"] != float("inf")

            child_dir = ind.pop("_child_dir", None)
            if child_dir:
                save_json(child_dir, "evaluation_results.json", stats, quiet=True)

            self.memory.add_idea(
                name=ind["name"],
                description=ind["description"],
                code=ind["code"],
                mean_swaps=ind["mean_swaps"],
                mean_depth=ind["mean_depth"],
                error=ind["error"],
                run_id=self.run_id,
            )
            self._record(phase, ind)

            if ind["exec_success"]:
                console.print(
                    f"[green]✔ {ind['name']}: {ind['mean_swaps']:.2f} avg swaps[/green]"
                )
            else:
                console.print(f"[red]✘ {ind['name']}: {ind['error']}[/red]")
        return individuals

    def _record(self, phase: str, ind: Dict) -> None:
        if self._on_eval is None:
            return

        snapshot = ind.get("_call_snapshot")
        self._on_eval(phase, ind["name"], {
            "mean_swaps": ind["mean_swaps"] if ind["mean_swaps"] != float("inf") else None,
            "mean_depth": ind["mean_depth"],
            "error": ind["error"],
        }, snapshot)


    def _apply_kg_updates(
        self,
        triples: List[Tuple[Dict, Dict, Dict]],
        best_at_iter_start: float,
    ) -> None:
        
        for _worse, better, child in triples:
            hyp_id = child.get("hypothesis_id")
            if hyp_id is None:
                continue
            self.kg.update_after_trial(
                hypothesis_id=hyp_id,
                parent_obj=better["mean_swaps"],
                offspring_obj=child["mean_swaps"],
                best_so_far_at_iter_start=best_at_iter_start if best_at_iter_start != float("inf") else better["mean_swaps"],
                success=child["exec_success"],
            )


    # parallel LLM helper

    def _parallel_llm(self, prompts: List[str], stage: str, label: str = "LLM calls") -> List[Tuple[Optional[str], Optional[Dict]]]:
        """Returns (response, call_snapshot) per prompt. The snapshot lets
        callers attribute exact cumulative token counts to each child even
        when many calls finish in close succession."""
        if not prompts:
            return []
        workers = max(1, min(self.config.evolution_workers, len(prompts)))
        results: List[Tuple[Optional[str], Optional[Dict]]] = [(None, None)] * len(prompts)

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"{label} ({workers} workers)", total=len(prompts))

            def _call(idx_prompt):
                idx, prompt = idx_prompt
                with self.llm.stage(stage):
                    resp, snap = self.llm.query(prompt, reset_conversation=True, show_counter=False)
                progress.advance(task)
                return idx, resp, snap

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, resp, snap in pool.map(_call, list(enumerate(prompts))):
                    results[idx] = (resp, snap)
        return results

    # main loop 

    def evolve(self, max_iterations: Optional[int] = None) -> List[Dict]:
        mode = "HD-KG hypothesis-driven" if self.use_kg else "ablation: KG off"
        console.print(Rule(f"Evolution ({mode})", style="bold blue"))
        if not getattr(self.config, "run_evolution", True):
            console.print("[yellow]run_evolution=False; skipping.[/yellow]")
            return self.population
        if not self.use_crossover and not self.use_mutation:
            console.print("[yellow]Both crossover and mutation disabled; nothing to evolve.[/yellow]")
            return self.population

        iters_done = 0
        while not self.budget.exhausted():
            if max_iterations is not None and iters_done >= max_iterations:
                break
            successful = [p for p in self.population if p["exec_success"]]
            # Crossover needs ≥2 distinct parents; mutation-only needs just the elite.
            min_needed = 2 if self.use_crossover else 1
            if len(successful) < min_needed:
                console.print(
                    f"[yellow]Fewer than {min_needed} successful individuals ({len(successful)}); "
                    "cannot continue evolutionary loop.[/yellow]"
                )
                break

            iter_dir = os.path.join(self.evolution_dir, f"iter_{self.iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            best_str = f"{self.best_obj_overall:.2f}" if self.best_obj_overall != float("inf") else "—"
            console.print(
                Rule(
                    f"Iteration {self.iteration}  (FE {self.budget.current}/{self.budget.max_fe}, best={best_str})",
                    style="bold blue",
                )
            )
            best_at_iter_start = self.best_obj_overall

            # select pairs (crossover only)
            selected = None
            if self.use_crossover:
                pool = self.population if (self.elitist is None or self.elitist in self.population) \
                    else [self.elitist] + self.population
                selected = self.random_select(pool)
                if selected is None:
                    console.print("[yellow]Selection could not find distinct-obj pairs; stopping.[/yellow]")
                    break

            # trait extraction + hypothesis generation (HD-KG only)
            if self.use_kg:
                top, bottom = self.split_top_bottom()
                self.extract_traits(top, bottom, iter_dir)
                self.generate_hypotheses(top, bottom, iter_dir)

            # crossover (hypothesis-conditioned)
            if self.use_crossover:
                triples = self.crossover(selected, iter_dir)
                children = [t[2] for t in triples]
                children = self._evaluate(children, phase=f"evolution/iter_{self.iteration}/crossover")
                self._apply_kg_updates(triples, best_at_iter_start)
                self.population.extend(children)
                self.update_elite()

                if self.budget.exhausted():
                    save_json(iter_dir, "knowledge_graph.json", self.kg.to_dict(), quiet=True)
                    self.iteration += 1
                    break

            # mutation (single mutation of the elite, highest-confidence hypothesis)
            if self.use_mutation:
                mutant_pairs = self.mutate(1, iter_dir)
                mutants = [m for _e, m in mutant_pairs]
                mutants = self._evaluate(mutants, phase=f"evolution/iter_{self.iteration}/mutation")
                self._apply_kg_updates(
                    [(self.elitist, self.elitist, m) for _e, m in mutant_pairs],
                    best_at_iter_start,
                )
                self.population.extend(mutants)
                self.update_elite()

            # persist KG snapshot for this iteration
            save_json(iter_dir, "knowledge_graph.json", self.kg.to_dict(), quiet=True)

            self.iteration += 1
            iters_done += 1

        stop = self.budget.stop_reason()
        stop_note = f", stop={stop}" if stop else ""
        console.print(
            f"[bold green]Evolution finished: FE={self.budget.current}, "
            f"elapsed={self.budget.elapsed():.0f}s{stop_note}, "
            f"best={self.best_obj_overall:.2f}[/bold green]"
        )
        return self.population
