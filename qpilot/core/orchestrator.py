import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.panel import Panel

from ..state.budget import BudgetTracker
from ..config.settings import OrchestratorConfig
from ..prompting.prompt_manager import PromptManager
from .llm_client import LLMClient
from ..prompting.idea_parser import IdeaParser
from .evaluator import CodeEvaluator
from .evolution import EvolutionLoop
from ..state.knowledge_graph import KnowledgeGraph
from ..state.memory import MemoryManager
from ..config.run_context import build_run_metadata, write_run_metadata
from ..config.schemas import BestEntry, RankingEntry, RunSummary
from ..viz.plotting import plot_tokens_vs_metrics
from utils.utils import save_log, save_json
from utils.logging_setup import log_event, setup_run_logging

console = Console()


def _seed_rngs(seed):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass


def _save_heuristic(run_log_dir: str, safe_name: str, code: str) -> str:
    """Dual-write a heuristic: global pool + per-run mirror. Returns per-run path."""
    os.makedirs("outputs/heuristics", exist_ok=True)
    with open(f"outputs/heuristics/{safe_name}.py", "w") as f:
        f.write(code)
    per_run_dir = os.path.join(run_log_dir, "heuristics")
    os.makedirs(per_run_dir, exist_ok=True)
    per_run_path = os.path.join(per_run_dir, f"{safe_name}.py")
    with open(per_run_path, "w") as f:
        f.write(code)
    return per_run_path


class Qpilot:
    """High-level controller tying the components together."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.run_id = config.run_id
        _seed_rngs(getattr(config, "seed", None))
        os.makedirs(self.config.log_dir, exist_ok=True)
        setup_run_logging(self.config.log_dir)
        self.prompts = PromptManager(config.prompts_dir, config.problem)
        self.llm = LLMClient(config)
        self.evaluator = CodeEvaluator(config)
        self.memory = MemoryManager(config.history_file, active_limit=getattr(config, 'active_memory_limit', 20))

        # Shared budget — debited by EvolutionLoop._evaluate AND by
        # re-ideation-phase re-implementations. Stops on whichever comes first:
        # the FE count (max_fe) or the wall-clock limit (max_time_seconds).
        self.budget = BudgetTracker(
            max_fe=config.max_fe,
            max_seconds=getattr(config, "max_time_seconds", None),
        )

        # Cross-run KG: load if memory has one, else fresh. Ablation (use_kg=False):
        # start from an empty graph and never load prior state — it stays inert
        # (no traits/hypotheses are ever registered, sampled, or updated).
        kg_dict = self.memory.get_knowledge_graph() if getattr(config, "use_kg", True) else None
        if kg_dict:
            self.kg = KnowledgeGraph.from_dict(kg_dict, pop_size=config.pop_size)
        else:
            self.kg = KnowledgeGraph(
                pop_size=config.pop_size,
                alpha=getattr(config, "kg_alpha", 0.2),
                confidence_threshold=getattr(config, "kg_confidence_threshold", 0.75),
                open_sample_prob=getattr(config, "kg_open_sample_prob", 0.3),
            )

        self._consecutive_failed_restarts = 0
        self._reideation_count = 0

        # Snapshot run metadata (config, seeds, prompt hashes, backend, benchmarks).
        metadata = build_run_metadata(
            config=config,
            run_id=self.run_id,
            experiment_name=getattr(config, "experiment_name", "run"),
            seeds={"python": getattr(config, "seed", None), "numpy": getattr(config, "seed", None)},
        )
        write_run_metadata(metadata, self.config.log_dir)
        log_event("init", "run_started", run_id=self.run_id, experiment=metadata.experiment_name)
        
        # State
        self.literature_insights = ""
        self.top_ideas = []
        # One entry per heuristic evaluated, carrying cumulative token counters
        # at the moment of evaluation. Used for tokens-vs-metrics plot.
        self.eval_trace: list[dict] = []

        self.stage_times = {
            "literature_review": 0.0,
            "ideas_generation": 0.0,
            "implementation": 0.0,
            "evolution": 0.0,
            "total_pipeline": 0.0
        }

        self.pipeline_start_time = None

        # Setup Dirs
        self.literature_review_dir = os.path.join(self.config.log_dir, "literature_review")
        self.ideas_dir = os.path.join(self.config.log_dir, "idea_generation")
        self.implementation_dir = os.path.join(self.config.log_dir, "implementation")
        self.evolution_dir = os.path.join(self.config.log_dir, "evolution")
        for d in [self.literature_review_dir, self.ideas_dir, self.implementation_dir, self.evolution_dir]:
            os.makedirs(d, exist_ok=True)

    def _record_eval(self, stage: str, label: str, stats: dict, call_snapshot: Optional[dict] = None) -> None:
        """Append one row to ``eval_trace`` capturing cumulative tokens *at the
        moment THIS heuristic's LLM call committed* (when ``call_snapshot`` is
        provided). Without a snapshot we fall back to the live global counter,
        which is correct for sequential calls but stamps every heuristic in a
        parallel batch with the same near-final value (the bug we're fixing)."""
        mean_swaps = stats.get("mean_swaps")
        mean_depth = stats.get("mean_depth")
        error = stats.get("error")
        try:
            mean_swaps = float(mean_swaps) if mean_swaps is not None else None
        except (TypeError, ValueError):
            mean_swaps = None
        try:
            mean_depth = float(mean_depth) if mean_depth is not None else None
        except (TypeError, ValueError):
            mean_depth = None
        if call_snapshot:
            cum_tot = int(call_snapshot["cumulative_total_tokens"])
        else:
            cum_tot = self.llm.total_tokens
        self.eval_trace.append({
            "stage": stage,
            "label": label,
            "cumulative_total_tokens": cum_tot,
            "mean_swaps": mean_swaps,
            "mean_depth": mean_depth,
            "error": error,
        })

    def literature_review(self):
        start_time = time.time()
        console.print(Rule("Literature Review", style="bold cyan"))

        if not getattr(self.config, 'run_stage1_literature_review', False):
            console.print("[yellow]Skipping Literature Review.[/yellow]")
            return

        with self.llm.stage("literature_review"):
            response, _ = self.llm.query(self.prompts.lit_review_prompt, reset_conversation=True)
        self.literature_insights = response or ""

        if response:
            save_log(self.literature_review_dir, "literature_review.txt", response)
            console.print("[bold green]✓ Literature Review complete.[/bold green]\n")

        self.stage_times["literature_review"] = time.time() - start_time

    def ideas_generation(self):
        start_time = time.time()
        console.print(Rule("Idea Generation", style="bold magenta"))

        use_memory = getattr(self.config, "use_ideation_memory", True)
        memory_dump = self.memory.get_all_summarized() if use_memory else "No past ideas in memory."
        memory_resume = ""
        if not use_memory:
            console.print("[yellow]Ablation: ideation memory disabled — generating ideas with no past-experiment context.[/yellow]")

        with self.llm.stage("ideas_generation"):
            if use_memory and memory_dump != "No past ideas in memory.":
                console.print("[cyan]Synthesizing all past memory into a global resume...[/cyan]")

                summary_prompt = (
                    f"{self.prompts.system_generator}\n"
                    f"{self.prompts.memory_summary_prompt}\n"
                    f"PAST EXPERIMENTS LOG:\n{memory_dump}\n"
                )

                save_log(self.ideas_dir, "memory_summary_prompt.txt", summary_prompt)

                memory_resume, _ = self.llm.query(summary_prompt, reset_conversation=True)
                save_log(self.ideas_dir, "global_memory_resume.txt", memory_resume or "")
                console.print("[green]✓ Memory resume generated.[/green]")

            lit_context = f"\nLiterature Context:\n{self.literature_insights}\n" if self.literature_insights else ""
            memory_context = f"\nGLOBAL RESUME OF PAST EXPERIMENTS:\n{memory_resume}\n" if memory_resume else ""

            prompt = (
                f"{self.prompts.system_generator}\n"
                f"{lit_context}\n"
                f"{self.prompts.idea_prompt}\n\n"
                f"{memory_context}\n\n"
                f"Do not repeat or slightly modify past experiments. Use them only to understand what has already been explored and avoid previously tried or failed approaches. Focus on proposing new and distinct ideas.\n\n"
            )

            save_log(self.ideas_dir, "prompt.txt", prompt)

            console.print("[cyan]Generating new novel ideas based on global context...[/cyan]")
            response, _ = self.llm.query(prompt, reset_conversation=not getattr(self.config, 'use_conversation_mode', True))
            save_log(self.ideas_dir, "raw_ideas.txt", response or "")

        # Parse ideas
        kept, eliminated = IdeaParser.parse_ideas(response or "")
        self.top_ideas = kept[:getattr(self.config, 'target_top_ideas', 5)]
        
        save_json(self.ideas_dir, "top_ideas.json", self.top_ideas)
        console.print(f"[bold green]✓ Idea Generation complete. Found {len(self.top_ideas)} ideas.[/bold green]\n")
        self.stage_times["ideas_generation"] = time.time() - start_time
        
        
    def build_implementation_task(self, idx: int, idea: dict, output_dir: str = None) -> dict:
        idea_name = idea.get('name', f'Idea_{idx}')
        idea_desc = idea.get('description', '')
        safe_idea_name = re.sub(r'[^\w\-]', '_', idea_name)
        base_dir = output_dir if output_dir is not None else self.implementation_dir
        idea_dir = os.path.join(base_dir, f"idea_{idx}_{safe_idea_name}")
        os.makedirs(idea_dir, exist_ok=True)

        prompt = (
            f"{self.prompts.system_generator}\n"
            f"{self.prompts.output_format}\n\n"
            f"ALGORITHM IDEA TO IMPLEMENT: {idea_name}\nDescription: {idea_desc}\n"
            f"{self.prompts.variables}"
        )
        save_log(idea_dir, "prompt.txt", prompt, quiet=True)

        return {
            "idx": idx,
            "idea_name": idea_name,
            "idea_desc": idea_desc,
            "safe_idea_name": safe_idea_name,
            "idea_dir": idea_dir,
            "prompt": prompt,
        }

    def implementation(self):
        """Stage 3 entry point — implements & evaluates self.top_ideas. Stage-3
        evals do NOT debit ``self.budget`` (those are seed work, not evolution FE).
        """
        start_time = time.time()
        ideas = self.top_ideas[:getattr(self.config, 'top_ideas_to_implement', 5)]
        results = self._run_implementations(
            ideas=ideas,
            output_dir=self.implementation_dir,
            stage_label="implementation",
            count_against_budget=False,
            heading="Implementation",
        )
        save_json(self.implementation_dir, "final_results.json", results)
        self.stage_times["implementation"] = time.time() - start_time
        return results

    def _run_implementations(
        self,
        ideas: list,
        output_dir: str,
        stage_label: str,
        count_against_budget: bool,
        heading: str,
    ) -> dict:
        """Shared core of implementation() — also called by _reideation_phase
        with ``count_against_budget=True`` so re-ideation evals debit
        ``self.budget``."""
        console.print(Rule(heading, style="bold green"))
        all_results: dict = {}
        if not ideas:
            return all_results

        tasks = [self.build_implementation_task(idx, idea, output_dir=output_dir) for idx, idea in enumerate(ideas)]

        workers = max(1, min(getattr(self.config, 'implementation_workers', 4), len(tasks)))
        completed: list[dict] = [None] * len(tasks)
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            bar = progress.add_task(f"implementations ({workers} workers)", total=len(tasks))

            def _run(t):
                result = self._fetch_llm_response_in_stage(t, stage_label)
                progress.advance(bar)
                return result

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {pool.submit(_run, t): t["idx"] for t in tasks}
                for fut in future_to_idx:
                    idx = future_to_idx[fut]
                    completed[idx] = fut.result()

        target_func = "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"
        for task in completed:
            idea_name = task["idea_name"]
            idea_desc = task["idea_desc"]
            idea_dir = task["idea_dir"]
            safe_idea_name = task["safe_idea_name"]
            response = task["response"]

            code = IdeaParser.extract_code(response or "", target_func)

            call_snapshot = task.get("call_snapshot")

            if not code:
                console.print(f"[red]Failed to extract code for {idea_name}[/red]")
                save_json(idea_dir, "evaluation_results.json", {"error": "Extraction Failed"})
                if count_against_budget:
                    self.budget.increment()
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed", "code": "", "description": idea_desc}
                self._record_eval(stage_label, idea_name, {"mean_swaps": None, "mean_depth": None, "error": "Extraction Failed"}, call_snapshot=call_snapshot)
                continue

            with open(os.path.join(idea_dir, "heuristic.py"), "w") as f:
                f.write(code)

            stats = self.evaluator.evaluate(code)
            if count_against_budget:
                self.budget.increment()
            save_json(idea_dir, "evaluation_results.json", stats)
            self._record_eval(stage_label, idea_name, stats if isinstance(stats, dict) else {}, call_snapshot=call_snapshot)

            self.memory.add_idea(
                idea_name,
                idea_desc,
                code,
                stats.get('mean_swaps', float('inf')),
                stats.get('mean_depth', float('inf')),
                stats.get('error'),
                run_id=self.run_id,
            )
            log_event(
                stage_label,
                "idea_evaluated",
                idea=idea_name,
                mean_swaps=stats.get('mean_swaps'),
                mean_depth=stats.get('mean_depth'),
                error=stats.get('error'),
            )

            entry = {"code": code, "description": idea_desc, **stats}
            if not stats.get('error'):
                console.print(f"[green]✔ SUCCESS {idea_name}: {stats['mean_swaps']:.2f} avg swaps[/green]")
                _save_heuristic(self.config.log_dir, safe_idea_name, code)
            else:
                console.print(f"[red]✘ FAILED {idea_name}: {stats.get('error')}[/red]")
            all_results[idea_name] = entry

        return all_results

    def _fetch_llm_response_in_stage(self, task: dict, stage_label: str) -> dict:
        with self.llm.stage(stage_label):
            response, call_record = self.llm.query(task["prompt"], reset_conversation=True, show_counter=False)
        save_log(task["idea_dir"], "raw_response.txt", response or "", quiet=True)
        return {**task, "response": response, "call_snapshot": call_record}

    def evolution(self, initial_results):
        """Outer cycle: run ONE evolution iteration (crossover + single mutation),
        then re-ideate against the current KG to inject novel directions, then
        reseed and run the next iteration. No stagnation gate — every cycle
        re-enters ideation/implementation as long as the FE budget allows."""

        start_time = time.time()
        if not getattr(self.config, "run_evolution", True):
            console.print("[yellow]Skipping Evolution per config.[/yellow]")
            self.stage_times["evolution"] = 0.0
            return initial_results

        pop_size = getattr(self.config, "pop_size", 10)
        crossover_count = getattr(self.config, "crossover_count", 5)
        use_crossover = getattr(self.config, "use_crossover", True)
        use_mutation = getattr(self.config, "use_mutation", True)
        use_reideation = getattr(self.config, "use_reideation", True)
        # One evolve iteration costs the FE of whichever operators are enabled:
        # `crossover_count` children (if crossover on) + 1 mutant (if mutation on).
        iter_cost = max(1, (crossover_count if use_crossover else 0) + (1 if use_mutation else 0))

        initial_population = self._build_initial_population(initial_results, pop_size)
        console.print(
            f"[cyan]Evolution seed pulled from global memory: "
            f"{len(initial_population)} individuals (top-{pop_size}).[/cyan]"
        )

        all_results = dict(initial_results or {})
        last_loop: EvolutionLoop = None
        # Running iteration index shared across outer cycles. Each fresh loop
        # resumes numbering here, so iter_* dirs stop overwriting one another
        # and last_loop.iteration reports the true total below.
        next_iteration = 0

        while self.budget.remaining() >= iter_cost:
            loop = EvolutionLoop(
                config=self.config,
                prompts=self.prompts,
                llm=self.llm,
                evaluator=self.evaluator,
                memory=self.memory,
                run_id=self.run_id,
                evolution_dir=self.evolution_dir,
                kg=self.kg,
                budget=self.budget,
                on_eval=self._record_eval,
                start_iteration=next_iteration,
            )
            loop.seed(initial_population)
            # Re-ideation OFF: let a single loop run crossover+mutation until the
            # FE budget is exhausted (no novelty injection between iterations).
            final_population = loop.evolve(max_iterations=1 if use_reideation else None)
            next_iteration = loop.iteration
            last_loop = loop

            for ind in final_population:
                all_results[ind["name"]] = {
                    "code": ind["code"],
                    "mean_swaps": ind["mean_swaps"],
                    "mean_depth": ind["mean_depth"],
                    "error": ind["error"],
                }

            # Mirror final KG state to memory for cross-run continuity.
            self.memory.set_knowledge_graph(self.kg.to_dict())

            if not use_reideation:
                log_event("evolution", "reideation_disabled_loop_to_budget", remaining=self.budget.remaining())
                break

            # Budget gate: need room for one more evolve iteration after re-ideation.
            if self.budget.remaining() < iter_cost:
                log_event("evolution", "cycle_skipped_low_budget", remaining=self.budget.remaining())
                break

            new_seeds, n_success, reideation_results = self._reideation_phase(self.kg.all_explored_statements())

            # Re-ideation evals debit the budget and live in memory/per-idea logs,
            # but they only enter `all_results` via the *next* loop's final_population.
            # If the budget exhausts before that loop runs, the evaluations are paid
            # for but invisible to the final report. Merge them in directly.
            for name, stats in reideation_results.items():
                all_results[name] = stats

            if n_success == 0:
                self._consecutive_failed_restarts += 1
                if self._consecutive_failed_restarts >= 2:
                    log_event(
                        "evolution",
                        "reideation_aborted_no_success",
                        consecutive_failures=self._consecutive_failed_restarts,
                    )
                    break
            else:
                self._consecutive_failed_restarts = 0

            elite_carry = loop.best_individual_overall
            initial_population = ([elite_carry] if elite_carry else []) + new_seeds

        if last_loop is None:
            # Loop never ran (budget < pop_size from the start).
            self.stage_times["evolution"] = time.time() - start_time
            return all_results

        # Save the true overall-best heuristic from all_results — this includes
        # re-ideation evaluations, which last_loop.best_individual_overall misses.
        best_name = None
        best_obj = float("inf")
        for name, stats in all_results.items():
            if stats.get("error") is not None or not stats.get("code"):
                continue
            ms = stats.get("mean_swaps")
            if ms is None or ms == float("inf"):
                continue
            if ms < best_obj:
                best_obj = ms
                best_name = name
        if best_name is not None:
            safe = re.sub(r"[^\w\-]", "_", best_name)
            _save_heuristic(self.config.log_dir, f"best_evolution_{safe}", all_results[best_name]["code"])

        # Final artifacts — final_population from the last loop, plus KG path.
        final_population = last_loop.population
        save_json(self.evolution_dir, "final_population.json", [
            {k: v for k, v in ind.items() if k != "code"} for ind in final_population
        ])
        kg_path = os.path.join(self.evolution_dir, "knowledge_graph_final.json")
        save_json(self.evolution_dir, "knowledge_graph_final.json", self.kg.to_dict())

        # Recompute the true best across BOTH loop-evolved individuals and any
        # re-ideation evaluations now merged into all_results. `last_loop`'s
        # internal best only ever sees crossover/mutation children.
        best_name = None
        best_obj = float("inf")
        for name, stats in all_results.items():
            if stats.get("error") is not None:
                continue
            ms = stats.get("mean_swaps")
            if ms is None or ms == float("inf"):
                continue
            if ms < best_obj:
                best_obj = ms
                best_name = name

        save_json(self.evolution_dir, "summary.json", {
            "function_evals": self.budget.current,
            "max_fe": self.config.max_fe,
            "elapsed_seconds": round(self.budget.elapsed(), 1),
            "max_seconds": self.budget.max_seconds,
            "stop_reason": self.budget.stop_reason(),
            "iterations": last_loop.iteration,
            "best_obj_overall": best_obj if best_obj != float("inf") else None,
            "best_name": best_name,
            "knowledge_graph_path": kg_path,
            "reideation_cycles": self._reideation_count,
            "consecutive_failed_restarts": self._consecutive_failed_restarts,
        })
        self.stage_times["evolution"] = time.time() - start_time
        return all_results

    def _build_initial_population(self, initial_results: dict, pop_size: int) -> list:
        """Pull from global memory; fall back to current Stage 3 output if memory is empty."""
        top_memory = self.memory.get_top_k(k=pop_size)
        out: list = []
        for entry in top_memory:
            code = entry.get("code")
            if not code:
                continue
            out.append({
                "name": entry["name"],
                "description": entry.get("description", ""),
                "code": code,
                "mean_swaps": entry.get("mean_swaps", float("inf")),
                "mean_depth": entry.get("mean_depth", 0) or 0,
                "error": entry.get("error"),
            })
        if not out:
            for name, stats in (initial_results or {}).items():
                out.append({
                    "name": name,
                    "description": stats.get("description", ""),
                    "code": stats.get("code", ""),
                    "mean_swaps": stats.get("mean_swaps", float("inf")),
                    "mean_depth": stats.get("mean_depth", 0) or 0,
                    "error": stats.get("error"),
                })
        return out

    def _reideation_phase(self, explored_statements: list) -> tuple:
        """Generate fresh ideas that avoid every direction the KG already
        tracks (open / confident / falsified / exhausted), implement +
        evaluate them (debits ``self.budget``), and return surviving seeds,
        the count of successful implementations, and the full impl_results
        dict (so the caller can merge it into ``all_results`` — needed when
        the budget exhausts before the next evolve loop runs)."""
        self._reideation_count += 1
        console.print(Rule(
            f"Re-ideation cycle #{self._reideation_count}",
            style="bold red",
        ))
        log_event(
            "evolution",
            "reideation_phase_start",
            cycle=self._reideation_count,
            explored_count=len(explored_statements),
            budget_remaining=self.budget.remaining(),
        )

        explored_block = (
            "\n".join(f"- {s}" for s in explored_statements)
            if explored_statements
            else "(no strategies explored yet — push for diverse directions anyway)"
        )
        target_ideas = max(2, getattr(self.config, "target_top_ideas", 5))

        prompt = (
            f"{self.prompts.system_generator}\n"
            + self.prompts.ideas_regeneration_prompt.format(
                explored_strategies=explored_block,
                variables=self.prompts.variables,
                target_top_ideas=target_ideas,
            )
        )
        with self.llm.stage(f"evolution/reideation_{self._reideation_count}/idea_generation"):
            response, _ = self.llm.query(prompt, reset_conversation=True)
        reideation_dir = os.path.join(self.evolution_dir, f"reideation_{self._reideation_count}")
        os.makedirs(reideation_dir, exist_ok=True)
        save_log(reideation_dir, "ideas_prompt.txt", prompt, quiet=True)
        save_log(reideation_dir, "ideas_raw.txt", response or "", quiet=True)

        kept, _eliminated = IdeaParser.parse_ideas(response or "")
        if not kept:
            log_event("evolution", "reideation_phase_no_ideas", cycle=self._reideation_count)
            return [], 0, {}

        # Cap by budget — never queue more than budget can afford.
        max_by_budget = self.budget.remaining()
        new_ideas = kept[:min(target_ideas, max_by_budget)]

        impl_results = self._run_implementations(
            ideas=new_ideas,
            output_dir=reideation_dir,
            stage_label=f"reideation_{self._reideation_count}",
            count_against_budget=True,
            heading=f"Re-ideation implementations ({self._reideation_count})",
        )

        seeds: list = []
        n_success = 0
        for name, stats in impl_results.items():
            if stats.get("error") is None and stats.get("mean_swaps") not in (None, float("inf")):
                n_success += 1
                seeds.append({
                    "name": name,
                    "description": stats.get("description", ""),
                    "code": stats.get("code", ""),
                    "mean_swaps": stats.get("mean_swaps"),
                    "mean_depth": stats.get("mean_depth", 0) or 0,
                    "error": None,
                })
                # Post-hoc hypothesis: first sentence of description, open + 0.5.
                desc = stats.get("description") or name
                first_sentence = re.split(r"(?<=[.!?])\s+", desc.strip(), maxsplit=1)[0].strip()
                if first_sentence and getattr(self.config, "use_kg", True):
                    self.kg.register_hypothesis(first_sentence, related_trait_labels=[], iter_idx=-1)

        log_event(
            "evolution",
            "reideation_phase_end",
            cycle=self._reideation_count,
            attempted=len(new_ideas),
            n_success=n_success,
            budget_remaining=self.budget.remaining(),
        )
        return seeds, n_success, impl_results

    def _build_summary(self, all_results, idea_bests, best_idea, successful_ideas) -> RunSummary:
        ranking = []
        per_run_heur = os.path.join(self.config.log_dir, "heuristics")
        for rank, ib in enumerate(idea_bests, 1):
            safe = re.sub(r'[^\w\-]', '_', ib['name'])
            candidate = os.path.join(per_run_heur, f"{safe}.py")
            code_path = candidate if os.path.exists(candidate) else None
            ranking.append(RankingEntry(
                rank=rank,
                idea_name=ib['name'],
                mean_swaps=float(ib['mean_swaps']) if ib['mean_swaps'] != float('inf') else float('inf'),
                mean_depth=float(ib.get('mean_depth', 0) or 0),
                error=ib.get('error'),
                code_path=code_path,
            ))

        best = None
        if best_idea:
            safe = re.sub(r'[^\w\-]', '_', best_idea['name'])
            candidate = os.path.join(per_run_heur, f"{safe}.py")
            best = BestEntry(
                idea_name=best_idea['name'],
                mean_swaps=float(best_idea['mean_swaps']),
                code_path=candidate if os.path.exists(candidate) else None,
            )

        token_totals = {"total": int(self.llm.total_tokens)}

        return RunSummary(
            run_id=self.run_id,
            status="ok" if successful_ideas else "no_success",
            stage_timings_seconds={k: float(v) for k, v in self.stage_times.items()},
            token_totals=token_totals,
            counts={
                "ideas_evaluated": len(all_results),
                "ideas_succeeded": len(successful_ideas),
            },
            ranking=ranking,
            best=best,
        )

    def generate_final_report(self, all_results):
        """Generates the final summary metrics across all evaluated heuristics."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"\nCross-Idea Comparison Report\nGenerated: {timestamp}\nIdeas Tested: {len(all_results)}\n"

        successful_ideas = []
        idea_bests = []
        
        for name, stats in all_results.items():
            mean_swaps = stats.get('mean_swaps', float('inf'))
            mean_depth = stats.get('mean_depth', 0)
            error = stats.get('error')
            
            idea_bests.append({
                'name': name,
                'mean_swaps': mean_swaps,
                'mean_depth': mean_depth,
                'error': error
            })
            
            if mean_swaps != float('inf') and error is None:
                successful_ideas.append(idea_bests[-1])

        generation_success_rate = (len(successful_ideas) / len(all_results)) * 100 if all_results else 0
        
        if successful_ideas:
            swaps_list = [ib['mean_swaps'] for ib in successful_ideas]
            avg_swaps = sum(swaps_list) / len(swaps_list)
            best_swaps = min(swaps_list)
            worst_swaps = max(swaps_list)
            best_idea = min(successful_ideas, key=lambda x: x['mean_swaps'])
        else:
            avg_swaps = float('inf')
            best_swaps = float('inf')
            worst_swaps = float('inf')
            best_idea = None

        def _fmt(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) and x != float('inf') else "N/A"

        report += "\n## Global Metrics\n"
        report += f"- **Generation Success Rate:** {generation_success_rate:.2f}%\n"
        report += f"- **Average Score (Mean Swaps):** {_fmt(avg_swaps)}\n"
        report += f"- **Best Score:** {_fmt(best_swaps)}\n"
        report += f"- **Worst Score:** {_fmt(worst_swaps)}\n"
        report += f"- **Total Number of Tokens:** {self.llm.total_tokens:,}\n"
        
        def format_time(seconds):
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            if h > 0: return f"{int(h)}h {int(m)}m {int(s)}s"
            if m > 0: return f"{int(m)}m {int(s)}s"
            return f"{s:.2f}s"

        report += "\n## Execution Times\n"
        report += f"- **Literature Review:** {format_time(self.stage_times.get('literature_review', 0))}\n"
        report += f"- **Idea Generation:** {format_time(self.stage_times.get('ideas_generation', 0))}\n"
        report += f"- **Implementation:** {format_time(self.stage_times.get('implementation', 0))}\n"
        report += f"- **Evolution:** {format_time(self.stage_times.get('evolution', 0))}\n"
        report += f"- **Total Pipeline:** {format_time(self.stage_times.get('total_pipeline', 0))}\n"

        idea_bests.sort(key=lambda x: x['mean_swaps'])

        report += "\n## Idea Ranking (by Mean Swaps)\n"
        for rank, ib in enumerate(idea_bests, 1):
            status = "✔" if ib['error'] is None else "✘"
            report += (
                f"\n### {rank}. {status} {ib['name']}\n"
                f"- Mean Swaps: {ib['mean_swaps'] if ib['error'] is None else 'FAILED'}\n"
                f"- Mean Depth: {ib['mean_depth']}\n"
            )

        if best_idea:
            report += f"\n## Overall Winner: {best_idea['name']} (Mean Swaps: {best_idea['mean_swaps']:.2f})\n"
        else:
            report += "\n## No successful ideas.\n"

        summary = self._build_summary(all_results, idea_bests, best_idea, successful_ideas)
        summary_path = os.path.join(self.config.log_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        console.print(f"[dim]  -> Saved {summary_path}[/dim]")

        # Tokens-vs-metrics artifacts: raw trace + call log always, PNG if enabled.
        save_json(self.config.log_dir, "tokens_vs_metrics.json", {
            "run_id": self.run_id,
            "token_totals": summary.token_totals,
            "eval_trace": self.eval_trace,
            "call_log": self.llm.call_log,
        })

        if getattr(self.config, "generate_plots", True) and self.eval_trace:
            try:
                plot_path = os.path.join(self.config.log_dir, "tokens_vs_metrics.png")
                plot_tokens_vs_metrics(self.eval_trace, plot_path, run_id=self.run_id)
                console.print(f"[dim]  -> Saved {plot_path}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: tokens_vs_metrics plot failed: {e}[/yellow]")

        return report

    def run(self):
        self.pipeline_start_time = time.time()
        log_event("literature_review", "stage_start")
        self.literature_review()
        log_event("literature_review", "stage_end", seconds=self.stage_times["literature_review"])

        log_event("ideas_generation", "stage_start")
        self.ideas_generation()
        log_event("ideas_generation", "stage_end", seconds=self.stage_times["ideas_generation"])

        log_event("implementation", "stage_start")
        stage3_results = self.implementation()
        log_event("implementation", "stage_end", seconds=self.stage_times["implementation"])

        log_event("evolution", "stage_start")
        final_results = self.evolution(stage3_results)
        log_event("evolution", "stage_end", seconds=self.stage_times["evolution"])

        self.stage_times["total_pipeline"] = time.time() - self.pipeline_start_time
        console.print("[bold]Generating final cross-idea comparison report...[/bold]")
        final_report = self.generate_final_report(final_results)
        save_log(self.config.log_dir, "final_pipeline_report.txt", final_report)
        console.print(Panel(final_report, title="[bold]Final Pipeline Comparison[/bold]", border_style="green"))

        log_event("pipeline", "run_completed", total_seconds=self.stage_times["total_pipeline"], total_tokens=self.llm.total_tokens)
        console.print(Rule("Pipeline Complete", style="bold white"))