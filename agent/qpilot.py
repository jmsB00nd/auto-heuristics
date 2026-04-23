import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel

from .config import OrchestratorConfig
from .prompt_manager import PromptManager
from .llm_client import LLMClient
from .idea_parser import IdeaParser
from .evaluator import CodeEvaluator
from .memory import MemoryManager
from .run_context import build_run_metadata, write_run_metadata
from .schemas import BestEntry, RankingEntry, RunSummary
from .plotting import plot_tokens_vs_metrics
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
            "reflection": 0.0,
            "iterative_refinement": 0.0,
            "total_pipeline": 0.0
        }
        
        self.pipeline_start_time = None
        
        # Setup Dirs
        self.literature_review_dir = os.path.join(self.config.log_dir, "literature_review")
        self.ideas_dir = os.path.join(self.config.log_dir, "idea_generation")
        self.implementation_dir = os.path.join(self.config.log_dir, "implementation")
        self.reflection_dir = os.path.join(self.config.log_dir, "reflection")
        self.iterative_refinement_dir = os.path.join(self.config.log_dir, "iterative_refinement")
        for d in [self.literature_review_dir, self.ideas_dir, self.implementation_dir, self.reflection_dir, self.iterative_refinement_dir]:
            os.makedirs(d, exist_ok=True)

    def _record_eval(self, stage: str, label: str, stats: dict) -> None:
        """Append one row to ``eval_trace`` capturing cumulative tokens at this eval."""
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
        self.eval_trace.append({
            "stage": stage,
            "label": label,
            "cumulative_input_tokens": int(self.llm.usage_totals["input_tokens"]),
            "cumulative_output_tokens": int(self.llm.usage_totals["output_tokens"]),
            "cumulative_total_tokens": self.llm.total_tokens,
            "cumulative_cost_usd": float(self.llm.usage_totals["total_cost_usd"]),
            "mean_swaps": mean_swaps,
            "mean_depth": mean_depth,
            "error": error,
        })

    def literature_review(self):
        start_time = time.time()
        console.print(Rule("Literature Review", style="bold cyan"))

        if not getattr(self.config, 'run_stage1_literature_review', False):
            console.print("[yellow]Skipping Literature Review per config.[/yellow]")
            return

        with self.llm.stage("literature_review"):
            response = self.llm.query(self.prompts.lit_review_prompt, reset_conversation=True)
        self.literature_insights = response or ""

        if response:
            save_log(self.literature_review_dir, "literature_review.txt", response)
            console.print("[bold green]✓ Stage I complete.[/bold green]\n")

        self.stage_times["literature_review"] = time.time() - start_time

    def ideas_generation(self):
        start_time = time.time()
        console.print(Rule("Idea Generation", style="bold magenta"))

        memory_dump = self.memory.get_all_summarized()
        memory_resume = ""

        with self.llm.stage("ideas_generation"):
            if memory_dump != "No past ideas in memory.":
                console.print("[cyan]Synthesizing all past memory into a global resume...[/cyan]")

                summary_prompt = (
                    f"{self.prompts.system_generator}\n"
                    f"{self.prompts.memory_summary_prompt}\n"
                    f"PAST EXPERIMENTS LOG:\n{memory_dump}\n"
                )

                save_log(self.ideas_dir, "memory_summary_prompt.txt", summary_prompt)

                memory_resume = self.llm.query(summary_prompt, reset_conversation=True)
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
                f"{self.prompts.variables}"
            )

            save_log(self.ideas_dir, "prompt.txt", prompt)

            console.print("[cyan]Generating new novel ideas based on global context...[/cyan]")
            response = self.llm.query(prompt, reset_conversation=not getattr(self.config, 'use_conversation_mode', True))
            save_log(self.ideas_dir, "raw_ideas.txt", response or "")

        # Parse ideas
        kept, eliminated = IdeaParser.parse_ideas(response or "")
        self.top_ideas = kept[:getattr(self.config, 'target_top_ideas', 5)]
        
        save_json(self.ideas_dir, "top_ideas.json", self.top_ideas)
        console.print(f"[bold green]✓ Idea Generation complete. Found {len(self.top_ideas)} ideas.[/bold green]\n")
        self.stage_times["ideas_generation"] = time.time() - start_time
        
        
    def _build_implementation_task(self, idx: int, idea: dict) -> dict:
        idea_name = idea.get('name', f'Idea_{idx}')
        idea_desc = idea.get('description', '')
        safe_idea_name = re.sub(r'[^\w\-]', '_', idea_name)
        idea_dir = os.path.join(self.implementation_dir, f"idea_{idx}_{safe_idea_name}")
        os.makedirs(idea_dir, exist_ok=True)

        prompt = (
            f"{self.prompts.system_generator}\n"
            f"{self.prompts.output_format}\n\n"
            f"ALGORITHM IDEA TO IMPLEMENT: {idea_name}\nDescription: {idea_desc}\n"
            f"{self.prompts.variables}"
        )
        save_log(idea_dir, "prompt.txt", prompt)

        return {
            "idx": idx,
            "idea_name": idea_name,
            "idea_desc": idea_desc,
            "safe_idea_name": safe_idea_name,
            "idea_dir": idea_dir,
            "prompt": prompt,
        }

    def _fetch_llm_response(self, task: dict) -> dict:
        """Worker executed in parallel — LLM call only, no shared mutation
        beyond the thread-safe ``LLMClient.query``."""
        console.print(f"[bold]Implementing {task['idea_name']}...[/bold]")
        with self.llm.stage("implementation"):
            response = self.llm.query(task["prompt"], reset_conversation=True)
        save_log(task["idea_dir"], "raw_response.txt", response or "")
        return {**task, "response": response}

    def implementation(self):
        start_time = time.time()
        console.print(Rule("Implementation", style="bold green"))
        ideas_to_implement = self.top_ideas[:getattr(self.config, 'top_ideas_to_implement', 5)]
        all_results = {}

        if not ideas_to_implement:
            save_json(self.implementation_dir, "final_results.json", all_results)
            self.stage_times["implementation"] = time.time() - start_time
            return all_results

        tasks = [self._build_implementation_task(idx, idea) for idx, idea in enumerate(ideas_to_implement)]

        # Phase 1: fan out LLM calls in parallel. Subprocess-backed CLI calls
        # spend most of their time waiting on I/O, so threads overlap well.
        workers = max(1, min(getattr(self.config, 'implementation_workers', 4), len(tasks)))
        console.print(f"[cyan]Dispatching {len(tasks)} LLM calls across {workers} workers...[/cyan]")
        completed: list[dict] = [None] * len(tasks)  # preserve original idea order
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(self._fetch_llm_response, t): t["idx"] for t in tasks}
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                completed[idx] = fut.result()

        # Phase 2: extract code, evaluate, and record memory serially. Evaluator
        # uses mp.get_context("fork"), which is unsafe to call from worker threads
        # while other threads may hold locks — keep this on the main thread.
        target_func = "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"
        for task in completed:
            idea_name = task["idea_name"]
            idea_desc = task["idea_desc"]
            idea_dir = task["idea_dir"]
            safe_idea_name = task["safe_idea_name"]
            response = task["response"]

            code = IdeaParser.extract_code(response or "", target_func)

            if not code:
                console.print(f"[red]Failed to extract code for {idea_name}[/red]")
                save_json(idea_dir, "evaluation_results.json", {"error": "Extraction Failed"})
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed", "code": ""}
                self._record_eval("implementation", idea_name, {"mean_swaps": None, "mean_depth": None, "error": "Extraction Failed"})
                continue

            with open(os.path.join(idea_dir, "heuristic.py"), "w") as f:
                f.write(code)

            stats = self.evaluator.evaluate(code)
            save_json(idea_dir, "evaluation_results.json", stats)
            self._record_eval("implementation", idea_name, stats if isinstance(stats, dict) else {})

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
                "implementation",
                "idea_evaluated",
                idea=idea_name,
                mean_swaps=stats.get('mean_swaps'),
                mean_depth=stats.get('mean_depth'),
                error=stats.get('error'),
            )

            if not stats.get('error'):
                console.print(f"[green]✔ SUCCESS {idea_name}: {stats['mean_swaps']:.2f} avg swaps[/green]")
                all_results[idea_name] = {"code": code, **stats}
                _save_heuristic(self.config.log_dir, safe_idea_name, code)
            else:
                console.print(f"[red]✘ FAILED {idea_name}: {stats.get('error')}[/red]")
                all_results[idea_name] = {"code": code, **stats}

        save_json(self.implementation_dir, "final_results.json", all_results)
        self.stage_times["implementation"] = time.time() - start_time
        return all_results

    def reflection(self):
        start_time = time.time()
        console.print(Rule("Reflection", style="bold yellow"))
        
        if not getattr(self.config, 'run_stage3_5_reflection', False):
            console.print("[yellow]Skipping Reflection per config.[/yellow]")
            return

        top_ideas = self.memory.get_top_k(3)

        if not top_ideas:
            console.print("[yellow]No successful ideas to reflect upon.[/yellow]")
            return

        blocks = []
        for rank, idea in enumerate(top_ideas, 1):
            blocks.append(
                f"TOP-{rank} HEURISTIC\n"
                f"  Name: {idea['name']}\n"
                f"  Score (mean_swaps): {idea['mean_swaps']:.3f}\n"
                f"  Description: {(idea.get('description') or '')[:200]}\n"
                f"  Code:\n```python\n{idea['code']}\n```\n"
            )

        reflection_context = "\n".join(blocks)
        prompt = f"{self.prompts.reflection_prompt}\n\n{reflection_context}"
        save_log(self.reflection_dir, "prompt.txt", prompt)
        with self.llm.stage("reflection"):
            response = self.llm.query(prompt, reset_conversation=True)
        
        if response:
            self.memory.add_reflection(response)
            save_log(self.reflection_dir, "reflection_insights.txt", response)
            console.print("[bold green]✓ Reflection complete. Insights added to memory.[/bold green]\n")
            
        self.stage_times["reflection"] = time.time() - start_time

    def iterative_refinement(self, initial_results):
        start_time = time.time()
        console.print(Rule("Iterative Refinement", style="bold blue"))

        if not getattr(self.config, 'run_stage4_iterative_refinement', False):
            console.print("[yellow]Skipping Iterative Refinement per config.[/yellow]")
            return initial_results

        all_results = dict(initial_results)
        refinement_rounds = getattr(self.config, 'refinement_rounds', 3)
        stagnation_threshold = getattr(self.config, 'stagnation_threshold', 3)
        diversity_pool_size = getattr(self.config, 'diversity_pool_size', 5)
        improvement_history = []
        force_diversity = False

        for round_idx in range(1, refinement_rounds + 1):
            top_memory_ideas = self.memory.get_top_k(2)
            if not top_memory_ideas:
                console.print("[red]No successful ideas in memory to refine. Exiting stage.[/red]")
                break

            current_best = top_memory_ideas[0]
            best_score = current_best['mean_swaps']

            console.print(f"\n[bold]Refinement Round {round_idx}/{refinement_rounds} (Current Best Score: {best_score:.2f})[/bold]")
            round_dir = os.path.join(self.iterative_refinement_dir, f"round_{round_idx}")
            os.makedirs(round_dir, exist_ok=True)

            reflection_insight = self.memory.get_latest_reflection()
            reflection_str = f"\nREFLECTION INSIGHTS:\n{reflection_insight}\n" if reflection_insight else ""

            stagnation_banner = ""
            if force_diversity:
                stagnation_banner = (
                    "\n[STAGNATION FLAG] The last "
                    f"{stagnation_threshold} rounds produced no improvement. "
                    "Propose a qualitatively different algorithmic approach instead of "
                    "refining the current primitive.\n"
                )

            # Decide Operation: Mutation or Crossover
            parents = []
            if len(top_memory_ideas) >= 2 and random.random() < getattr(self.config, 'crossover_rate', 0.5):
                parents = self.memory.get_diverse_parents(k=2)

            if len(parents) >= 2:
                operation = "CROSSOVER"
                parent1, parent2 = parents[0], parents[1]
                context = (
                    f"PARENT 1: {parent1['name']} — mean_swaps={parent1['mean_swaps']:.3f}\n"
                    f"  Summary: {(parent1.get('description') or '')[:200]}\n"
                    f"  Code:\n```python\n{parent1['code']}\n```\n\n"
                    f"PARENT 2: {parent2['name']} — mean_swaps={parent2['mean_swaps']:.3f}\n"
                    f"  Summary: {(parent2.get('description') or '')[:200]}\n"
                    f"  Code:\n```python\n{parent2['code']}\n```\n"
                )
                prompt_template = getattr(self.prompts, 'crossover_prompt', 'Combine these two algorithms into a better one.')
            else:
                operation = "MUTATION"
                if force_diversity:
                    pool = self.memory.get_top_k(diversity_pool_size * 2)
                    pool = pool[len(pool) // 2:] or pool
                    mutation_target = random.choice(pool)
                else:
                    top_pool = self.memory.get_top_k(diversity_pool_size)
                    mutation_idx = (round_idx - 1) % len(top_pool)
                    mutation_target = top_pool[mutation_idx]
                context = (
                    f"PARENT: {mutation_target['name']} — mean_swaps={mutation_target['mean_swaps']:.3f}\n"
                    f"  Summary: {(mutation_target.get('description') or '')[:200]}\n"
                    f"  Code:\n```python\n{mutation_target['code']}\n```\n"
                )
                prompt_template = getattr(self.prompts, 'refinement_prompt', 'Improve the current best heuristic.')

            if force_diversity:
                console.print(f"[yellow]Diversity mode active — exploring beyond top performers[/yellow]")

            console.print(f"[magenta]Applying Operation: {operation}[/magenta]")

            variables_ref = getattr(self.prompts, 'variables', '') or self.prompts.code

            prompt = (
                f"{stagnation_banner}"
                f"{reflection_str}\n\n"
                f"{context}\n"
                f"{prompt_template}\n"
                f"{variables_ref}"
            )

            save_log(round_dir, "prompt.txt", prompt)
            with self.llm.stage(f"refinement/round_{round_idx}"):
                response = self.llm.query(prompt, reset_conversation=True)
            save_log(round_dir, "raw_response.txt", response or "")

            if response is None:
                console.print(f"[red]LLM call failed in round {round_idx}[/red]")
                fail_name = f"Refined_{operation}_R{round_idx}"
                all_results[fail_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "LLM call failed"}
                self._record_eval(f"refinement/round_{round_idx}", fail_name,
                                  {"mean_swaps": None, "mean_depth": None, "error": "LLM call failed"})
                improvement_history.append(False)
                log_event("refinement", "llm_failed", round=round_idx, operation=operation)
                continue

            # Extract custom Name and Description
            name_match = re.search(r'NAME:\s*(.+)', response, re.IGNORECASE)
            desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\n[A-Z_]{3,}\s*:|\n```|$)', response, re.IGNORECASE | re.DOTALL)

            idea_name = name_match.group(1).strip() if name_match else f"Refined_{operation}_R{round_idx}"
            idea_desc = desc_match.group(1).strip() if desc_match else f"Generated via {operation}"

            target_func = "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"
            code = IdeaParser.extract_code(response, target_func)

            if not code:
                console.print(f"[red]Failed to extract code in round {round_idx}[/red]")
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed"}
                self._record_eval(f"refinement/round_{round_idx}", idea_name,
                                  {"mean_swaps": None, "mean_depth": None, "error": "Extraction Failed"})
                improvement_history.append(False)
                continue

            with open(os.path.join(round_dir, "heuristic.py"), "w") as f:
                f.write(code)

            # Evaluate
            stats = self.evaluator.evaluate(code)
            save_json(round_dir, "evaluation_results.json", stats)
            self._record_eval(f"refinement/round_{round_idx}", idea_name,
                              stats if isinstance(stats, dict) else {})

            self.memory.add_idea(
                name=idea_name,
                description=idea_desc,
                code=code,
                mean_swaps=stats.get('mean_swaps', float('inf')),
                mean_depth=stats.get('mean_depth', 0),
                error=stats.get('error'),
                run_id=self.run_id,
            )
            log_event(
                "refinement",
                "round_evaluated",
                round=round_idx,
                operation=operation,
                idea=idea_name,
                mean_swaps=stats.get('mean_swaps'),
                error=stats.get('error'),
            )

            if not stats.get('error'):
                current_score = stats['mean_swaps']
                console.print(f"[green]✔ SUCCESS: {current_score:.2f} avg swaps[/green]")
                all_results[idea_name] = stats

                if current_score < best_score:
                    console.print(f"[bold green] New Best Score! {best_score:.2f} -> {current_score:.2f}[/bold green]")
                    _save_heuristic(self.config.log_dir, f"best_refined_round_{round_idx}", code)
                    log_event(
                        "refinement",
                        "new_best",
                        round=round_idx,
                        previous=best_score,
                        current=current_score,
                    )
                    improvement_history.append(True)
                    force_diversity = False
                else:
                    improvement_history.append(False)
            else:
                console.print(f"[red]✘ FAILED: {stats.get('error')}[/red]")
                all_results[idea_name] = stats
                improvement_history.append(False)

            # Check for stagnation
            recent = improvement_history[-stagnation_threshold:]
            if len(recent) >= stagnation_threshold and not any(recent):
                console.print(f"[yellow]Stagnation detected ({stagnation_threshold} rounds without improvement). Forcing diversity restart.[/yellow]")
                force_diversity = True

        save_json(self.iterative_refinement_dir, "refinement_results.json", all_results)
        self.stage_times["iterative_refinement"] = time.time() - start_time
        return all_results

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

        usage = self.llm.usage_totals
        token_totals = {
            "input": int(usage["input_tokens"]),
            "output": int(usage["output_tokens"]),
            "cache_creation": int(usage["cache_creation_input_tokens"]),
            "cache_read": int(usage["cache_read_input_tokens"]),
            "total": int(self.llm.total_tokens),
            "total_cost_usd": float(usage["total_cost_usd"]),
        }

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
        report += f"- **Reflection:** {format_time(self.stage_times.get('reflection', 0))}\n"
        report += f"- **Iterative Refinement:** {format_time(self.stage_times.get('iterative_refinement', 0))}\n"
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

        log_event("reflection", "stage_start")
        self.reflection()
        log_event("reflection", "stage_end", seconds=self.stage_times["reflection"])

        log_event("iterative_refinement", "stage_start")
        final_results = self.iterative_refinement(stage3_results)
        log_event("iterative_refinement", "stage_end", seconds=self.stage_times["iterative_refinement"])

        self.stage_times["total_pipeline"] = time.time() - self.pipeline_start_time
        console.print("[bold]Generating final cross-idea comparison report...[/bold]")
        final_report = self.generate_final_report(final_results)
        save_log(self.config.log_dir, "final_pipeline_report.txt", final_report)
        console.print(Panel(final_report, title="[bold]Final Pipeline Comparison[/bold]", border_style="green"))

        log_event("pipeline", "run_completed", total_seconds=self.stage_times["total_pipeline"], total_tokens=self.llm.total_tokens)
        console.print(Rule("Pipeline Complete", style="bold white"))