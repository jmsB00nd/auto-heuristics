import os
import re
import time
import random
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel

from config import OrchestratorConfig
from prompt_manager import PromptManager
from llm_client import LLMClient
from idea_parser import IdeaParser
from evaluator import CodeEvaluator
from memory import MemoryManager
from utils.utils import save_log, save_json

console = Console()

class OrchestratorV2:
    """High-level controller tying the components together."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.prompts = PromptManager(config.prompts_dir, config.problem)
        self.llm = LLMClient(config)
        self.evaluator = CodeEvaluator(config)
        self.memory = MemoryManager(config.history_file)
        
        # State
        self.literature_insights = ""
        self.top_ideas = []
        
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
        self.stage1_dir = os.path.join(self.config.log_dir, "literature_review")
        self.stage2_dir = os.path.join(self.config.log_dir, "idea_generation")
        self.stage3_dir = os.path.join(self.config.log_dir, "implementation")
        self.stage3_5_dir = os.path.join(self.config.log_dir, "reflection")
        self.stage4_dir = os.path.join(self.config.log_dir, "iterative_refinement")
        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir, self.stage3_5_dir, self.stage4_dir]:
            os.makedirs(d, exist_ok=True)

    def literature_review(self):
        start_time = time.time()
        console.print(Rule("STAGE I: Literature Review", style="bold cyan"))
        
        if not getattr(self.config, 'run_stage1_literature_review', False):
            console.print("[yellow]Skipping Literature Review per config.[/yellow]")
            return
            
        response = self.llm.query(self.prompts.lit_review_prompt, reset_conversation=True)
        self.literature_insights = response or ""
        
        if response:
            save_log(self.stage1_dir, "literature_review.txt", response)
            console.print("[bold green]✓ Stage I complete.[/bold green]\n")
            
        self.stage_times["literature_review"] = time.time() - start_time

    def ideas_generation(self):
        start_time = time.time()
        console.print(Rule("STAGE II: Idea Generation", style="bold magenta"))
        
        lit_context = f"\nLiterature Context:\n{self.literature_insights}\n" if self.literature_insights else ""
        
        # Inject Memory into Generation
        top_past_ideas = self.memory.get_top_k(3)
        memory_context = ""
        if top_past_ideas:
            memory_context = "\nPAST SUCCESSFUL IDEAS (For Inspiration):\n"
            for past in top_past_ideas:
                memory_context += f"- {past['name']}: {past['description']} (Score: {past['mean_swaps']})\n"
        
        prompt = f"{self.prompts.system_generator}\n{lit_context}\n{memory_context}\n{self.prompts.idea_prompt}\n{self.prompts.code}"
        response = self.llm.query(prompt, reset_conversation=not getattr(self.config, 'use_conversation_mode', True))
        save_log(self.stage2_dir, "raw_ideas.txt", response or "")

        kept, eliminated = IdeaParser.parse_ideas(response or "")
        self.top_ideas = kept[:getattr(self.config, 'target_top_ideas', 5)]
        
        save_json(self.stage2_dir, "top_ideas.json", self.top_ideas)
        console.print(f"[bold green]✓ Stage II complete. Found {len(self.top_ideas)} ideas.[/bold green]\n")
        self.stage_times["ideas_generation"] = time.time() - start_time

    def implementation(self):
        start_time = time.time()
        console.print(Rule("STAGE III: Implementation", style="bold green"))
        ideas_to_implement = self.top_ideas[:getattr(self.config, 'top_ideas_to_implement', 5)]
        all_results = {}

        for idx, idea in enumerate(ideas_to_implement):
            idea_name = idea.get('name', f'Idea_{idx}')
            idea_desc = idea.get('description', '')
            
            safe_idea_name = re.sub(r'[^\w\-]', '_', idea_name)
            idea_dir = os.path.join(self.stage3_dir, f"idea_{idx}_{safe_idea_name}")
            os.makedirs(idea_dir, exist_ok=True)
            
            console.print(f"[bold]Implementing {idea_name}...[/bold]")
            
            prompt = f"{self.prompts.output_format}\nALGORITHM IDEA TO IMPLEMENT: {idea_name}\nDescription: {idea_desc}\n{getattr(self.prompts, 'code', '')}"
            save_log(idea_dir, "prompt.txt", prompt)
            
            response = self.llm.query(prompt, reset_conversation=True)
            save_log(idea_dir, "raw_response.txt", response or "")
            target_func = "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"
            
            code = IdeaParser.extract_code(response or "", target_func)
            
            if not code:
                console.print(f"[red]Failed to extract code for {idea_name}[/red]")
                save_json(idea_dir, "evaluation_results.json", {"error": "Extraction Failed"})
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed", "code": ""}
                continue

            with open(os.path.join(idea_dir, "heuristic.py"), "w") as f:
                f.write(code)

            stats = self.evaluator.evaluate(code)
            save_json(idea_dir, "evaluation_results.json", stats)
            
            # Save to Memory Manager
            self.memory.add_idea(idea_name, idea_desc, code, stats.get('mean_swaps', float('inf')), stats.get('mean_depth', 0), stats.get('error'))
            
            if not stats.get('error'):
                console.print(f"[green]✔ SUCCESS: {stats['mean_swaps']:.2f} avg swaps[/green]")
                all_results[idea_name] = {"code": code, **stats}
                
                os.makedirs("heuristics", exist_ok=True)
                with open(f"heuristics/{safe_idea_name}.py", "w") as f:
                    f.write(code)
            else:
                console.print(f"[red]✘ FAILED: {stats.get('error')}[/red]")
                all_results[idea_name] = {"code": code, **stats}

        save_json(self.stage3_dir, "final_results.json", all_results)
        self.stage_times["implementation"] = time.time() - start_time
        return all_results

    def reflection(self):
        start_time = time.time()
        console.print(Rule("STAGE III.V: Reflection", style="bold yellow"))
        
        if not getattr(self.config, 'run_stage3_5_reflection', False):
            console.print("[yellow]Skipping Reflection per config.[/yellow]")
            return

        top_idea = self.memory.get_top_k(1)
        worst_idea = self.memory.get_worst_k(1)
        
        if not top_idea:
            console.print("[yellow]No successful ideas to reflect upon.[/yellow]")
            return

        reflection_context = f"BEST PERFORMING HEURISTIC:\nName: {top_idea[0]['name']}\nScore: {top_idea[0]['mean_swaps']}\nCode:\n{top_idea[0]['code']}\n\n"
        
        if worst_idea and worst_idea[0]['name'] != top_idea[0]['name']:
            reflection_context += f"POORLY PERFORMING HEURISTIC (For contrast):\nName: {worst_idea[0]['name']}\nScore: {worst_idea[0]['mean_swaps']}\nCode:\n{worst_idea[0]['code']}\n"

        prompt = f"{self.prompts.reflection_prompt}\n{reflection_context}"
        response = self.llm.query(prompt, reset_conversation=True)
        
        if response:
            self.memory.add_reflection(response)
            save_log(self.stage3_5_dir, "reflection_insights.txt", response)
            console.print("[bold green]✓ Stage III.V complete. Insights added to memory.[/bold green]\n")
            
        self.stage_times["reflection"] = time.time() - start_time

    def iterative_refinement(self, initial_results):
        start_time = time.time()
        console.print(Rule("STAGE IV: Iterative Refinement (Mutation & Crossover)", style="bold blue"))

        if not getattr(self.config, 'run_stage4_iterative_refinement', False):
            console.print("[yellow]Skipping Iterative Refinement per config.[/yellow]")
            return initial_results

        best_idea_name = None
        best_score = float('inf')
        
        for name, stats in initial_results.items():
            if stats.get('error') is None and stats.get('mean_swaps', float('inf')) < best_score:
                best_score = stats['mean_swaps']
                best_idea_name = name

        all_results = initial_results.copy()
        refinement_rounds = getattr(self.config, 'refinement_rounds', 3)

        for round_idx in range(1, refinement_rounds + 1):
            # Fetch fresh from memory every round to ensure we have the absolute best
            top_memory_ideas = self.memory.get_top_k(2)
            if not top_memory_ideas:
                console.print("[red]No successful ideas in memory to refine. Exiting stage.[/red]")
                break
                
            current_best = top_memory_ideas[0]
            best_score = current_best['mean_swaps']
            
            console.print(f"\n[bold]Refinement Round {round_idx}/{refinement_rounds} (Current Best Score: {best_score:.2f})[/bold]")
            round_dir = os.path.join(self.stage4_dir, f"round_{round_idx}")
            os.makedirs(round_dir, exist_ok=True)

            reflection_insight = self.memory.get_latest_reflection()
            reflection_str = f"\nREFLECTION INSIGHTS:\n{reflection_insight}\n" if reflection_insight else ""

            # Decide Operation: Mutation or Crossover
            if len(top_memory_ideas) >= 2 and random.random() < getattr(self.config, 'crossover_rate', 0.5):
                operation = "CROSSOVER"
                parent1, parent2 = top_memory_ideas[0], top_memory_ideas[1]
                context = (
                    f"PARENT 1 ({parent1['name']} - Score {parent1['mean_swaps']}):\n```python\n{parent1['code']}\n```\n\n"
                    f"PARENT 2 ({parent2['name']} - Score {parent2['mean_swaps']}):\n```python\n{parent2['code']}\n```\n"
                )
                prompt_template = getattr(self.prompts, 'crossover_prompt', 'Combine these two algorithms into a better one.')
            else:
                operation = "MUTATION"
                context = f"CURRENT BEST HEURISTIC ({current_best['name']} - Score {current_best['mean_swaps']}):\n```python\n{current_best['code']}\n```\n"
                prompt_template = getattr(self.prompts, 'refinement_prompt', 'Improve the current best heuristic.')

            console.print(f"[magenta]Applying Operation: {operation}[/magenta]")

            prompt = (
                f"TASK: {operation}.\n"
                f"{reflection_str}"
                f"{context}\n"
                f"{prompt_template}\n"
                f"{getattr(self.prompts, 'output_format', '')}\n"
                f"{getattr(self.prompts, 'baseline', '')}"
            )
            
            save_log(round_dir, "prompt.txt", prompt)
            response = self.llm.query(prompt, reset_conversation=True)
            save_log(round_dir, "raw_response.txt", response or "")
            
            target_func = "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"
            code = IdeaParser.extract_code(response or "", target_func)
            idea_name = f"Refined_Idea_{operation}_Round_{round_idx}"

            if not code:
                console.print(f"[red]Failed to extract code in round {round_idx}[/red]")
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed"}
                continue
                
            with open(os.path.join(round_dir, "heuristic.py"), "w") as f:
                f.write(code)

            # Evaluate
            stats = self.evaluator.evaluate(code)
            save_json(round_dir, "evaluation_results.json", stats)
            
            # Save to memory immediately so it can be used in the next round
            self.memory.add_idea(idea_name, f"Generated via {operation}", code, stats.get('mean_swaps', float('inf')), stats.get('mean_depth', 0), stats.get('error'))
            
            if not stats.get('error'):
                current_score = stats['mean_swaps']
                console.print(f"[green]✔ SUCCESS: {current_score:.2f} avg swaps[/green]")
                all_results[idea_name] = stats
                
                if current_score < best_score:
                    console.print(f"[bold green] New Best Score! {best_score:.2f} -> {current_score:.2f}[/bold green]")
                    os.makedirs("heuristics", exist_ok=True)
                    with open(f"heuristics/best_refined_round_{round_idx}.py", "w") as f:
                        f.write(code)
            else:
                console.print(f"[red]✘ FAILED: {stats.get('error')}[/red]")
                all_results[idea_name] = stats

        save_json(self.stage4_dir, "refinement_results.json", all_results)
        self.stage_times["iterative_refinement"] = time.time() - start_time
        return all_results

    def generate_cross_idea_report(self, all_results):
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

        report += "\n## Global Metrics\n"
        report += f"- **Generation Success Rate:** {generation_success_rate:.2f}%\n"
        report += f"- **Average Score (Mean Swaps):** {avg_swaps if avg_swaps != float('inf') else 'N/A':.2f}\n"
        report += f"- **Best Score:** {best_swaps if best_swaps != float('inf') else 'N/A':.2f}\n"
        report += f"- **Worst Score:** {worst_swaps if worst_swaps != float('inf') else 'N/A':.2f}\n"        
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

        return report

    def run_full_pipeline(self):
        self.pipeline_start_time = time.time()
        self.literature_review()
        self.ideas_generation()        
        stage3_results = self.implementation()
        self.reflection() # NEW STAGE
        final_results = self.iterative_refinement(stage3_results)        
        self.stage_times["total_pipeline"] = time.time() - self.pipeline_start_time
        console.print("[bold]Generating final cross-idea comparison report...[/bold]")
        final_report = self.generate_cross_idea_report(final_results)        
        save_log(self.config.log_dir, "final_pipeline_report.txt", final_report)
        console.print(Panel(final_report, title="[bold]Final Pipeline Comparison[/bold]", border_style="green"))

        console.print(Rule("Pipeline Complete", style="bold white"))