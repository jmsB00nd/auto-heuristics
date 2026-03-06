import os
import re
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from config import OrchestratorConfig
from prompt_manager import PromptManager
from llm_client import LLMClient
from idea_parser import IdeaParser
from evaluator import CodeEvaluator
from utils.utils import save_log, save_json
import time

console = Console()

class OrchestratorV2:
    """High-level controller tying the components together."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.prompts = PromptManager(config.prompts_dir)
        self.llm = LLMClient(config)
        self.evaluator = CodeEvaluator(config)
        
        # State
        self.literature_insights = ""
        self.top_ideas = []
        
        self.stage_times = {
            "literature_review": 0.0,
            "ideas_generation": 0.0,
            "implementation": 0.0,
            "total_pipeline": 0.0
        }
        
        self.pipeline_start_time = None
        
        # Setup Dirs
        self.stage1_dir = os.path.join(self.config.log_dir, "literature_review")
        self.stage2_dir = os.path.join(self.config.log_dir, "idea_generation")
        self.stage3_dir = os.path.join(self.config.log_dir, "implementation")
        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir]:
            os.makedirs(d, exist_ok=True)

    def literature_review(self):
        start_time = time.time()
        console.print(Rule("STAGE I: Literature Review", style="bold cyan"))
        
        if not self.config.run_stage1_literature_review:
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
        prompt = f"{self.prompts.system_generator}\n{lit_context}\n{self.prompts.idea_prompt}\n{self.prompts.baseline}\n{self.prompts.idea_history}"
        
        response = self.llm.query(prompt, reset_conversation=not self.config.use_conversation_mode)
        save_log(self.stage2_dir, "raw_ideas.txt", response or "")

        kept, eliminated = IdeaParser.parse_ideas(response or "")
        self.top_ideas = kept[:self.config.target_top_ideas]
        
        save_json(self.stage2_dir, "top_ideas.json", self.top_ideas)
        console.print(f"[bold green]✓ Stage II complete. Found {len(self.top_ideas)} ideas.[/bold green]\n")
        self.stage_times["ideas_generation"] = time.time() - start_time

    def implementation(self):
        start_time = time.time()
        console.print(Rule("STAGE III: Implementation", style="bold green"))
        ideas_to_implement = self.top_ideas[:self.config.top_ideas_to_implement]
        all_results = {}

        for idx, idea in enumerate(ideas_to_implement):
            idea_name = idea.get('name', f'Idea_{idx}')
            
            safe_idea_name = re.sub(r'[^\w\-]', '_', idea_name)
            idea_dir = os.path.join(self.stage3_dir, f"idea_{idx}_{safe_idea_name}")
            os.makedirs(idea_dir, exist_ok=True)
            
            console.print(f"[bold]Implementing {idea_name}...[/bold]")
            
            prompt = f"{self.prompts.system_generator}\nTASK: Implement Cost Function.\n{self.prompts.output_format}\nIdea: {idea_name}\nDesc: {idea.get('description')}\n{self.prompts.baseline}"
            save_log(idea_dir, "prompt.txt", prompt)
            
            response = self.llm.query(prompt, reset_conversation=True)
            save_log(idea_dir, "raw_response.txt", response or "")
            
            code = IdeaParser.extract_code(response or "")
            
            if not code:
                console.print(f"[red]Failed to extract code for {idea_name}[/red]")
                save_json(idea_dir, "evaluation_results.json", {"error": "Failed to extract Python code from response."})
                all_results[idea_name] = {"mean_swaps": float('inf'), "mean_depth": 0, "error": "Extraction Failed"}
                continue

            with open(os.path.join(idea_dir, "heuristic.py"), "w") as f:
                f.write(code)

            stats = self.evaluator.evaluate(code)
            save_json(idea_dir, "evaluation_results.json", stats)
            
            if not stats.get('error'):
                console.print(f"[green]✔ SUCCESS: {stats['mean_swaps']:.2f} avg swaps[/green]")
                all_results[idea_name] = stats
                
                os.makedirs("heuristics", exist_ok=True)
                with open(f"heuristics/{safe_idea_name}.py", "w") as f:
                    f.write(code)
            else:
                console.print(f"[red]✘ FAILED: {stats.get('error')}[/red]")
                all_results[idea_name] = stats

        save_json(self.stage3_dir, "final_results.json", all_results)
        self.stage_times["implementation"] = time.time() - start_time
        if self.pipeline_start_time:
            self.stage_times["total_pipeline"] = time.time() - self.pipeline_start_time
        else:
            self.stage_times["total_pipeline"] = sum(self.stage_times.values())

        console.print("[bold]Generating final cross-idea comparison report...[/bold]")
        final_report = self.generate_cross_idea_report(all_results)
        save_log(self.stage3_dir, "final_cross_idea_report.txt", final_report)
        console.print(Panel(final_report, title="[bold]Cross-Idea Comparison[/bold]", border_style="green"))

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
        self.implementation()
        console.print(Rule("Pipeline Complete", style="bold white"))