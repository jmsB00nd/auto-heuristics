import re
import subprocess
import types as _types_module  
import threading as _threading_module  
import json
import os
from datetime import datetime
import time as _time_module
import traceback as tb_module
from pathlib import Path as _Path  
from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import *
from src.graph.graph import *
from qpu.src.load_backend import *
from utils.utils import save_log, save_json, _timestamp, file_to_string, run_with_token_counter
import numpy as _np  
from tqdm import tqdm as _tqdm  
from rich.console import Console
from rich.panel import Panel as _Panel  
from rich.rule import Rule
import pandas as pd

console = Console()
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = f"logs/{current_time}_run"

class OrchestratorV2:

    def __init__(
        self,
        cli_command,
        backend,
        benchmark_dir,
        history_file="experiment_history.json",
        use_conversation_mode=True,
        send_context_api=False,
        show_token_counter=True,
        max_idea_gen_rounds=3,
        target_top_ideas=10,
        min_score_threshold=7,
        run_stage1_literature_review=True,
        timeout_seconds=300,
        top_ideas_to_implement=3
    ):
        self.CLI_COMMAND = cli_command
        self.BACKEND = backend
        self.BENCHMARK_DIR = benchmark_dir
        self.HISTORY_FILE = history_file
        self.use_conversation_mode = use_conversation_mode
        self.send_context_api = send_context_api
        self.show_token_counter = show_token_counter
        self.timeout_seconds = timeout_seconds
        # Stage control
        self.run_stage1_literature_review = run_stage1_literature_review

        # Stage II parameters
        self.max_idea_gen_rounds = max_idea_gen_rounds
        self.target_top_ideas = target_top_ideas
        self.min_score_threshold = min_score_threshold
        
        # Stage  III IV parameters
        self.top_ideas_to_implement = top_ideas_to_implement

        # State
        self.literature_insights = None
        self.all_generated_ideas = []   # every idea ever generated (kept + eliminated)
        self.top_ideas = []             # final top ideas after filtering
        self.implementation_plan = None
        self.conversation_history = []

        # Stage log dirs
        self.stage1_dir = os.path.join(LOG_DIR, "literature_review")
        self.stage2_dir = os.path.join(LOG_DIR, "idea_generation")
        self.stage3_dir = os.path.join(LOG_DIR, "implementation")
        
        self.total_tokens = 0 
        self.sota_baselines = self.load_sota_baselines()
        self.init_prompt()

    def init_prompt(self) :
        self.baseline = file_to_string('/home/jmsb00nd/Documents/auto-heuristics/prompts/routing/baseline.txt')
        self.system_generator_prompt = file_to_string('/home/jmsb00nd/Documents/auto-heuristics/prompts/common/system_generator.txt')
        self.documentation = file_to_string('/home/jmsb00nd/Documents/auto-heuristics/prompts/common/documentation.txt')
        self.output_format = file_to_string('/home/jmsb00nd/Documents/auto-heuristics/prompts/routing/output_format.txt')
        self.idea_prompt = file_to_string("/home/jmsb00nd/Documents/auto-heuristics/prompts/routing/ideas_generation.txt")
    
    def query_llm(self, prompt_text, reset_conversation=False):
        try:
            if reset_conversation:
                self.conversation_history = []

            self.conversation_history.append({"role": "user", "content": prompt_text})

            if self.use_conversation_mode and len(self.conversation_history) > 1:
                formatted_input = self.format_conversation_for_cli()
            else:
                formatted_input = prompt_text

            if self.show_token_counter:
                response, token_count = run_with_token_counter(self.CLI_COMMAND, formatted_input)
                self.total_tokens += token_count
            else:
                with console.status("[bold green]Querying LLM...", spinner="dots"):
                    result = subprocess.run(
                        self.CLI_COMMAND,
                        input=formatted_input,
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding='utf-8',
                        shell=True,
                    )
                    response = result.stdout.strip()
                    self.total_tokens += len(response.split())

            if self.use_conversation_mode:
                self.conversation_history.append({"role": "assistant", "content": response})

            return response
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error:[/bold red] CLI returned exit code {e.returncode}")
            return None
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] Execution failed: {e}")
            return None


    def load_sota_baselines(self):
        """Loads and calculates SOTA benchmarks for final comparison."""
        results = {}
        try:
            df = pd.read_csv("/home/jmsb00nd/Documents/auto-heuristics/benchmarks/train_set_with_baselines.csv")
            numeric_cols = df.columns.drop(["original_folder", "original_filename", "new_filename"], errors="ignore")
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            methods = ["sabre", "qmap", "pytket", "cirq"]
            for method in methods:
                swap_col = f"{method}_sherbrooke_swaps_trivial"
                depth_col = f"{method}_sherbrooke_depth_trivial"
                
                if swap_col in df.columns and depth_col in df.columns:
                    results[method] = {
                        "mean_swaps": df[swap_col].mean(),
                        "mean_depth": df[depth_col].mean()
                    }
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not load SOTA baselines. Check CSV path. Error: {e}[/bold yellow]")
        return results

    def format_conversation_for_cli(self):
        formatted = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n\n"
            else:
                formatted += f"Assistant: {msg['content']}\n\n"
        return formatted.strip()


    def inject_and_run(self, heuristic_code_str, timeout_seconds=300):
        local_scope = {}

        exec_globals = dict(globals())
        try:
            exec(heuristic_code_str, exec_globals, local_scope)
        except Exception as e:
            return {"mean_swaps": float('inf'), "error": f"Syntax Error: {str(e)}"}

        new_heuristic_func = local_scope.get('qlosure_poly_heuristic')

        # Fallback: LLM may have wrapped the function inside a class
        if new_heuristic_func is None:
            for obj in local_scope.values():
                if isinstance(obj, type):
                    method = getattr(obj, 'qlosure_poly_heuristic', None)
                    if method is not None:
                        # Extract the underlying function from the class
                        new_heuristic_func = method
                        console.print("[yellow]Note: Extracted qlosure_poly_heuristic from class wrapper.[/yellow]")
                        break

        if new_heuristic_func is None:
            return {"mean_swaps": float('inf'), "error": "Function not found."}
        edges = load_backend_edges(self.BACKEND)
        circuit_files = list(_Path(self.BENCHMARK_DIR).glob("*.json"))

        if not circuit_files:
            return {"error": f"No .json files found in {self.BENCHMARK_DIR}"}

        results = {"swaps": [], "depth": [], "runtimes": [], "failures": []}

        console.print(_Panel(
            f"[bold blue]Batch Execution[/bold blue]\n"
            f"Processing [bold]{len(circuit_files)}[/bold] circuits\n"
            f"Timeout: {timeout_seconds}s per circuit",
            border_style="blue",
        ))

        for circuit_path in _tqdm(circuit_files, desc="Routing progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            try:
                data = json_file_to_isl(str(circuit_path))
                router = Qlosure(edges, data)
                router.qlosure_poly_heuristic = _types_module.MethodType(new_heuristic_func, router)

                result_container = {}
                exception_container = {}

                def run_with_timeout():
                    try:
                        min_swaps, min_depth, _ = router.run(heuristic_method="Qlosure")
                        result_container['swaps'] = min_swaps
                        result_container['depth'] = min_depth
                    except Exception as e:
                        exception_container['error'] = e

                start_time = _time_module.time()
                thread = _threading_module.Thread(target=run_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout_seconds)
                end_time = _time_module.time()

                if thread.is_alive():
                    console.print(f"[yellow]![/yellow] Timeout on {circuit_path.name}")
                    return {"mean_swaps": float('inf'), "error": f"Timeout on {circuit_path.name}"}

                if 'error' in exception_container:
                    raise exception_container['error']

                results["swaps"].append(result_container['swaps'])
                results["depth"].append(result_container['depth'])
                results["runtimes"].append(end_time - start_time)

            except Exception as e:
                error_tb = tb_module.format_exc()
                results["failures"].append({"circuit": circuit_path.name, "error": str(e), "traceback": error_tb})
                # Early-stop: if 3+ failures with the same error, stop wasting time
                if len(results["failures"]) >= 3:
                    error_msgs = [f["error"] for f in results["failures"]]
                    if len(set(error_msgs)) == 1:
                        console.print(f"[bold red]Early stop: same error on {len(results['failures'])} circuits: {error_msgs[0]}[/bold red]")
                        break

        first_failure_tb = results["failures"][0]["traceback"] if results["failures"] else None
        first_failure_err = results["failures"][0]["error"] if results["failures"] else None
        return {
            "mean_swaps": _np.mean(results["swaps"]) if results["swaps"] else float('inf'),
            "mean_depth": _np.mean(results["depth"]) if results["depth"] else 0,
            "mean_runtime": _np.mean(results["runtimes"]) if results["runtimes"] else 0,
            "total_circuits": len(circuit_files),
            "successful_runs": len(results["swaps"]),
            "failed_runs": len(results["failures"]),
            "error": None if results["swaps"] else "All circuits failed",
            "first_failure_error": first_failure_err,
            "first_failure_traceback": first_failure_tb,
        }

    def parse_response(self, response_text):
        code = None
        # Try multiple patterns for code extraction (LLMs vary formatting)
        patterns = [
            r"```python\s*\n(.*?)```",       # ```python\n...```
            r"```python(.*?)```",             # ```python...``` (no newline)
            r"```py\s*\n(.*?)```",            # ```py\n...```
            r"```\s*\n(def qlosure_poly_heuristic.*?)```",  # ``` with no language tag
            r"(def qlosure_poly_heuristic\(self,\s*swap_gate\):.*?)(?:```|\Z)",  # raw function
        ]
        for pattern in patterns:
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                candidate = code_match.group(1).strip()
                if 'qlosure_poly_heuristic' in candidate or 'def ' in candidate:
                    code = candidate
                    break

        if code:
            code = self._sanitize_heuristic_code(code)
        return code

    @staticmethod
    def _sanitize_heuristic_code(code: str) -> str:
        """Extract the standalone qlosure_poly_heuristic function from LLM output.

        The LLM sometimes wraps the function inside a placeholder class or
        adds import statements / helper classes. This method strips everything
        down to just the function definition so that ``exec()`` places
        ``qlosure_poly_heuristic`` directly into ``local_scope``.
        """
        lines = code.split('\n')
        func_start = None
        func_indent = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('def qlosure_poly_heuristic'):
                func_start = i
                func_indent = len(line) - len(stripped)
                break

        if func_start is None:
            return code

        func_lines = [lines[func_start]]
        for j in range(func_start + 1, len(lines)):
            line = lines[j]
            if line.strip() == '':
                func_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= func_indent and line.strip() != '':
                break
            func_lines.append(line)

        dedented = []
        for line in func_lines:
            if line.strip() == '':
                dedented.append('')
            elif len(line) >= func_indent:
                dedented.append(line[func_indent:])
            else:
                dedented.append(line.lstrip())

        import_lines = [l for l in lines[:func_start] if l.strip().startswith(('import ', 'from '))]

        result_lines = import_lines + [''] + dedented if import_lines else dedented
        return '\n'.join(result_lines)

    def literature_review(self):
        console.print(Rule("STAGE I: Literature Review", style="bold cyan"))
        console.print(_Panel(
            "[bold cyan]Conducting systematic survey of quantum routing literature...[/bold cyan]\n"
            "This may take a few minutes due to the depth of analysis required.",
            border_style="cyan",
        ))

        
        prompt = file_to_string("/home/jmsb00nd/Documents/auto-heuristics/prompts/routing/literature_review.txt")
        response = self.query_llm(prompt, reset_conversation=True)

        if not response:
            console.print("[bold yellow]Literature review failed. Proceeding without it.[/bold yellow]")
            self.literature_insights = ""
            return None

        self.literature_insights = response

        section_keys = [
            "LITERATURE_OVERVIEW", "MAJOR_APPROACHES", "COMPARATIVE_ANALYSIS",
            "MATHEMATICAL_INSIGHTS", "OPEN_PROBLEMS", "RESEARCH_GAPS",
            "NOVEL_DIRECTIONS", "KEY_RECOMMENDATIONS",
        ]
        section_styles = ["cyan", "blue", "magenta", "green", "yellow", "red", "bright_blue", "bright_green"]
        for key, style in zip(section_keys, section_styles):
            pattern = rf"{key}:\s*(.*?)(?={'|'.join(k for k in section_keys if k != key)}:|\Z)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                display = content if len(content) < 2000 else content[:2000] + "\n\n[dim]... (truncated)[/dim]"
                console.print(_Panel(display, title=f"[bold]{key}[/bold]", border_style=style))

        save_log(self.stage1_dir, "literature_review.txt", response)
        save_log(self.stage1_dir, "prompt.txt", prompt)
        _Path("literature_review.txt").write_text(response, encoding="utf-8")

        console.print("[bold green]✓ Stage I complete.[/bold green]\n")
        return response

    def ideas_generation(self):
        console.print(Rule("STAGE II: Idea Generation", style="bold magenta"))
        console.print(_Panel(
            "[bold magenta]Generating, evaluating, scoring, and filtering ideas in a single prompt...[/bold magenta]",
            border_style="magenta",
        ))

        prompt = self.build_ideas_generation_prompt()
        response = self.query_llm(prompt, reset_conversation=not self.use_conversation_mode)
        save_log(self.stage2_dir, "single_pass_ideas.txt", response or "")

        # Parse ideas from the combined response
        console.print("[bold]Parsing structured ideas...[/bold]")
        kept_ideas, eliminated_ideas = self.parse_ideas_from_response(response or "")

        # If parsing from structured format fails, try fallback
        if not kept_ideas:
            console.print("[yellow]No kept ideas parsed from structured format. Trying fallback...[/yellow]")
            kept_ideas = self.parse_ideas_from_generation_response(response or "")
            if kept_ideas:
                console.print(f"[green]Recovered {len(kept_ideas)} ideas from fallback parser.[/green]")

        # Tag ideas
        for idea in kept_ideas:
            idea["round"] = 1
            idea["status"] = "kept"
        for idea in eliminated_ideas:
            idea["round"] = 1
            idea["status"] = "eliminated"

        all_ideas = kept_ideas + eliminated_ideas

        # Save all artifacts
        self.all_generated_ideas = all_ideas
        self.top_ideas = kept_ideas[:self.target_top_ideas]
        save_json(self.stage2_dir, "all_ideas_complete.json", all_ideas)
        save_json(self.stage2_dir, "top_ideas_final.json", self.top_ideas)
        save_json(self.stage2_dir, "eliminated_ideas_complete.json", eliminated_ideas)

        console.print(f"[green]{len(kept_ideas)} kept, {len(eliminated_ideas)} eliminated[/green]")
        console.print(f"[bold green]✓ Stage II complete. {len(self.top_ideas)} top ideas identified.[/bold green]\n")

        return self.top_ideas

    def build_ideas_generation_prompt(self):
        literature_ref = self.literature_ref_section()
        return f"""{self.system_generator_prompt}

{literature_ref}


{self.idea_prompt}

{self.baseline}
"""

    def literature_ref_section(self):
        if self.literature_insights and self.use_conversation_mode:
            return (
                "\nBased on the comprehensive literature review we discussed earlier, "
                "ensure your proposed ideas address identified research gaps and avoid "
                "reinventing approaches already covered.\n"
            )
        elif self.literature_insights:
            return (
                "\nLiterature Review Context\n"
                f"{self.literature_insights}\n\n"
                "Your ideas MUST build on gaps identified in this review.\n"
            )
        return ""

    def parse_ideas_from_response(self, response_text, generation_response=None):
        """Parse structured idea blocks from LLM response.

        Uses multiple parsing strategies with fallbacks:
        1. IDEA: ... END_IDEA blocks
        2. IDEA_NAME: ... --- separated blocks
        3. Numbered/bulleted list items with IDEA_NAME or NAME fields
        4. Fallback: extract ideas from the original generation response
        """
        kept = []
        eliminated = []

        # --- Strategy 1: IDEA: ... END_IDEA blocks ---
        idea_blocks = re.findall(
            r'IDEA:\s*(.*?)END_IDEA',
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if idea_blocks:
            for block in idea_blocks:
                idea = self._parse_single_idea_block(block)
                if "KEPT" in idea.get('status', 'KEPT'):
                    kept.append(idea)
                else:
                    eliminated.append(idea)

        # --- Strategy 2: --- separated blocks with IDEA_NAME or NAME ---
        if not kept and not eliminated:
            blocks = re.split(r'\n---+\n', response_text)
            for block in blocks:
                if not block.strip():
                    continue
                # Check if block has named idea fields
                name_m = re.search(r'(?:IDEA_NAME|NAME):\s*(.+)', block, re.IGNORECASE)
                if name_m:
                    idea = self._parse_single_idea_block(block)
                    if "KEPT" in idea.get('status', 'KEPT'):
                        kept.append(idea)
                    else:
                        eliminated.append(idea)

        # --- Strategy 3: Numbered list items (e.g., "1. **Name** ...") ---
        if not kept and not eliminated:
            # Match patterns like: "1. **Idea Name** (Score: 7.3)" or "1. IDEA_NAME: ..."
            numbered_items = re.findall(
                r'(?:^|\n)\s*\d+\.\s*(?:\*\*)?([^*\n]+?)(?:\*\*)?\s*(?:\((?:Score|Average)[:\s]*(\d+(?:\.\d+)?)\))?\s*(?:\n|$)(.*?)(?=(?:\n\s*\d+\.|\Z))',
                response_text,
                re.DOTALL | re.IGNORECASE,
            )
            for name, score, body in numbered_items:
                name = name.strip().rstrip(':')
                if len(name) < 3 or len(name) > 200:
                    continue
                idea = self._parse_single_idea_block(f"NAME: {name}\n{body}")
                idea['name'] = name
                if score:
                    idea['average_score'] = float(score)
                # Determine status from body text
                status_m = re.search(r'STATUS:\s*(KEPT|ELIMINATED)', body, re.IGNORECASE)
                if status_m and 'ELIM' in status_m.group(1).upper():
                    idea['status'] = 'ELIMINATED'
                    eliminated.append(idea)
                else:
                    idea['status'] = 'KEPT'
                    kept.append(idea)

        # --- Strategy 4: Extract from original generation response ---
        if not kept and not eliminated and generation_response:
            console.print("[yellow]Fallback: Extracting ideas from generation response...[/yellow]")
            kept = self.parse_ideas_from_generation_response(generation_response)

        # --- Strategy 5: Last resort - try to extract ANY idea names from text ---
        if not kept and not eliminated:
            console.print("[yellow]Last resort: Extracting idea names from text...[/yellow]")
            # Look for any IDEA_NAME: lines
            name_matches = re.findall(r'IDEA_NAME:\s*(.+)', response_text, re.IGNORECASE)
            if not name_matches:
                # Try NAME: lines
                name_matches = re.findall(r'(?:^|\n)\s*NAME:\s*(.+)', response_text, re.IGNORECASE)
            for name in name_matches:
                name = name.strip().strip('*').strip()
                if len(name) < 3 or len(name) > 200:
                    continue
                # Try to find description near this name in the text
                desc_pattern = rf'{re.escape(name)}.*?(?:DESCRIPTION|Description):\s*(.*?)(?=\n\s*[A-Z_]+:|---|-\s*\*|\n\n|\Z)'
                desc_m = re.search(desc_pattern, response_text, re.DOTALL | re.IGNORECASE)
                desc = desc_m.group(1).strip() if desc_m else ""
                kept.append({
                    'name': name,
                    'description': desc,
                    'status': 'KEPT',
                    'novelty_score': 5,
                    'probability_score': 5,
                    'effort_score': 5,
                    'average_score': 5,
                    'elimination_reason': 'N/A',
                })

        # Final safety net: if absolutely nothing was parsed, save raw response for manual review
        if not kept and not eliminated:
            console.print("[bold red]Warning: Could not parse ANY structured ideas from response.[/bold red]")
            console.print("[dim]Response preview (first 500 chars):[/dim]")
            console.print(f"[dim]{response_text[:500]}[/dim]")
            # Create emergency fallback entry to prevent complete failure
            # This allows the pipeline to continue even if parsing fails
            kept.append({
                'name': 'PARSING_FAILED_manual_review_needed',
                'description': 'Failed to parse ideas from LLM response. Raw response saved in stage logs. Manual review required.',
                'status': 'KEPT',
                'novelty_score': 1,
                'probability_score': 1,
                'effort_score': 1,
                'average_score': 1,
                'elimination_reason': 'N/A',
                'raw_response_preview': response_text[:500],
            })
            console.print("[yellow]Created emergency fallback entry. Check stage logs for raw LLM response.[/yellow]")

        # Sort kept by average score descending
        kept.sort(key=lambda x: x.get('average_score', 0), reverse=True)

        console.print(f"[dim]  Parsed {len(kept)} kept + {len(eliminated)} eliminated ideas[/dim]")
        return kept, eliminated

    def _parse_single_idea_block(self, block):
        """Parse a single idea block text into a dict, tolerating format variations."""
        idea = {}
        # NAME (try multiple patterns)
        name_m = (re.search(r'(?:IDEA_)?NAME:\s*(.+)', block, re.IGNORECASE) or
                  re.search(r'^\s*\*\*(.+?)\*\*', block, re.MULTILINE))
        status_m = re.search(r'STATUS:\s*(\S+)', block, re.IGNORECASE)
        # DESCRIPTION - be lenient on what follows
        desc_m = re.search(r'DESCRIPTION:\s*(.*?)(?=\n\s*(?:[A-Z_]{3,}\s*:|---)|\Z)', block, re.IGNORECASE | re.DOTALL)
        # Scores - match score patterns flexibly
        nov_m = re.search(r'NOVELTY[_\s]*SCORE:\s*(\d+(?:\.\d+)?)', block, re.IGNORECASE)
        prob_m = re.search(r'PROBABILITY[_\s]*SCORE:\s*(\d+(?:\.\d+)?)', block, re.IGNORECASE)
        eff_m = re.search(r'EFFORT[_\s]*SCORE:\s*(\d+(?:\.\d+)?)', block, re.IGNORECASE)
        avg_m = re.search(r'AVERAGE[_\s]*SCORE:\s*(\d+(?:\.\d+)?)', block, re.IGNORECASE)
        elim_m = re.search(r'ELIMINATION[_\s]*REASON:\s*(.*?)(?=\n\s*(?:[A-Z_]{3,}\s*:|---)|\Z)', block, re.IGNORECASE | re.DOTALL)

        idea['name'] = name_m.group(1).strip().strip('*') if name_m else "Unknown"
        idea['status'] = status_m.group(1).strip().upper() if status_m else "KEPT"
        idea['description'] = desc_m.group(1).strip() if desc_m else ""
        idea['novelty_score'] = float(nov_m.group(1)) if nov_m else 5
        idea['probability_score'] = float(prob_m.group(1)) if prob_m else 5
        idea['effort_score'] = float(eff_m.group(1)) if eff_m else 5
        # Compute average if not provided
        if avg_m:
            idea['average_score'] = float(avg_m.group(1))
        else:
            idea['average_score'] = round((idea['novelty_score'] + idea['probability_score'] + idea['effort_score']) / 3, 2)
        idea['elimination_reason'] = elim_m.group(1).strip() if elim_m else "N/A"

        # Normalize status
        if any(kw in idea['status'] for kw in ['ELIM', 'REJECT', 'REMOVE', 'FAIL']):
            idea['status'] = 'ELIMINATED'
        else:
            idea['status'] = 'KEPT'

        return idea

    def parse_ideas_from_generation_response(self, gen_response):
        """Fallback parser: extract ideas directly from the generation response (Step 1).

        The generation response uses IDEA_NAME: / DESCRIPTION: / --- format.
        This is more reliable than the structured extraction since it's the original.
        Also handles markdown headers and numbered lists as separators.
        """
        ideas = []

        # Strategy A: Split by --- separators (most common)
        blocks = re.split(r'\n---+\n', gen_response)

        # Strategy B: If no --- separators found, try splitting by CATEGORY: or IDEA_NAME: headers
        if len(blocks) <= 1:
            blocks = re.split(r'\n(?=(?:CATEGORY:|IDEA_NAME:|#{1,3}\s+(?:Idea|IDEA)))', gen_response)

        # Strategy C: If still one block, try splitting by numbered items
        if len(blocks) <= 1:
            blocks = re.split(r'\n(?=\d+\.\s+(?:\*\*|IDEA))', gen_response)

        for block in blocks:
            if not block.strip():
                continue
            # Try multiple name patterns in order of specificity
            name_m = re.search(r'IDEA_NAME:\s*(.+)', block, re.IGNORECASE)
            if not name_m:
                name_m = re.search(r'(?:^|\n)\s*(?:#+\s*)?(?:IDEA\s*\d*\s*:?\s*)(.+)', block, re.IGNORECASE)
            if not name_m:
                name_m = re.search(r'(?:^|\n)\s*(?:#+\s*)?(?:\d+\.\s*)?\*\*(.+?)\*\*', block)
            if not name_m:
                name_m = re.search(r'(?:^|\n)\s*NAME:\s*(.+)', block, re.IGNORECASE)
            if not name_m:
                continue
            name = name_m.group(1).strip().strip('*').strip(':').strip()
            if len(name) < 3 or len(name) > 200:
                continue
            # Skip if name looks like a section header rather than an idea
            if any(kw in name.upper() for kw in ['CATEGORY', 'SECTION', 'TASK', 'OUTPUT FORMAT', 'REQUIREMENTS']):
                continue
            desc_m = re.search(r'DESCRIPTION:\s*(.*?)(?=\n\s*(?:[A-Z_]{3,}:|---|\Z))', block, re.DOTALL | re.IGNORECASE)
            desc = desc_m.group(1).strip() if desc_m else ""
            # Deduplicate: skip if we already have this idea name
            if any(i['name'].lower() == name.lower() for i in ideas):
                continue
            ideas.append({
                'name': name,
                'description': desc,
                'status': 'KEPT',
                'novelty_score': 5,
                'probability_score': 5,
                'effort_score': 5,
                'average_score': 5,
                'elimination_reason': 'N/A',
            })

        console.print(f"[dim]  Fallback parser extracted {len(ideas)} ideas from generation response[/dim]")
        return ideas

    def implementation(self):
        console.print(Rule("STAGE III: Implementation", style="bold green"))

        if not self.top_ideas:
            console.print("[bold red]No top ideas available. Run Stage II first.[/bold red]")
            return None

        # Limit to top_ideas_to_implement ideas
        ideas_to_implement = self.top_ideas[:self.top_ideas_to_implement]
        console.print(f"[bold cyan]Implementing top {len(ideas_to_implement)} of {len(self.top_ideas)} ideas[/bold cyan]")

        all_ideas_results = {}

        for idea_idx, idea in enumerate(ideas_to_implement):
            idea_name = idea.get('name', f'Idea_{idea_idx}')
            idea_name_safe = re.sub(r'[^\w\-]', '_', idea_name)
            idea_stage_dir = os.path.join(self.stage3_dir, f"idea_{idea_idx}_{idea_name_safe}")

            console.print(Rule(
                f"Idea {idea_idx + 1}/{len(ideas_to_implement)}: {idea_name}",
                style="bold green",
            ))

            implementation_results = self.implement_single_idea(idea, idea_idx, idea_stage_dir)
            all_ideas_results[idea_name] = implementation_results

        # Final cross-idea comparison report
        console.print("[bold]Generating final cross-idea comparison report...[/bold]")
        final_report = self.generate_cross_idea_report(all_ideas_results)
        save_log(self.stage3_dir, "final_cross_idea_report.txt", final_report)
        save_json(self.stage3_dir, "all_ideas_results.json", all_ideas_results)

        console.print(_Panel(final_report[:4000], title="[bold]Cross-Idea Comparison[/bold]", border_style="green"))
        console.print(f"[bold green]✓ Stage IV complete. {len(all_ideas_results)} ideas tested.[/bold green]\n")

        return all_ideas_results

    def implement_single_idea(self, idea, idea_idx, idea_stage_dir):
        """Implement a single idea with one prompt, plus fix retries if needed."""
        idea_name = idea.get('name', f'Idea_{idea_idx}')
        idea_name_safe = re.sub(r'[^\w\-]', '_', idea_name)

        MAX_FIX_RETRIES = 0

        console.print(f"[bold]Implementing {idea_name} ...[/bold]")
        prompt = self.build_single_implementation_prompt(idea)
        response = self.query_llm(prompt, reset_conversation=True)
        save_log(idea_stage_dir, "implementation.txt", response or "")

        # Extract code
        code = self.parse_response(response)
        if not code:
            console.print(f"[bold red]Failed to extract code for {idea_name}.[/bold red]")
            result = {
                "iteration": 1, "idea_name": idea_name, "code": "", "stats": {},
                "status": "FAILED", "error": "Code extraction failed",
                "mean_swaps": float('inf'), "mean_depth": 0, "timestamp": _timestamp(),
            }
            save_json(idea_stage_dir, "implementation_results.json", [result])
            return [result]

        # Run code
        console.print(f"[bold]Running and testing {idea_name}...[/bold]")
        stats = self.inject_and_run(code, timeout_seconds=self.timeout_seconds)

        # Fix retry loop (error recovery AND timeout optimization)
        for retry in range(1, MAX_FIX_RETRIES + 1):
            if not stats.get('error'):
                break

            err_msg = stats.get('first_failure_error', stats.get('error', ''))
            is_timeout = "Timeout on" in str(err_msg)

            if is_timeout:
                console.print(f"[bold yellow]Timeout detected - attempting optimization {retry}/{MAX_FIX_RETRIES} for {idea_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]Fix attempt {retry}/{MAX_FIX_RETRIES} for {idea_name}...[/bold yellow]")
                if stats.get('first_failure_traceback'):
                    tb_lines = stats['first_failure_traceback'].strip().split('\n')
                    tb_preview = '\n'.join(tb_lines[-5:])
                    console.print(f"[dim]{tb_preview}[/dim]")

            # Use timeout-specific or regular fix prompt depending on error type
            if is_timeout:
                fix_prompt = self.build_timeout_optimization_prompt(idea, code, retry)
            else:
                fix_prompt = self.build_fix_prompt(idea, 1, code, stats)
            
            fix_response = self.query_llm(fix_prompt)
            
            # Save with appropriate label
            if is_timeout:
                save_log(idea_stage_dir, f"timeout_optimization_{retry}.txt", fix_response or "")
            else:
                save_log(idea_stage_dir, f"fix_attempt_{retry}.txt", fix_response or "")

            new_code = self.parse_response(fix_response or "")
            if new_code:
                code = new_code
                action = "optimized" if is_timeout else "fixed"
                console.print(f"[bold]Re-running after {action} code (attempt {retry})...[/bold]")
                stats = self.inject_and_run(code, timeout_seconds=self.timeout_seconds)
            else:
                action = "timeout optimization" if is_timeout else "fix"
                console.print(f"[bold red]Failed to extract code from {action} attempt {retry}.[/bold red]")
                stats = {"error": f"Code extraction failed on {action} attempt"}

        succeeded = not stats.get('error')
        if succeeded:
            console.print(
                f"[bold green]✔ {idea_name} SUCCESS[/bold green]\n"
                f"  Avg Swaps: {stats['mean_swaps']:.2f} | Avg Depth: {stats['mean_depth']:.2f}"
            )
        else:
            error_msg = stats.get('error', 'Unknown')
            console.print(f"[bold red]✘ {idea_name} FAILED: {error_msg}[/bold red]")
            if "Timeout" in str(error_msg):
                console.print(f"[dim]  Note: Function too slow even after {MAX_FIX_RETRIES} optimization attempts[/dim]")

        # Save code to heuristics/
        if code:
            os.makedirs("heuristics", exist_ok=True)
            filename = f"heuristics/idea_{idea_idx}_iter1_{idea_name_safe}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(
                    f"# Idea: {idea_name}\n"
                    f"# Stats: {json.dumps(stats, default=str)}\n\n"
                    f"{code}"
                )
            console.print(f"[dim]  -> Saved {filename}[/dim]")

        result = {
            "iteration": 1,
            "idea_name": idea_name,
            "code": code or "",
            "stats": stats or {},
            "status": "SUCCESS" if succeeded else "FAILED",
            "error": stats.get('error') if stats else "Code extraction failed",
            "mean_swaps": stats.get('mean_swaps', float('inf')) if stats else float('inf'),
            "mean_depth": stats.get('mean_depth', 0) if stats else 0,
            "timestamp": _timestamp(),
        }

        # Per-idea final report
        console.print(f"[bold]Generating report for {idea_name}...[/bold]")
        idea_report = self.generate_stage3_final_report(idea, [result])
        save_log(idea_stage_dir, "final_report.txt", idea_report)
        save_json(idea_stage_dir, "implementation_results.json", [result])

        console.print(_Panel(
            idea_report[:2000],
            title=f"[bold]{idea_name} Report[/bold]",
            border_style="green",
        ))

        return [result]

    def build_single_implementation_prompt(self, idea):        
        return f"""{self.system_generator_prompt}

You are at Stage 3: Idea Implementation, You will be provided with: An idea and A description of that idea, Your task is to generate code that implements the given idea according to its description.

TASK: Implement Complete Cost Function.

{self.output_format}

Idea: {idea.get('name', 'Unknown')}
Description: {idea.get('description', '')}

{self.documentation}

{self.baseline}
"""

    def build_fix_prompt(self, idea, iter_num, failed_code, stats):
        """Build a prompt that feeds the actual runtime error/traceback back to the LLM for fixing."""

        if stats and stats.get('first_failure_traceback'):
            error_section = f"""## Error Message: {stats.get('first_failure_error', stats.get('error', 'Unknown'))}

## Full Traceback:
```
{stats['first_failure_traceback']}
```"""
        elif stats and stats.get('error'):
            error_section = f"""## Error:
{stats['error']}"""
        else:
            error_section = "## Error: Unknown runtime failure"

        return f"""{self.system_generator_prompt}

TASK: Fix Runtime Error in Iteration {iter_num}

Idea: {idea.get('name', 'Unknown')}

Descreption : {idea.get('description', '')}
## The following code CRASHED when running on quantum circuits:

```python
{failed_code or 'No code available — extraction failed, see error below'}
```

{error_section}

## Instructions:
1. Analyze the traceback carefully. Identify the ROOT CAUSE.
2. Fix the bug
3. Add defensive guards: check for -1 (unmapped), check dict membership, avoid division by zero, clamp values before log/sqrt.
4. Return a COMPLETE, working `def qlosure_poly_heuristic(self, swap_gate):` function.
5. DO NOT use try/except blocks. Handle errors explicitly with guards and conditional checks.

{self.output_format}
"""

    def build_timeout_optimization_prompt(self, idea, slow_code, attempt_num):
        """Build a prompt asking LLM to optimize/simplify the function that timed out."""

        return f"""{self.system_generator_prompt}

                TASK: Optimize Function That Timed Out (Attempt {attempt_num}/3)

                Idea: {idea.get('name', 'Unknown')}
                Description: {idea.get('description', '')}

                PROBLEM: The following implementation TIMED OUT (didn't converge)

                ```python
                {slow_code}
                ```
                {self.output_format}
"""

    def generate_stage3_final_report(self, idea, results):
        report = f"""
# Implementation Report
# Generated: {_timestamp()}

## Idea: {idea.get('name', 'Unknown')}
## Description: {idea.get('description', '')}

## Iteration Summary
"""
        for r in results:
            report += f"""
### Iteration {r['iteration']}
- Status: {r['status']}
- Mean Swaps: {r.get('mean_swaps', 'N/A')}
- Mean Depth: {r.get('mean_depth', 'N/A')}
- Error: {r.get('error', 'None')}
"""

        # Find best iteration
        successful = [r for r in results if r['status'] == 'SUCCESS']
        if successful:
            best = min(successful, key=lambda x: x.get('mean_swaps', float('inf')))
            report += f"""
## Best Result
- Iteration: {best['iteration']}
- Mean Swaps: {best.get('mean_swaps', 'N/A')}
- Mean Depth: {best.get('mean_depth', 'N/A')}
"""
            report += f"\n## Best Code:\n```python\n{best.get('code', 'N/A')}\n```\n"
        else:
            report += "\n## No successful iterations.\n"

        report += f"""
## Overall Statistics
- Total iterations: {len(results)}
- Successful: {len(successful)}
- Failed: {len(results) - len(successful)}
"""
        return report

    def generate_cross_idea_report(self, all_ideas_results):
        """Generate a comparison report across all tested ideas including detailed SOTA metrics."""
        report = f"""
Cross-Idea Comparison Report
Generated: {_timestamp()}
Ideas Tested: {len(all_ideas_results)}
"""
        # Collect best result per idea
        idea_bests = []
        for idea_name, results in all_ideas_results.items():
            successful = [r for r in results if r['status'] == 'SUCCESS']
            if successful:
                best = min(successful, key=lambda x: x.get('mean_swaps', float('inf')))
                idea_bests.append({
                    'name': idea_name,
                    'mean_swaps': best.get('mean_swaps', float('inf')),
                    'mean_depth': best.get('mean_depth', 0),
                    'iteration': best['iteration'],
                    'total_iterations': len(results),
                    'successful_iterations': len(successful),
                })
            else:
                idea_bests.append({
                    'name': idea_name,
                    'mean_swaps': float('inf'),
                    'mean_depth': 0,
                    'iteration': None,
                    'total_iterations': len(results),
                    'successful_iterations': 0,
                })

        # Calculate requested metrics
        successful_ideas = [ib for ib in idea_bests if ib['mean_swaps'] < float('inf')]
        generation_success_rate = (len(successful_ideas) / len(idea_bests)) * 100 if idea_bests else 0
        
        if successful_ideas:
            avg_score = _np.mean([ib['mean_swaps'] for ib in successful_ideas])
            best_score = min([ib['mean_swaps'] for ib in successful_ideas])
            worst_score = max([ib['mean_swaps'] for ib in successful_ideas])
            best_idea = min(successful_ideas, key=lambda x: x['mean_swaps'])
        else:
            avg_score = float('inf')
            best_score = float('inf')
            worst_score = float('inf')
            best_idea = None

        report += "\n## Global Metrics\n"
        report += f"- **Generation Success Rate:** {generation_success_rate:.2f}%\n"
        report += f"- **Average Score (Mean Swaps):** {avg_score if avg_score != float('inf') else 'N/A':.2f}\n"
        report += f"- **Best Score:** {best_score if best_score != float('inf') else 'N/A':.2f}\n"
        report += f"- **Worst Score:** {worst_score if worst_score != float('inf') else 'N/A':.2f}\n"
        report += f"- **Total Number of Tokens:** {self.total_tokens:,}\n"

        # Compare against SOTA if baselines loaded and we have at least 1 successful heuristic
        if best_idea and hasattr(self, 'sota_baselines') and self.sota_baselines:
            report += f"\n## Outperform Rate (vs SOTA)\n"
            report += f"Comparing Best Heuristic ({best_idea['name']} with **{best_score:.2f}** mean swaps) against baseline methods:\n\n"
            
            for method, metrics in self.sota_baselines.items():
                sota_swaps = metrics["mean_swaps"]
                # Outperformance formula: Positive percentage means the heuristic is better (fewer swaps)
                outperform_pct = ((sota_swaps - best_score) / sota_swaps) * 100
                status = "OUTPERFORMS" if outperform_pct > 0 else "UNDERPERFORMS"
                report += f"- **{method.upper()}** (Score: {sota_swaps:.2f}): {status} by **{outperform_pct:.2f}%**\n"
        elif not getattr(self, 'sota_baselines', None):
            report += "\n## Outperform Rate (vs SOTA)\n"
            report += "[SOTA Benchmark CSV not found or failed to load. Outperform metrics skipped.]\n"

        # Sort by best mean_swaps
        idea_bests.sort(key=lambda x: x['mean_swaps'])

        report += "\n## Idea Ranking (by Best Mean Swaps)\n"
        for rank, ib in enumerate(idea_bests, 1):
            status = "✔" if ib['mean_swaps'] < float('inf') else "✘"
            report += (
                f"\n### {rank}. {status} {ib['name']}\n"
                f"- Best Mean Swaps: {ib['mean_swaps'] if ib['mean_swaps'] < float('inf') else 'N/A (all failed)'}\n"
                f"- Best Mean Depth: {ib['mean_depth']}\n"
                f"- Best at Iteration: {ib['iteration']}\n"
                f"- Successful / Total Iterations: {ib['successful_iterations']} / {ib['total_iterations']}\n"
            )

        if best_idea:
            report += f"\n## Overall Winner: {best_idea['name']} (Mean Swaps: {best_idea['mean_swaps']:.2f})\n"
        else:
            report += "\n## No successful ideas.\n"

        return report



    def run_full_pipeline(self):
        console.print(Rule(style="bold white"))
        console.print("[bold white on blue]  HYPER-HEURISTIC SEARCH ENGINE  [/bold white on blue]", justify="center")
        console.print(Rule(style="bold white"))

        stages_completed = []

        # literature review
        if self.run_stage1_literature_review:
            self.literature_review()
            stages_completed.append("I")
        else:
            console.print("[bold yellow]Skipped Literature Review[/bold yellow]")
            lit_review_path = _Path("literature_review.txt")
            if lit_review_path.exists():
                self.literature_insights = lit_review_path.read_text(encoding="utf-8")
                console.print("[bold green]✓ Loaded existing literature review from literature_review.txt[/bold green]\n")
            else:
                console.print("[bold yellow]⚠ No existing literature_review.txt found. Proceeding without literature context.[/bold yellow]\n")

        # ideas generation
        self.ideas_generation()
        stages_completed.append("II")

        # implement and test all ideas
        all_results = self.implementation()
        stages_completed.append("III")

        console.print(Rule("Pipeline Complete", style="bold white"))
        console.print(f"[bold green]Stages {', '.join(stages_completed)} completed successfully.[/bold green]")

        # Save final pipeline summary
        summary = {
            "timestamp": _timestamp(),
            "stages_completed": stages_completed,
            "stage1_run": self.run_stage1_literature_review,
            "total_ideas_generated": len(self.all_generated_ideas),
            "top_ideas_count": len(self.top_ideas),
            "ideas_tested": list(all_results.keys()) if all_results else [],
            "total_tokens_used": getattr(self, 'total_tokens', 0)
        }
        save_json(LOG_DIR, "pipeline_summary.json", summary)
        