import re
import subprocess
import types
import signal 
import json 
from pathlib import Path
from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import * 
from src.graph.graph import *
from qpu.src.load_backend import *
import numpy as np
import time
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

console = Console()

class HeuristicTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise HeuristicTimeoutError("Execution timed out")

class Orchestrator:
    def __init__(self, CLI_COMMAND, MAX_TRIALS, HISTORY_FILE, BACKEND, BENCHMARK_DIR):
        self.CLI_COMMAND = CLI_COMMAND
        self.MAX_TRIALS = MAX_TRIALS
        self.HISTORY_FILE = HISTORY_FILE
        self.BACKEND = BACKEND
        self.BENCHMARK_DIR = BENCHMARK_DIR
        
    def query_claude(self, prompt_text):
        # Using a Rich Status spinner for the LLM call
        with console.status("[bold green]Querying Claude for a new heuristic strategy...", spinner="dots"):
            try:
                result = subprocess.run(
                    self.CLI_COMMAND, 
                    input=prompt_text,
                    capture_output=True, 
                    text=True, 
                    check=True, 
                    encoding='utf-8'
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]Error:[/bold red] CLI returned exit code {e.returncode}")
                return None
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] Execution failed: {e}")
                return None

    def parse_response(self, response_text):
        strategy = "Unknown"
        intuition = "None provided"
        s_match = re.search(r"STRATEGY:\s*(.*)", response_text, re.IGNORECASE)
        if s_match: strategy = s_match.group(1).strip()
        i_match = re.search(r"INTUITION:\s*(.*?)(?=CODE:|```)", response_text, re.IGNORECASE | re.DOTALL)
        if i_match: intuition = i_match.group(1).strip()
        code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        return strategy, intuition, code

    def construct_prompt(self, context_api, history):
        history_summary = ""
        if history:
            for h in history:
                history_summary += f"- {h['strategy']}: {h['swaps']} swaps {h['depth']} depth (Result: {h['status']})\n"
        
        return context_api + f"""
                    We Have Already Tested These Ideas :\n
                    
                    {history_summary}
                    """
                    
    def inject_and_run(self, heuristic_code_str, timeout_seconds=300): 
        local_scope = {}
        try:
            exec(heuristic_code_str, globals(), local_scope)
        except Exception as e:
            return {"mean_swaps": float('inf'), "error": f"Syntax Error: {str(e)}"}

        if 'qlosure_poly_heuristic' not in local_scope:
            return {"mean_swaps": float('inf'), "error": "Function not found."}
        
        new_heuristic_func = local_scope['qlosure_poly_heuristic']
        edges = load_backend_edges(self.BACKEND)
        circuit_files = list(Path(self.BENCHMARK_DIR).glob("*.json"))
        
        if not circuit_files:
            return {"error": f"No .json files found in {self.BENCHMARK_DIR}"}

        results = {"swaps": [], "depth": [], "runtimes": [], "failures": []}

        # Styled log for batch start
        console.print(Panel(f"[bold blue]Batch Execution[/bold blue]\nProcessing [bold]{len(circuit_files)}[/bold] circuits\nTimeout: {timeout_seconds}s per circuit", border_style="blue"))
        
        signal.signal(signal.SIGALRM, timeout_handler)

        for circuit_path in tqdm(circuit_files, desc="Routing progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            try:
                data = json_file_to_isl(str(circuit_path))
                router = Qlosure(edges, data)
                router.qlosure_poly_heuristic = types.MethodType(new_heuristic_func, router)
                
                start_time = time.time()
                signal.alarm(timeout_seconds)
                min_swaps, min_depth, _ = router.run(heuristic_method="Qlosure")
                signal.alarm(0)
                
                end_time = time.time()
                results["swaps"].append(min_swaps)
                results["depth"].append(min_depth)
                results["runtimes"].append(end_time - start_time)

            except HeuristicTimeoutError:
                signal.alarm(0)
                console.print(f"[yellow]![/yellow] Timeout on {circuit_path.name}")
                return {"mean_swaps": float('inf'), "error": f"Timeout on {circuit_path.name}"}
            except Exception as e:
                signal.alarm(0)
                results["failures"].append({"circuit": circuit_path.name, "error": str(e)})
        
        signal.alarm(0)
        return {
            "mean_swaps": np.mean(results["swaps"]) if results["swaps"] else float('inf'),
            "mean_depth": np.mean(results["depth"]) if results["depth"] else 0,
            "mean_runtime": np.mean(results["runtimes"]) if results["runtimes"] else 0,
            "total_circuits": len(circuit_files),
            "successful_runs": len(results["swaps"]),
            "failed_runs": len(results["failures"]),
            "error": None if results["swaps"] else "All circuits failed"
        }