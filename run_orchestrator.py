from orchestrator import Orchestrator
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "trial": "bold magenta"
})
console = Console(theme=custom_theme)

#CLI_COMMAND = ["claude", "-p", "--output-format", "text"]
CLI_COMMAND = ["gemini", "-m", "gemini-3-flash-preview"]
MAX_TRIALS = 10
HISTORY_FILE = "experiment_history.json"
BACKEND = "ibm_sherbrooke"
BENCHMARK_DIR = "/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-large"

orchestrator = Orchestrator(CLI_COMMAND, MAX_TRIALS, HISTORY_FILE, BACKEND, BENCHMARK_DIR)

console.print(Rule(style="bold white"))
console.print("[bold white on blue]  HYPER-HEURISTIC SEARCH ENGINE  [/bold white on blue]", justify="center")
console.print(Rule(style="bold white"))

try:
    with open("context_api.txt", "r") as f: context_api = f.read()
except:
    console.print("[error]Error: context_api.txt missing.[/error]")

history = [] 

for i in range(1, MAX_TRIALS + 1):
    console.print(f"\n[trial]Trial {i} of {MAX_TRIALS}[/trial]")
    
    prompt = orchestrator.construct_prompt(context_api, history)
    response = orchestrator.query_claude(prompt)
    
    if not response:
        console.print("[error]No response from LLM. Skipping...[/error]")
        continue

    strategy, intuition, code = orchestrator.parse_response(response)
    if not code:
        console.print("[error]Parse failed: No code block found.[/error]")
        continue
        
    # Pretty display of the proposed strategy
    console.print(Panel(f"[bold green]Idea:[/bold green] {strategy}\n[italic]{intuition}[/italic]", 
                        title="LLM Proposal", border_style="green"))
    
    stats = orchestrator.inject_and_run(code)
    
    if stats['error']:
        console.print(f"   [bold red]✘ FAIL:[/bold red] {stats['error']}")
        status = "Crashed"
        swaps = float('inf')
        depth = float('inf')
    else:
        # Success output
        summary = (f"[success]✔ SUCCESS[/success]\n"
                   f"Avg Swaps: [bold]{stats['mean_swaps']:.2f}[/bold]\n"
                   f"Avg Depth: [bold]{stats['mean_depth']:.2f}[/bold]")
        console.print(summary)
        
        status = "Success"
        swaps = stats['mean_swaps']
        depth = stats['mean_depth']
        
        filename = f"heuristics/idea_{i}_{strategy.replace(' ', '_')}.py"
        os.makedirs("heuristics", exist_ok=True)
        with open(filename, "w") as f:
            f.write(f"# Strategy: {strategy}\n# Intuition: {intuition}\n# Stats: {stats}\n\n{code}")

    history.append({"strategy": strategy, "swaps": swaps, "depth" : depth, "status": status})
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

console.print(Rule("Search Complete", style="bold white"))