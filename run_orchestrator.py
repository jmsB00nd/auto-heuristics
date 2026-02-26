from orchestrator import OrchestratorV2
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
    "trial": "bold magenta",
})
console = Console(theme=custom_theme)
CLI_COMMAND = ["claude", "--model", "claude-sonnet-4-6", "-p", "--output-format", "text"]
#CLI_COMMAND = ["gemini", "-m", "gemini-2.5-flash"]
#CLI_COMMAND = ["npx", "gemini", "-m", "gemini-3-flash-preview"]  # Updated to Gemini 3 Flash
#CLI_COMMAND = ["npx", "gemini", "-m", "gemini-3-pro-preview"]  # Updated to Gemini 3 Pro Preview

BACKEND = "ibm_sherbrooke"
BENCHMARK_DIR = "benchmarks/qasmbench-large/"
HISTORY_FILE = "experiment_history.json"

# Stage control
RUN_STAGE1_LITERATURE_REVIEW = False 
# Stage II parameters
MAX_IDEA_GEN_ROUNDS = 3       
TARGET_TOP_IDEAS = 5           
MIN_SCORE_THRESHOLD = 6   

# Stage IV parameters
TIMEOUT_SECONDS = 300     

TOP_IDEAS_TO_IMPLEMENT = 3   
# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------
if not os.path.exists("context_api.txt"):
    console.print("[error]Error: context_api.txt is missing.[/error]")
    exit(1)

if not os.path.exists(BENCHMARK_DIR):
    console.print(f"[error]Error: Benchmark directory '{BENCHMARK_DIR}' does not exist.[/error]")
    exit(1)
    
prompt_path = "/home/jmsb00nd/Documents/auto-heuristics/context_api.txt"
if not os.path.exists(prompt_path):
    print(f"Error: Could not find {prompt_path}")

with open(prompt_path, "r", encoding="utf-8") as f:
    base_prompt = f.read()

# ----------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------
orchestrator = OrchestratorV2(
    cli_command=CLI_COMMAND,
    backend=BACKEND,
    benchmark_dir=BENCHMARK_DIR,
    history_file=HISTORY_FILE,
    use_conversation_mode=True,
    send_context_api=False,
    show_token_counter=True,
    max_idea_gen_rounds=MAX_IDEA_GEN_ROUNDS,
    target_top_ideas=TARGET_TOP_IDEAS,
    min_score_threshold=MIN_SCORE_THRESHOLD,
    run_stage1_literature_review=RUN_STAGE1_LITERATURE_REVIEW,
    top_ideas_to_implement=TOP_IDEAS_TO_IMPLEMENT,
    timeout_seconds=TIMEOUT_SECONDS
)


NUM_ITERATIONS = 5 
    
print(f"Starting iterative search for {NUM_ITERATIONS} iterations...")
results = orchestrator.iterative_heuristic_search(
    num_iterations=NUM_ITERATIONS, 
    base_prompt=base_prompt
)

print(f"\nFinished! Processed {len(results)} heuristics.")
