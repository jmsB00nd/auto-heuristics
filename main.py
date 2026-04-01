from config import OrchestratorConfig
from orchestrator import OrchestratorV2

config = OrchestratorConfig(
    cli_command=["claude", "--model", "claude-opus-4-6", "-p", "--output-format", "text"],
    backend="ibm_sherbrooke",
    benchmark_dir="benchmarks/qasmbench-large/",
    prompts_dir="/home/jmsb00nd/Documents/auto-heuristics/prompts",
    problem="mapping",
    run_stage3_5_reflection=True,      # Toggled On
    run_stage4_iterative_refinement=True, # Toggled On
    crossover_rate=0.5                 # 50/50 split between mutation and crossover
)

orchestrator = OrchestratorV2(config)
orchestrator.run_full_pipeline()