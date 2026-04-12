from config import OrchestratorConfig
from orchestrator import OrchestratorV2

config = OrchestratorConfig(
    cli_command="claude --model claude-opus-4-6 -p --output-format text",
    backend="ibm_sherbrooke",
    benchmark_dir="benchmarks/training/",
    prompts_dir="/home/jmsb00nd/Documents/auto-heuristics/prompts",
    problem="mapping",
    run_stage3_5_reflection=True,      
    run_stage4_iterative_refinement=True, 
)

orchestrator = OrchestratorV2(config)
orchestrator.run_full_pipeline()