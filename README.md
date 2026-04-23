# auto-heuristics

LLM-driven framework for automatic heuristic generation for quantum compilation passes.

## Layout

```
framework/        # pipeline agents: orchestrator, memory, llm_client, prompt_manager,
                  #                  idea_parser, evaluator, config
scripts/          # standalone entry points (run_circuit, qlosure_results)
notebooks/        # analysis notebooks
prompts/          # prompt templates (mapping/, routing/)
benchmarks/       # circuit benchmark suites
src/              # quantum domain code (mapping, graph, utils)
qpu/              # backend topology definitions
baselines/        # baseline methods
visualization/    # visualization helpers
generate-plots/   # paper figure scripts
utils/            # shared IO/logging helpers
outputs/          # generated artifacts (heuristics, logs, results, experiment_history.json)
main.py           # primary entry point
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

Standalone scripts:

```bash
python -m scripts.run_circuit
python -m scripts.qlosure_results
```
