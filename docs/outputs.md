# Run Artifacts Reference

Every pipeline invocation writes a self-contained, timestamped bundle under
`outputs/logs/{run_id}/` plus a few append-only updates to the shared pools
(`outputs/heuristics/`, `outputs/experiment_history.json`).

`run_id` format: `YYYY-MM-DD_HH-MM-SS_{experiment-name}` (slug defaults to
`run`, settable via `main.py --experiment-name NAME`).

## Per-run bundle (`outputs/logs/{run_id}/`)

| Path | Produced by | Format | Schema |
|------|-------------|--------|--------|
| `run_metadata.json` | orchestrator init | JSON | [`run_metadata.schema.json`](schemas/run_metadata.schema.json) |
| `run.log` | all stages | text, `ts \| level \| stage \| msg` | — |
| `events.jsonl` | all stages | JSON lines, one event per line | — |
| `summary.json` | end of pipeline | JSON | [`run_summary.schema.json`](schemas/run_summary.schema.json) |
| `final_pipeline_report.txt` | end of pipeline | text, human-readable | — |
| `heuristics/*.py` | stages III & IV | Python source | — |
| `literature_review/literature_review.txt` | stage I (optional) | text | — |
| `idea_generation/raw_ideas.txt` | stage II | text (raw LLM) | — |
| `idea_generation/top_ideas.json` | stage II | JSON array | — |
| `idea_generation/global_memory_resume.txt` | stage II | text | — |
| `implementation/idea_{i}_{name}/prompt.txt` | stage III | text | — |
| `implementation/idea_{i}_{name}/raw_response.txt` | stage III | text | — |
| `implementation/idea_{i}_{name}/heuristic.py` | stage III | Python source | — |
| `implementation/idea_{i}_{name}/evaluation_results.json` | stage III | JSON | [`evaluation_result.schema.json`](schemas/evaluation_result.schema.json) |
| `implementation/final_results.json` | stage III | JSON | — |
| `reflection/reflection_insights.txt` | stage III.V (optional) | text | — |
| `iterative_refinement/round_{N}/…` | stage IV (optional) | same shape as `implementation/idea_*/` | — |
| `iterative_refinement/refinement_results.json` | stage IV | JSON | — |

### `run_metadata.json`
Single source of truth for reproducibility. Captures:
- `run_id`, `experiment_name`, UTC timestamp
- git commit (+ dirty flag), framework version, Python version, platform
- LLM: CLI command and model name (parsed from `--model`)
- RNG seeds (Python / NumPy)
- Full `OrchestratorConfig` dump
- Feature flags (which stages were active)
- Backend: name, qubit count, topology-file hash
- Benchmarks: directory, circuit list, set hash (covers membership + contents)
- Prompts: `{filename -> sha256}` for every `*.txt` under `prompts/{problem}/`

### `events.jsonl`
One JSON object per line: `{ts, level, stage, event, message, data}`. Emitted at
stage boundaries and key decisions (`idea_evaluated`, `round_evaluated`,
`new_best`, `run_completed`). Parse with `jq` or pandas for cross-run analysis.

### `summary.json`
Machine-readable version of `final_pipeline_report.txt`. Contains per-stage
timings, token totals, counts, full ranking with per-heuristic code paths, and
the overall winner.

## Shared pools (global, updated across runs)

| Path | Purpose | Schema |
|------|---------|--------|
| `outputs/heuristics/*.py` | Pool of successful heuristic implementations. Dual-written with per-run copies. | — |
| `outputs/experiment_history.json` | Cumulative memory consumed by subsequent runs. New entries carry a `run_id` field. | [`history_entry.schema.json`](schemas/history_entry.schema.json) |

## Regenerating schemas

```bash
python scripts/export_schemas.py
```

Writes `docs/schemas/*.json` from the pydantic models in `agent/schemas.py`.

## Running an experiment

```bash
python main.py --experiment-name baseline-v2 --seed 42
```

Artifacts land in `outputs/logs/{timestamp}_baseline-v2/`. With `--seed`
specified, Python `random` and NumPy RNGs are seeded and recorded in
`run_metadata.json` under `seeds`.
