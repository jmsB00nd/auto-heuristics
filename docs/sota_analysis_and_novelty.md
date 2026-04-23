# Framework Analysis vs SOTA LLM Hyper-Heuristics

Date: 2026-04-22
Branch: routing/cost-fn-v2

## Where the project sits today

`OrchestratorV2` is architecturally a solid **EoH + ReEvo hybrid**:

| Capability | Your code | SOTA equivalent |
|---|---|---|
| Literature-primed idea generation | `literature_review` → `ideas_generation` | EoH initialization |
| Mutation + crossover with diversity pool | `iterative_refinement` (`random.random() < crossover_rate`) | EoH E1/M1–M3 |
| Reflection feeding into refinement | `reflection` → `memory.get_latest_reflection()` | ReEvo long-term reflection |
| Parent selection w/ Jaccard text diversity | `memory.get_diverse_parents` — word-level Jaccard | HSEvo (Shannon-Wiener over code) |
| Stagnation trigger → diversity | `force_diversity` after N non-improving rounds | Island migration (FunSearch) |
| Single scalar fitness (`mean_swaps`) | `evaluator.py` returns `mean_swaps`, `mean_depth` but only `mean_swaps` drives selection | MEoH uses multi-objective Pareto |

### Current gaps vs published 2025–2026 methods

- **HSEvo (AAAI 2025)** beats FunSearch / EoH / ReEvo with harmony search + Shannon-Wiener *code-level* diversity — Jaccard on the *description* field (`memory.py:88-95`) is a weaker proxy.
- **MCTS-AHD (ICML 2025)** does what the stagnation-trigger approximates, but principled (UCT over a search tree of heuristic variants).
- **MEoH (2025)** — `mean_depth` is computed but discarded in selection.
- **EoH-S (AAAI 2026)** — designs a *set* of complementary heuristics with instance-wise performance vectors (not one average). +60% on combinatorial tasks.
- **ReVEL (arXiv Mar 2026)** — multi-turn reflection with *behavioral clustering* and structured performance feedback, not just top-K scores.
- **CycleQD (ICLR 2025) / EvoLattice (Dec 2025)** — MAP-Elites / QD archives for LLM-generated code.
- **Racing** (successive halving / Hoeffding) for LLM-HH evaluation is emerging but not mainstream — current code evaluates every heuristic on the full benchmark set (`evaluator.py:75`), wasting ~60–80% of compute on bad heuristics.

## Quantum-compilation LLM-HH: the gap

No published LLM-*hyper-heuristic* method (generate–evolve–refine code) specifically targeting **qubit mapping / routing**. What exists in 2025–26:

- **2505.07711** fine-tunes an LLM to directly output circuit *partitions* (53% accuracy) — not evolutionary, not code-generating.
- **POPL'26 "Generating Compilers for QMR"** — DSL + parametric algorithm; no LLM.
- **2506.09323** — DRL for modular mapping.

The domain is open. The risk is publishing a "port EoH to qubit mapping" paper that reviewers reject as incremental. A domain-native mechanism that *couldn't* trivially be ported back to TSP is what makes this publishable.

## Proposed novelty — strongest angle

### Trace-Anchored Counterfactual Reflection (TACR)

Exploit the fact that quantum routing is **decision-by-decision inspectable** in a way TSP/BPP are not.

**The core mechanism:**

Instead of reflection getting `{name, mean_swaps, description}` tuples (current `reflection()` at `orchestrator.py:251-259`), the evaluator becomes a **diagnostic oracle** that:

1. Replays the heuristic on each training circuit *with decision logging* — at every routing step, records the chosen swap, the coupling-graph state, and the frontier of pending 2-qubit gates.
2. For the top-K "regret decisions" (decisions where a counterfactual alternative swap would have reduced downstream swap count), emit a structured critique:
   ```
   {circuit: bss_16q_05, cycle: 23, chosen: SWAP(q2,q5),
    counterfactual: SWAP(q1,q3), regret: 3 swaps,
    cause: "q3 needed by q1 for next 12 CNOTs; chosen swap broke that locality"}
   ```
3. Aggregate across circuits into *behavioral failure patterns* ("heuristic greedily reduces current-layer distance at expense of layers +2/+3").
4. Feed these into the mutation prompt — so the LLM doesn't just see a score, it sees **where and why** its last proposal made a wrong call.

**Why this is publishable as SOTA:**

- **Strictly richer than ReVEL's structured feedback.** ReVEL clusters heuristics by scalar performance signals; TACR gives causal attribution at *decision-point granularity*.
- **Uniquely enabled by the quantum domain.** Routing is a deterministic sequence of discrete choices over a known graph — counterfactual replay is cheap (~2× evaluation cost), unlike in TSP where counterfactual tours aren't locally defined.
- **Orthogonal to existing SOTA mechanisms** — HSEvo-style code diversity and MEoH-style multi-objective Pareto can still stack on top.

### Supporting contributions (for a full paper, not a workshop note)

- **(B) QMAP-Elites archive** with domain-native descriptors: (lookahead horizon, locality bias, swap/depth tradeoff, coupling-graph sensitivity). CycleQD / EvoLattice have MAP-Elites-for-LLM; no one has quantum-semantic behavior axes.
- **(C) Racing evaluator**: start on 3 circuits, Hoeffding-advance promising heuristics to the full suite. Likely 3–5× iteration throughput — easy win.

### Story arc

> Port ReEvo / EoH / HSEvo to QMR → they underperform classical Sabre on deeper circuits → diagnose why (scalar reflection is insufficient for long-horizon decisions) → introduce TACR → show per-decision critique closes the gap and then beats Sabre/QMAP on QUEKO/QASMBench. Then layer QMAP-Elites and racing for efficiency + generalization across topologies (ibm_sherbrooke, Google Sycamore, modular).

## Concrete hooks into existing code

Minimal-churn integration:

- `src/mapping/routing.py` — `Qlosure.run()` needs a `trace=True` mode returning per-cycle decision records (choice, frontier, alternatives considered).
- New `agent/trace_analyzer.py` — consumes the trace, produces the structured critique.
- `agent/evaluator.py:102-108` — when successful, also dump trace → `trace_analyzer` → append critique to stats dict.
- `agent/memory.py` — store `critique` alongside `code` / `mean_swaps`.
- `agent/orchestrator.py:251-269` (`reflection`) and `:354-360` (refinement prompt build) — inject critiques into the prompt context.
- `prompts/mapping/reflection.txt` and `refinement.txt` — add a "FAILURE CASES" section schema.

No changes to `src/` beyond adding a trace parameter. No restructure.

## Positioning

- **Target venues:** QCE 2026 (quantum systems audience) or ICML 2026 / NeurIPS 2026 (AHD audience).
- **Required baselines:** Sabre, QMAP, ReEvo-QMR, HSEvo-QMR (own ports), MEoH-QMR.
- **Ablations:** TACR alone, +QMAP-Elites, +racing.

## Sources

- [EoH-S: Evolution of Heuristic Set using LLMs (AAAI 2026)](https://arxiv.org/abs/2508.03082)
- [ReVEL: Multi-Turn Reflective LLM-Guided Heuristic Evolution (arXiv Mar 2026)](https://arxiv.org/html/2604.04940)
- [ReEvo (NeurIPS 2024)](https://ai4co.github.io/reevo/)
- [HSEvo (AAAI 2025)](https://github.com/datphamvn/HSEvo)
- [MCTS-AHD (ICML 2025)](https://github.com/zz1358m/MCTS-AHD-master)
- [EvoLattice: MAP-Elites for LLM-Guided Program Discovery (Dec 2025)](https://arxiv.org/html/2512.13857)
- [CycleQD (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/755acd0c7c07180d78959b6d89768207-Paper-Conference.pdf)
- [Generating Compilers for Qubit Mapping and Routing (POPL'26)](https://arxiv.org/abs/2508.10781)
- [Circuit Partitioning Using LLMs for Quantum Compilation (2025)](https://arxiv.org/html/2505.07711v1)
- [HeuriGym benchmark (2025)](https://www.cs.cornell.edu/gomes/pdf/2025_chen_arxiv_heurigym.pdf)
- [Rethinking LLM-Driven Heuristic Design: Dynamics-Aware Optimization (2026)](https://arxiv.org/html/2601.20868v1)
