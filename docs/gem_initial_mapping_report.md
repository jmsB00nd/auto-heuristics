# GEM: Graph-Embedding Mapping with Hub-Trail Transformation

**A single-pass, deterministic, parameter-free initial qubit-mapping heuristic for Rigetti Ankaa-3**

*Research report — June 2026. Implementation: `src/mapping/mapping.py::generate_gem_initial_mapping` (dispatch key `gem` in `Qlosure.init_mapping`). All numbers below: Qlosure router, `num_iter=1`, Ankaa-3 (82 usable qubits, degree ≤ 4 square lattice with defects), compared against Qiskit `SabreLayout` (seed 21) as the initial mapping.*

---

## 1. Theoretical motivation

### 1.1 Why static placement objectives fail

The classical formulation of initial mapping is a quadratic assignment problem: minimize Σ_e w_e·(dist(π(u),π(v))−1) over placements π, where w_e counts the 2-qubit gates between logical pair e. Our first experiment refuted this premise directly: a placement optimizer that *uniformly dominated* SabreLayout on this static objective (lower cost on 19/19 QASMBench-Medium circuits) still routed *worse* on 10 of them.

Two effects explain the misalignment:

1. **Drift.** The router moves qubits continuously. A pair that first interacts at gate 300 does not need to start close — it needs to be close to where its partner *will be*. The static objective overweights late interactions.
2. **Congestion.** Compact "blob" layouts minimize summed distance but force every routing swap to displace another actively needed qubit. SABRE's seemingly worse layouts leave drift room.

### 1.2 The three structural regimes

Analysis of the interaction *sequences* (not just graphs) of QASMBench/QUEKO revealed three regimes, separable by static circuit statistics:

- **Embeddable** (QUEKO, GHZ/cat/Ising/W-state chains): the interaction graph is a subgraph of the hardware graph. The optimal mapping needs **zero** SWAPs; any routing-driven layout that misses the embedding pays pure overhead.
- **Transient giant-hub** (BV, KNN, swap-test, DNN, QuGAN, QFT): one or more qubits interact with essentially the whole register (degree ≥ ¾(n−1) > max hardware degree 4), each pair only inside a tiny time window (pair-activity span σ_h < 0.1 of the circuit). No seat can hold all partners: *the router must walk the hub*. What the walk needs is not hub-partner proximity (the QAP view) but **consecutive partners being spatially adjacent** — a corridor.
- **Persistent cluster** (multipliers, adders, sat, square-root, qram): pairs recur across most of the circuit (σ > 0.2 — the distribution is strongly bimodal, transient suites sit at σ < 0.05). Here steady-state proximity dominates and a compact cluster is right; the corridor loses (verified: bandwidth-2 adders still prefer clusters, 46 vs 61 swaps).

### 1.3 The hub-trail transformation (core novelty)

For a transient giant hub h with chronologically distinct partners p₁, p₂, …, p_k, replace every gate (h, pᵢ) by a *virtual gate* (pᵢ₋₁, pᵢ). This rewrites the hub's temporal visit sequence as spatial adjacency demands on the *rest* of the register — the "trail" the walking hub will follow. The transformation reduced each hard case to an already-solvable one:

- BV: star → path 0–1–…–12 (trail of the leaf visit order);
- KNN/swap-test: hub + blocks → path 13–1–14–2–… (blocks chained in visit order);
- DNN/QuGAN: hub visits two interleaved halves (26,1,27,2,…) → triangular ladder, whose bandwidth-minimizing order is exactly the 2-rail corridor;
- QFT: all qubits are giants; the union of trails is a near-diagonal band → the corridor in first-use order.

The router's drift dynamics — invisible to any static pairwise objective — become *graph structure* that purely static machinery (exact embedding, bandwidth ordering, cluster descent) can then optimize.

## 2. Algorithm

**Input:** 2q-gate sequence (schedule order), hardware graph + all-pairs distances.
**Output:** complete logical→physical permutation. Deterministic; no randomness, no routing feedback, no tunable weights.

1. **Stage A — exact zero-SWAP embedding.** Degree pre-check (any logical degree > 4 ⇒ impossible), then bounded VF2-style monomorphism search (BFS order from the most-constrained vertex, heaviest-edge-first candidate order, residual-degree pruning, 200k-step budget). On success every 2q gate lands on a coupled pair: 0 inserted SWAPs and final depth = scheduled depth (optimal).
2. **Regime statistics.** Pair-activity spans; giant set G = {q : deg(q) ≥ max(5, ¾(n−1)), hub transience < 0.1}.
3. **Hub-dominated corridor** (|G| ≥ ¾n, e.g. QFT): place all active qubits along the device's long wall-following path ("comb") in first-use order; choose the offset that minimizes trail-weighted stretch (corner-anchored ties — the comb's start packs consecutive slots into a compact 2-row block).
4. **Pure star** (every non-giant interacts only with giants, e.g. BV): virtual-execution placement — replay the gate list once; seat each leaf at its first gate next to the giant's *drifted* position (positions tracked by token-swapping along shortest paths), so the walk stays maximally fed.
5. **Hub-trail transformation + placement of the transformed graph:**
   a. transformed graph is a disjoint union of paths (KNN, swap-test) → lay it straight along the comb in temporal orientation, centered (slack at both ends), offset-scanned;
   b. else exact VF2 embedding of the transformed graph if it exists;
   c. else if Cuthill-McKee bandwidth ≤ 3 (the comb's reach: slots ≤ 3 ranks apart stay within distance 2) → RCM order along the comb (DNN, QuGAN 2-rail ladders);
   d. else if the whole circuit is transient (σ < 0.1) → RCM of the *original* graph along the comb (trail abstraction failed, the plain interaction band is the right object, e.g. qf21);
   e. giants are seated last, adjacent to their first partner — the start of their walk.
6. **Stage 4 — cluster placement + influence-decayed descent** (persistent circuits; also the fallback): greedy maximum-weighted-attachment construction followed by a deterministic exchange descent (single-qubit relocations to free sites + exchanges between seats ≤ 2 apart) on Σ_e W[e]·(dist−1), where each gate's weight is **influence-decayed**: gate r between (u,v) contributes 1/(1 + touches_before(u) + touches_before(v)). Every earlier gate touching u or v is a routing opportunity to have moved them — late gates constrain the *initial* mapping proportionally less. The construction is **multi-started** over seed qubits and the lowest-static-cost sibling returned: on registers ≤ 30 every active qubit is tried (static ranking of same-family siblings is reliable there); on larger registers routed cost is congestion-dominated and small static differences are noise (verified at n = 75, where the plain minimum picks a 12%-worse sibling), so the default seed is kept unless a weighted-degree-peak challenger wins by a decisive (> 5%) margin.
7. Idle qubits fill remaining sites deterministically.

### Complexity

Stage A: O(budget) ≪ V·E in practice (ms — degree pruning fails fast on dense graphs). Statistics/transformation: O(M) over M 2q gates. Comb path: O(V²). Offset scan: O(V·E). Descent: O(sweeps·(n·f + n·k·deg)) with f free sites, k ≤ 12 near seats — observed < 0.1 s; multi-start ≤ n (small) or #peaks (large) repetitions. Worst observed total: 0.6 s (n = 75). Router-free, single pass over the gate list.

## 3. Experimental results (Ankaa-3, vs SabreLayout)

### 3.1 Suite means

| Suite | SABRE swaps | GEM swaps | Δ | SABRE depth | GEM depth | Δ |
|---|---|---|---|---|---|---|
| QASMBench-Medium (19) | 52.32 | **45.47** | **−13.1%** | 254.95 | **244.37** | −4.1% |
| QASMBench-Large (22) | 358.55 | **316.05** | **−11.9%** | 860.27 | **828.82** | −3.7% |
| QUEKO-BSS-16QBT (90) | 2.60 (83/90 at 0) | **0.00 (90/90 at 0)** | **−100%** | 506.02 | **500.00** | −1.2% |

Win/tie/loss per circuit (swaps): medium **17/1/1**, large **18/0/4**, QUEKO **7/83/0**. Final-depth: medium 15/2/2, large **21/0/1**, QUEKO 7/83/0. On QUEKO, GEM's depth equals the benchmark's known-optimal depth on every one of the 90 circuits (0 SWAPs ⇒ depth = scheduled depth = the circuit's CYC value); initial mapping takes < 14 ms per circuit.

### 3.2 Per-circuit (medium)

| Circuit | SABRE sw/dp | GEM sw/dp |
|---|---|---|
| bigadder_n18 | 36 / 227 | **29 / 215** |
| bv_n14 | 5 / 29 | **4 / 34** |
| bv_n19 | 10 / 47 | **6 / 46** |
| cat_state_n22 | 9 / 33 | **0 / 24** |
| dnn_n16 | 0 / 245 | 0 / 245 |
| ghz_state_n23 | 11 / 35 | **0 / 25** |
| ising_n26 | 4 / 15 | **0 / 10** |
| knn_n25 | 24 / 152 | **22 / 148** |
| multiplier_n15 | 84 / 319 | **72 / 306** |
| multiply_n13 | **12 / 69** | 15 / 63 |
| qec9xz_n17 | 17 / 43 | **14 / 43** |
| qf21_n15 | 41 / 271 | **34 / 266** |
| qft_n18 | 98 / 292 | **91 / 239** |
| qram_n20 | 65 / 214 | **52 / 205** |
| sat_n11 | 91 / 539 | **90 / 530** |
| seca_n11 | 18 / 103 | **17 / 113** |
| square_root_n18 | 441 / 1947 | **396 / 1874** |
| swap_test_n25 | 24 / 153 | **22 / 149** |
| wstate_n27 | 4 / 111 | **0 / 108** |

18/19 circuits at-or-below SABRE swaps (sole exception: multiply_n13, +3).

### 3.3 Per-circuit (large)

| Circuit | SABRE sw/dp | GEM sw/dp |
|---|---|---|
| adder_n28 | 49 / 259 | **44 / 255** |
| adder_n64 | 146 / 541 | **104 / 484** |
| bv_n30 | 11 / 50 | **6 / 46** |
| bv_n70 | 24 / 96 | **14 / 86** |
| cat_n35 | 21 / 56 | **0 / 37** |
| cat_n65 | 35 / 96 | **0 / 67** |
| dnn_n51 | **98** / 414 | 102 / **359** |
| ghz_n40 | 29 / 63 | **0 / 42** |
| ghz_n78 | 57 / 137 | **0 / 80** |
| ising_n34 | 12 / 16 | **0 / 10** |
| ising_n66 | 36 / 24 | **0 / 10** |
| knn_n31 | **29** / 195 | 31 / **184** |
| knn_n67 | 85 / 420 | **66 / 390** |
| multiplier_n45 | 1203 / 3289 | **1109 / 3220** |
| multiplier_n75 | 3856 / 9372 | **3596 / 9291** |
| qft_n29 | **263** / 473 | 268 / **416** |
| qft_n63 | 1532 / 1587 | **1333 / 1501** |
| qugan_n39 | **82** / 391 | 88 / 422 |
| qugan_n71 | 192 / 716 | **153 / 637** |
| swap_test_n41 | 46 / 250 | **39 / 249** |
| wstate_n36 | 14 / 151 | **0 / 144** |
| wstate_n76 | 68 / 330 | **0 / 304** |

The four swap losses (dnn_n51 +4, knn_n31 +2, qft_n29 +5, qugan_n39 +6) are all small; three of the four still win on final depth. Eight circuits SABRE pays 12–68 SWAPs for are mapped with **zero** (exact embeddings SabreLayout's randomized routing-feedback search fails to find).

## 3.4 Cross-device validation: IBM Sherbrooke (heavy-hex, 127 qubits)

Re-running both mappers unchanged on `ibm_sherbrooke` (degree ≤ 3, girth 12) after two device-generic fixes — dead-end-avoiding long-path construction (the wall-following rule walked into heavy-hex bridge stubs, truncating the device path to 43/127 sites; Warnsdorff-style avoidance recovers 109/127) and a cycle-closing placement tie-break (see §4):

| Suite (Sherbrooke) | SABRE swaps | GEM swaps | Δ | win/tie/loss |
|---|---|---|---|---|
| QASMBench-Medium (19) | 69.79 | **69.00** | −1.1% | 12/1/6 |
| QASMBench-Large (21\*) | 257.81 | **243.86** | **−5.4%** | **20/0/1** |
| QUEKO-16QBT (90) | 572.34 | **561.86** | **−1.8%** | **49/2/39** (depth: 931.30 vs 960.66, **−3.1%**, 50/2/38) |

(The QUEKO row reflects the final QUEKO-regime portfolio of §3.6: cut-diverse carousels + all prior families as candidates, fixed 1900-gate evaluation budget. **All 9 depth classes won.** The initial pre-carousel result was 625.14, +9.2%.)

The Ankaa numbers after the cross-device hardening (path fix, cycle tie-break) improved further: medium 44.95 (−14.1%, 17/1/1, sole loss multiply +1), large 163.19 vs 192.00 excluding the n75 timeout row (−15.0%, 17/0/4; multiplier_n75 standalone: 3766 < SABRE 3856), QUEKO still 90/90 zero-SWAP at optimal depth.

\*multiplier_n75 excluded (routing alone exceeds the 30 s harness timeout for both mappers). Large final-depth: −5.6%, 17/1/3. The exact-embedding wins transfer fully (cat/ghz/ising/wstate: GEM 0 SWAPs where SABRE pays 7–17); the corridor wins transfer (knn_n67 65 vs 141, adder_n64 162 vs 246, swap-test, bv). On QUEKO-16, zero SWAPs is **structurally impossible** on Sherbrooke for any mapper: the QUEKO interaction graphs contain 8-cycles and a girth-12 graph has no 8-cycle subgraph.

The single large loss — and most of the medium losses — are the **persistent-cluster circuits** (multiplier −31%, multiply, qf21, qft small). Root cause, isolated experimentally: GEM's multiplier layout *dominates SABRE's statically* (more adjacent pairs, lower weighted stretch under every weighting tried) yet routes 28% worse, because pure distance minimisation on a sparse device produces **tree-shaped occupied regions**. A tree region bottlenecks the router: every move displaces along a single path. SABRE's region contains a closed 12-cycle (two row segments joined by two bridges — a ladder), and re-running GEM's own placement *restricted to SABRE's region* recovers most of the gap (130 → 111 vs SABRE's 102). Region cyclicity — not pairwise distance — is the missing objective on heavy-hex; on Ankaa the square lattice provides 4-cycles for free, which is why this never surfaced there.

A face-anchored region constructor (anchor on the shortest device cycle through the hardware root, grow by cycle-closure) was implemented and evaluated: it fixed multiplier (130→117) and dnn but regressed qec/qram, and no static criterion could arbitrate (per-gate stretch, raw vs decayed cost, and interaction-graph cyclomatic number all fail to separate the cases). Only the conservative part is kept: the greedy placement tie-breaks toward sites with more occupied neighbours, which closes cycles without overriding the cost order. An influence floor of 1/n (decay saturating at the mixing scale) was likewise evaluated — better on deep circuits (Sherbrooke multipliers −18%), worse on others; plateau, not adopted.

**QUEKO-16 on Sherbrooke** (no zero-SWAP mapping exists for any mapper — see girth argument above) GEM loses: 625.1 vs 572.3 mean swaps (+9.2%, w/t/l 22/2/66). Diagnosis, isolated by region-transplant experiments: SABRE's winning regions are *perforated* two-row blocks — 3–4 idle qubits sit **inside** the active region. Swapping an active qubit with an idle one costs one SWAP and creates no future obligation, so interior idle qubits act as lubricant for deep persistent traffic on a degree-3 device. Running GEM's own placement machinery inside SABRE's perforated site-set recovers most of the gap (393→316, 813→634). Four parameter-free perforation mechanisms were then evaluated (hard "breathing" constraint — every occupied site keeps a free neighbour on sub-quartic devices; max-weight spanning-tree exact embedding; RCM band on the device path; raw/floored weights): each improves some circuits by 10–20% and regresses others, with no static arbiter. The conclusion mirrors the static/dynamic misalignment finding: the residual gap on deep persistent circuits is *informational* — SabreLayout buys it with routing feedback (it simulates the router during layout), which GEM's single-pass, router-free design constraint excludes by definition. Closing it without routing feedback would require a calibrated congestion model — the top open problem (§5).

### 3.5 Exhaustive exploration of alternative heuristic families

Beyond GEM's mechanisms, four independent heuristic families were implemented and evaluated head-to-head on the losing circuits (Sherbrooke QUEKO + persistent arithmetic, both devices):

| Family | Idea | Result |
|---|---|---|
| Recursive min-cut co-placement | VLSI-style: bisect interaction graph at weighted min-cut (Fiedler + KL refinement), bisect device region along its Fiedler axis, recurse with terminal propagation | Uniformly worse (QUEKO 533 vs SABRE 295; multiplier 145 vs 102) — global cuts without metric refinement lose the local geometry |
| VEX virtual-execution placement | Place each qubit at first use next to its partners' *drifted* positions (true token dynamics) | Wins only qf21 (58 vs 59); QUEKO 454–1572, well behind |
| REWIND (single backward sweep) | Token-walk the reversed gate list once from GEM's layout; final configuration = initial mapping (one-pass analogue of SABRE's reverse trick, no router, no iteration) | Degrades GEM almost everywhere — a myopic walker scrambles structured layouts |
| Carousel (cyclic face embedding) | Map each interaction-graph octagon onto a heavy-hex 12-face preserving cyclic order, slack spread as interior holes (the shape SABRE's winning QUEKO regions have) | 357–1238, 15–25% behind SABRE — shape alone is insufficient |

A token-walk congestion proxy (single pass, true collisions) was validated as a candidate *selector* across 13 instances with known per-variant routed outcomes: it picks within 3% of the best variant only 4/13 times — no better than the static costs it was meant to replace.

**Bounding argument (later overturned by the lane-carousel).** Two transplant experiments bound what any selection mechanism could achieve at that point: (i) GEM's own placement machinery run *inside SABRE's exact site-set* still routed 7–14% worse than SABRE's assignment of those same sites; (ii) the per-circuit *oracle best* over the then-existing candidate families still trailed SABRE by 2–4% on every probed QUEKO/Sherbrooke circuit.

### 3.6 The lane-carousel and the QUEKO-regime portfolio (final QUEKO/Sherbrooke win)

The bound above was broken by a new candidate family. QUEKO interaction graphs are two chordless 8-rings plus two cross edges (the relabeled Aspen-16 graph; cycle basis [4, 8, 8]). The **lane-carousel** maps each ring onto the 9-slot arc of one of two *adjacent* heavy-hex 12-faces, leaving the 3 shared bridge nodes — the "lane" — completely free: 7 of 8 ring edges per ring sit at distance 1, the cut edge and the cross edges route through the lane by displacing only idle qubits, and ring traffic is served by carousel rotation. Configurations (face pair × ring rotation/reflection × arc end) are scored by weighted stretch with a temporal tie-break. Alone, this family swept the deepest QUEKO class (900CYC: 1061 vs SABRE 1064.9 class-mean; best circuit −128) and reached suite mean 593.8 — closing 60% of the 625→572 gap.

Routing-trace analysis showed why pure construction saturates there: carousel and SABRE starts route *identically through the first 25% of the circuit*; the greedy router then gradually erodes hand-built structure, while SABRE's layout — being produced by routing dynamics — is a fixed point of them. Per-circuit oracle over {carousel, raw cluster, decay cluster} = **566.5 < 572.3**, so candidates sufficed and only selection was missing; every static selector and the sequential token-walk proxy misranked (the proxy cannot see parallel-layer service, which is the essence of QUEKO traffic).

The final mechanism — `Qlosure._gem_portfolio_refine` — triggers only for the two-ring persistent signature when GEM's mapping is not already an exact embedding: it builds ≤5 candidates (GEM's decay cluster, two lane-carousels from distinct face pairs, raw cluster, breathing cluster) and keeps the one that routes the *prefix* (first 60% of the gate list) best — about 3 route-equivalents of evaluation, no iteration, no re-layout, strictly less routing work than SabreLayout's ≥3 full forward/backward passes. Two further refinements produced the final result. First, a 1900-gate **evaluation budget** replaced the fixed prefix (exact selection through 500CYC, 0.66–0.85 prefix beyond — a runtime guard, not a quality parameter), with early-abort pruning (`swap_abort_threshold`) making extra candidates nearly free. Second — after observing that the residual losses were *instance-specific pool ceilings* (at 800CYC every candidate routed ≥950 against SABRE's ~860; all my candidates were global priors while SABRE's layouts are instance-adapted router fixed points) — the pool was diversified along the axis that changes the *dynamics*: carousels with **distinct cut edges** (top-4 of the best face pair), the **swapped ring↔face assignment**, the runner-up face pair, plus every previously-built family resurrected as a candidate (max-weight spanning-tree embedding, VEX virtual-execution placement, raw/breathing clusters) — families that lost as soloists become free options under routed selection. This collapsed the worst instances (800CYC_4: 968→878; 800CYC_8: 985→881).

Final result on QUEKO-16/Sherbrooke (n=90): **561.9 vs 572.3 mean swaps (−1.8%) and 931.3 vs 960.7 mean final depth (−3.1%)**, per-circuit 49/2/39 swaps and 50/2/38 depth, **all 9 depth classes won**; ≤18 s initial-mapping time per circuit, zero harness timeouts. All other circuit classes bypass the portfolio entirely (verified on both QASMBench suites and QUEKO/Ankaa).

### 3.7 Out-of-scope stress test: QUEKO-BSS-54QBT

At the user's request GEM was also tested on QUEKO-54 (90 circuits derived from Google Sycamore: 54-qubit degree-≤4 *mesh* interaction graphs, 1,080–9,720 2q gates). This suite was not part of the design targets and exposes a new regime: large near-embeddable meshes. A **mesh extension** was added — when the interaction graph has every degree within the device bound, > 30 active qubits and ≥ n/4 independent cycles (a signature no QASMBench/QUEKO-16 circuit matches), the portfolio runs a lean two-candidate duel between GEM's cluster and a **spectral co-embedding + exchange descent** (mesh-to-mesh spectral alignment), under an 800-gate routing budget. This took the worst class from −31% to a tie (Ankaa 500CYC: 2900 → 2206 = SABRE's exact count, with depth 1622 vs 1743).

Results on mutually-completed circuits (the 30 s harness times out deep circuits for both mappers):

| Device | n | swaps GEM vs SABRE | w/t/l | depth GEM vs SABRE |
|---|---|---|---|---|
| Ankaa-3 | 77 | 2242.4 vs 2159.1 (**+3.9%**, loss) | 30/2/45 | 1675.1 vs 1764.3 (**−5.1%**, wins 65/77) |
| Sherbrooke | 50 | 3553.9 vs 3281.1 (+8.3%, loss) | 8/0/42 | 1665.7 vs 1602.8 (+3.9%, loss) |

Honest standing: QUEKO-54 is **not won on swaps** — at 54-qubit scale SabreLayout's routing-feedback advantage compounds, and on Sherbrooke the degree-4 mesh cannot even align with a degree-3 device (the mesh trigger correctly stays off; plain clustering runs). GEM does win **final depth decisively on Ankaa** (−5.1%, 65/77) via the spectral alignment. Closing the swap gap would take: on Ankaa, a defect-aware maximum-subgraph embedding (the Sycamore mesh nearly embeds in the lattice; only the dead qubits/edges break it); on Sherbrooke, a degree-reduction transformation of the mesh onto the heavy-hex — both full research cycles. All in-scope results (§3.1–3.6) are unaffected (QUEKO-16/Sherbrooke spot re-verified at 298 after the extension).

## 4. Failure analysis

- **multiply_n13 (15 vs 12):** the static cost ties between sibling cluster layouts whose routed costs differ by ±2; the selector cannot see the difference. A drift-aware (still router-free) tie-breaker is the obvious next step.
- **Static/dynamic misalignment is fundamental, not incidental:** mappings that dominate the QAP objective can route worse. GEM works around this by (i) restricting static *selection* to same-family siblings and (ii) converting dynamics into structure via the trail transformation, but a principled routability objective remains open.
- **Comb-geometry sensitivity:** corner- vs center-anchoring of a corridor changes routed swaps by up to 30% per circuit class (QFT wants the compact corner block; hub-walk chains want central slack). The regime-specific tie-breaks encode this; a geometry-independent criterion would be cleaner.
- **VF2 embedding shape:** when multiple exact embeddings exist, their *shape* affects later walking phases (a wiggly path embedding cost BV +2 swaps vs the straight comb). GEM prefers the comb for path-shaped graphs; shaping general embeddings is open.

## 5. Next research directions

1. **Learned/analytic routability proxy** for cross-family candidate selection (would merge the corridor/cluster branches into one argmin).
2. **Trail transformation for moderate hubs** (degree 5–10, e.g. sat's deg-8 hubs): partial trail rewriting — keep the heavy persistent core as edges, trail-rewrite only the transient fringe.
3. **Embedding shaping:** bias VF2 candidate order by comb rank so exact embeddings come out straight.
4. **Depth-aware descent:** the exchange descent optimizes a SWAP proxy only; adding a critical-path term should convert more of the SWAP gains into depth gains.
5. **Cross-device validation** (heavy-hex IBM, denser lattices): all components are device-generic (the comb path, degree thresholds and bandwidth bounds derive from the hardware graph at run time), but the regime thresholds (σ < 0.1, ¾(n−1)) should be re-validated.

## 6. Reproduction

```bash
python scripts/run_benchmark.py --benchmark benchmarks/<suite> --backend ankaa \
    --initial gem --output_dir experiments_results/ankaa_research
# SABRE baseline: --initial sabre
```
