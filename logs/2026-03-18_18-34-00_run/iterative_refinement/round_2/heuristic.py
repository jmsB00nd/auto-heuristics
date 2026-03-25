def init_mapping(self):
    """
    Layer-Weighted Pair-Seed Greedy with Perturbation Restarts (LWPSGPR)

    Synthesises the best-performing elements from all previous rounds:

    1. Layer-normalised temporal decay [new vs Round 1]:
       Compute topological circuit layers from qubit data dependencies.
       Gate at layer L receives weight exp(-beta * L / (max_layer+1)).
       This concentrates placement quality on near-term gates, which drive
       the most routing before any SWAP-induced remapping occurs.
       beta=2.0 → last-layer weight ≈ 13.5% of first-layer weight.

    2. Pair seed [Round 1, retained]:
       The highest-weight interaction pair is placed on the hardware pair
       that minimises their physical distance (tie-break: highest combined
       hw degree, favouring central nodes). Keeps the most critical
       two-qubit relationship close from the start.

    3. Full free-qubit greedy expansion [Round 1, retained]:
       At each step ALL unoccupied physical qubits are candidates
       (not just BFS neighbours of the current frontier). This avoids
       forcing a compact cluster that may not match the circuit's
       interaction topology. Cost = weighted sum of BFS distances to
       ALL already-placed interaction partners.

    4. First-improvement 2-opt local search [Round 1, retained]:
       Iterates over all O(n^2) logical qubit pairs; applies the first
       swap that reduces total cost. max_passes=10 to bound runtime.

    5. Perturbation-restart escape [Round 3 idea, now on correct base]:
       After 2-opt converges, repeatedly: sort interaction pairs by
       conflict (w*dist), rotate the window of high-conflict pairs to
       diversify targets, apply 2–4 forced swaps, re-run 2-opt, keep
       the globally best result. This escapes shallow local optima
       that first-improvement 2-opt cannot climb out of.

    Why this combination beats its predecessors:
    - Rounds 2-3 regressed by restricting expansion to BFS neighbours;
      this version retains Round 1's full search (the dominant advantage).
    - Round 1 had no temporal bias and no restart mechanism, leaving
      early-gate conflicts unweighted and stuck in a single local optimum.
    - Perturbation restarts are now applied on top of a stronger base
      solution (pair-seed + full-search), so each restart starts from
      a better basin and is more likely to find a genuine improvement.
    """
    import math
    from collections import defaultdict

    gate_list = list(self.access.items())
    n_gates = len(gate_list)

    # ── Compute topological circuit layers ────────────────────────────────
    last_gate_on_qubit = {}
    gate_layer_arr = []
    for gate_idx, (gate, qubits) in enumerate(gate_list):
        layer = 0
        for q in qubits:
            if q in last_gate_on_qubit:
                prev = gate_layer_arr[last_gate_on_qubit[q]] + 1
                if prev > layer:
                    layer = prev
        gate_layer_arr.append(layer)
        for q in qubits:
            last_gate_on_qubit[q] = gate_idx

    max_layer = max(gate_layer_arr) if gate_layer_arr else 0

    # ── Layer-decayed interaction graph ───────────────────────────────────
    # beta=2.0: layer-0 weight=1.0, last-layer weight≈exp(-2)≈0.135
    beta = 2.0
    logical_qubit_set = set()
    interaction_weight = {}   # lq -> {lq2: weight}
    total_iw = defaultdict(float)

    for gate_idx, (gate, qubits) in enumerate(gate_list):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            layer = gate_layer_arr[gate_idx]
            w = math.exp(-beta * layer / (max_layer + 1))
            if q1 not in interaction_weight:
                interaction_weight[q1] = {}
            if q2 not in interaction_weight:
                interaction_weight[q2] = {}
            interaction_weight[q1][q2] = interaction_weight[q1].get(q2, 0.0) + w
            interaction_weight[q2][q1] = interaction_weight[q2].get(q1, 0.0) + w
            total_iw[q1] += w
            total_iw[q2] += w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # ── Fallback ──────────────────────────────────────────────────────────
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    hw_degree = {p: len(self.backend[p]) for p in physical_qubits}

    # ── Seed: highest-weight interaction pair → best central HW pair ──────
    best_lq_pair = None
    best_seed_w = -1.0
    for lq1 in logical_qubits:
        if lq1 not in interaction_weight:
            continue
        for lq2, w in interaction_weight[lq1].items():
            if lq1 < lq2 and w > best_seed_w:
                best_seed_w, best_lq_pair = w, (lq1, lq2)

    lq_to_phys = {}
    placed_phys = set()

    if best_lq_pair is not None:
        lq1_s, lq2_s = best_lq_pair
        best_phys_pair = None
        best_hw_score = (float('inf'), float('inf'))
        for p1 in physical_qubits:
            for p2 in physical_qubits:
                if p1 >= p2:
                    continue
                d = self.distance_matrix[p1][p2]
                if d == float('inf'):
                    continue
                score = (d, -(hw_degree[p1] + hw_degree[p2]))
                if score < best_hw_score:
                    best_hw_score = score
                    best_phys_pair = (p1, p2)
        if best_phys_pair:
            lq_to_phys[lq1_s] = best_phys_pair[0]
            lq_to_phys[lq2_s] = best_phys_pair[1]
            placed_phys.update(best_phys_pair)

    if not lq_to_phys:
        seed_phys = max(physical_qubits, key=lambda p: hw_degree[p])
        lq_to_phys[logical_qubits[0]] = seed_phys
        placed_phys.add(seed_phys)

    # ── Full free-qubit greedy expansion ──────────────────────────────────
    unplaced = set(lq for lq in logical_qubits if lq not in lq_to_phys)

    while unplaced:
        best_lq = max(
            unplaced,
            key=lambda lq: (
                sum(interaction_weight.get(lq, {}).get(pl, 0.0) for pl in lq_to_phys),
                total_iw[lq],
            ),
        )

        free_phys = [p for p in physical_qubits if p not in placed_phys]
        if not free_phys:
            break

        iw_best = interaction_weight.get(best_lq, {})

        def placement_cost(p, _iw=iw_best):
            cost = 0.0
            for pl_lq, pl_phys in lq_to_phys.items():
                w = _iw.get(pl_lq, 0.0)
                if w == 0.0:
                    continue
                d = self.distance_matrix[p][pl_phys]
                if d != float('inf'):
                    cost += w * d
            return cost

        best_phys = min(free_phys, key=placement_cost)
        lq_to_phys[best_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.discard(best_lq)

    # ── Assign remaining (non-interacting) logical qubits ─────────────────
    placed_phys_set = set(lq_to_phys.values())
    for lq in logical_qubits:
        if lq not in lq_to_phys:
            for p in physical_qubits:
                if p not in placed_phys_set:
                    lq_to_phys[lq] = p
                    placed_phys_set.add(p)
                    break

    # ── Build mapping from lq_to_phys via in-place transpositions ─────────
    def build_mapping(lq_to_p):
        m = list(range(self.num_qubits))
        rm = list(range(self.num_qubits))
        for lq, tp in lq_to_p.items():
            cp = m[lq]
            if cp == tp:
                continue
            dlq = rm[tp]
            m[lq] = tp
            m[dlq] = cp
            rm[tp] = lq
            rm[cp] = dlq
        return m, rm

    mapping_dict, reverse_mapping_dict = build_mapping(lq_to_phys)

    # ── Local search helpers ───────────────────────────────────────────────
    active_lqs = [lq for lq in logical_qubits if lq in interaction_weight]
    active_set = set(active_lqs)
    n_active = len(active_lqs)

    def total_cost(m):
        cost = 0.0
        for lq1 in active_lqs:
            for lq2, w in interaction_weight[lq1].items():
                if lq2 > lq1:
                    d = self.distance_matrix[m[lq1]][m[lq2]]
                    if d != float('inf'):
                        cost += w * d
        return cost

    def delta_swap(lq_a, lq_b, m):
        p_a, p_b = m[lq_a], m[lq_b]
        delta = 0.0
        for lq_c, w in interaction_weight[lq_a].items():
            if lq_c == lq_b:
                continue
            p_c = m[lq_c]
            d_old = self.distance_matrix[p_a][p_c]
            d_new = self.distance_matrix[p_b][p_c]
            if d_old != float('inf') and d_new != float('inf'):
                delta += w * (d_new - d_old)
        for lq_c, w in interaction_weight[lq_b].items():
            if lq_c == lq_a:
                continue
            p_c = m[lq_c]
            d_old = self.distance_matrix[p_b][p_c]
            d_new = self.distance_matrix[p_a][p_c]
            if d_old != float('inf') and d_new != float('inf'):
                delta += w * (d_new - d_old)
        return delta

    def apply_swap(lq_a, lq_b, m, rm):
        p_a, p_b = m[lq_a], m[lq_b]
        m[lq_a], m[lq_b] = p_b, p_a
        rm[p_a], rm[p_b] = lq_b, lq_a

    def run_2opt(m, rm, max_passes=10):
        """First-improvement 2-opt: apply first found improving swap per pass."""
        for _ in range(max_passes):
            improved = False
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    if delta_swap(active_lqs[i], active_lqs[j], m) < -1e-9:
                        apply_swap(active_lqs[i], active_lqs[j], m, rm)
                        improved = True
            if not improved:
                break
        return m, rm

    # Initial 2-opt pass
    mapping_dict, reverse_mapping_dict = run_2opt(mapping_dict, reverse_mapping_dict)
    best_cost = total_cost(mapping_dict)
    best_mapping = mapping_dict[:]
    best_reverse = reverse_mapping_dict[:]

    # ── Perturbation restarts ──────────────────────────────────────────────
    def get_conflict_pairs(m):
        pairs = []
        for lq1 in active_lqs:
            for lq2, w in interaction_weight[lq1].items():
                if lq2 > lq1 and lq2 in active_set:
                    d = self.distance_matrix[m[lq1]][m[lq2]]
                    pairs.append((w * (d if d != float('inf') else 1e9), lq1, lq2))
        pairs.sort(reverse=True)
        return pairs

    # Adaptive restart budget: more restarts for smaller active sets (cheaper)
    n_restarts = min(20, max(5, 60 // max(1, n_active // 10)))

    for r in range(n_restarts):
        m_new = best_mapping[:]
        rm_new = best_reverse[:]

        pairs = get_conflict_pairs(m_new)
        if not pairs:
            break

        # Rotate perturbation window each restart to diversify targets
        n_pool = min(len(pairs), max(6, n_active // 4))
        shift = (r * 2) % n_pool
        rotated = pairs[shift:n_pool] + pairs[:shift]

        # 2, 3, or 4 perturbation swaps (cycle through magnitudes)
        n_swaps = 2 + (r % 3)
        done, swapped = set(), 0
        for _, q1, q2 in rotated:
            if swapped >= n_swaps:
                break
            if q1 not in done and q2 not in done:
                apply_swap(q1, q2, m_new, rm_new)
                done.add(q1)
                done.add(q2)
                swapped += 1

        if swapped == 0:
            break

        m_new, rm_new = run_2opt(m_new, rm_new)
        c = total_cost(m_new)
        if c < best_cost:
            best_cost = c
            best_mapping = m_new[:]
            best_reverse = rm_new[:]

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)