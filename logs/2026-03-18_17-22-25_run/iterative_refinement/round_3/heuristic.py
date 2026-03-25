def init_mapping(self):
    """
    Layer-Normalized BFS with Multi-Anchor and Perturbation Restarts (LNBFS-MPR).

    Key differences from TWPSD (Refined_Idea_Round_2):

    1. Layer-normalized exponential decay:
       Interaction weights use exp(-β · layer / max_layer) rather than
       exp(-1/n_gates)^gate_idx. The topological layer of each gate is computed
       from its data dependencies (shared qubits), so the decay is proportional
       to circuit *depth* rather than gate count. A 10-layer circuit and a
       1000-layer circuit now get identical relative weight distributions, instead
       of the fixed per-gate rate treating them wildly differently.

    2. Multi-anchor Phase 1:
       The top-3 highest-weighted-degree logical qubits are each tried as the BFS
       seed anchor. The placement with the lowest total interaction cost is
       forwarded to Phase 2. This costs ~3× Phase 1 time but avoids locking into
       a poor basin determined by a single greedy seed choice.

    3. Perturbation-restart escape:
       After steepest-descent converges, we cycle through N_RESTARTS perturbations:
       each restart shifts the window of highest-conflict qubit pairs (so each
       restart targets a different subset), applies 2–4 forced swaps (increasing
       the cost intentionally to escape the current basin), then re-runs full
       steepest descent. The globally best mapping across all restarts is retained.
       This is the principal addition: single-start descent is known to get trapped
       in shallow local optima on heavy-hex topologies with many near-equivalent
       placements, and restart diversity systematically explores alternate basins.
    """
    import math
    from collections import defaultdict

    gate_list = list(self.access.items())
    n_gates = len(gate_list)

    if not gate_list:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── Compute topological layers (data-dependency DAG) ─────────────────
    # Layer = earliest layer in which a gate can execute, given qubit dependencies.
    last_gate_on_qubit = {}
    gate_layer = []
    for gate_idx, (gate, qubits) in enumerate(gate_list):
        layer = 0
        for q in qubits:
            if q in last_gate_on_qubit:
                layer = max(layer, gate_layer[last_gate_on_qubit[q]] + 1)
        gate_layer.append(layer)
        for q in qubits:
            last_gate_on_qubit[q] = gate_idx

    max_layer = max(gate_layer) if gate_layer else 0

    # ── Layer-normalized interaction graph ────────────────────────────────
    # beta=2.0: layer 0 → weight 1.0; last layer → weight exp(-2) ≈ 0.135.
    # Dividing by (max_layer + 1) keeps weight > 0 even for single-layer circuits.
    beta = 2.0
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    for gate_idx, (gate, qubits) in enumerate(gate_list):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_layer[gate_idx]
            w = math.exp(-beta * layer / (max_layer + 1))
            interaction_weight[key] += w

    weighted_degree = defaultdict(float)
    interaction_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        weighted_degree[q1] += w
        weighted_degree[q2] += w
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0) + w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── Most central physical qubit (minimises mean BFS distance) ─────────
    def phys_mean_dist(p):
        dists = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(dists) / len(dists) if dists else float('inf')

    anchor_physical = min(physical_qubits, key=phys_mean_dist)

    # ── Phase 1: interaction-weighted BFS placement ───────────────────────
    def run_phase1(anchor_l):
        lq_to_phys = {anchor_l: anchor_physical}
        placed_phys = {anchor_physical}
        unplaced = [lq for lq in logical_qubits if lq != anchor_l]

        while unplaced:
            # Logical qubit with the most interaction weight to already-placed set
            next_lq = max(
                unplaced,
                key=lambda lq: sum(
                    interaction_neighbors[lq].get(pl, 0) for pl in lq_to_phys
                )
            )
            # Candidate physical qubits: hardware-adjacent to any placed qubit
            candidates = list({
                nb
                for phys in placed_phys
                for nb in self.backend[phys]
                if nb not in placed_phys
            })
            if not candidates:
                candidates = [p for p in physical_qubits if p not in placed_phys]
            if not candidates:
                break

            def placement_cost(phys_c, lq=next_lq):
                return sum(
                    interaction_neighbors[lq].get(pl, 0) * (
                        self.distance_matrix[phys_c][pp]
                        if self.distance_matrix[phys_c][pp] != float('inf') else 1e9
                    )
                    for pl, pp in lq_to_phys.items()
                )

            best_phys = min(candidates, key=placement_cost)
            lq_to_phys[next_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.remove(next_lq)

        # Non-interacting qubits → high-degree physical qubits (reduce future conflicts)
        rem_lq = [lq for lq in logical_qubits if lq not in lq_to_phys]
        rem_phys = sorted(
            [p for p in physical_qubits if p not in placed_phys],
            key=lambda p: len(self.backend[p]), reverse=True
        )
        for lq, phys in zip(rem_lq, rem_phys):
            lq_to_phys[lq] = phys

        # Build mapping via in-place transpositions on identity
        m = list(range(self.num_qubits))
        rm = list(range(self.num_qubits))
        for lq, tp in lq_to_phys.items():
            cp = m[lq]
            if cp == tp:
                continue
            dlq = rm[tp]
            m[lq] = tp
            m[dlq] = cp
            rm[tp] = lq
            rm[cp] = dlq
        return m, rm

    # ── Multi-anchor: pick Phase-1 result with lowest cost ────────────────
    def total_cost(m):
        return sum(
            w * (
                self.distance_matrix[m[q1]][m[q2]]
                if self.distance_matrix[m[q1]][m[q2]] != float('inf') else 1e9
            )
            for (q1, q2), w in interaction_weight.items()
        )

    top_anchors = sorted(
        logical_qubits, key=lambda q: weighted_degree[q], reverse=True
    )[:3]

    mapping_dict, reverse_mapping_dict = run_phase1(top_anchors[0])
    best_p1_cost = total_cost(mapping_dict)
    for anchor_l in top_anchors[1:]:
        m, rm = run_phase1(anchor_l)
        c = total_cost(m)
        if c < best_p1_cost:
            best_p1_cost = c
            mapping_dict, reverse_mapping_dict = m, rm

    # ── Phase 2: steepest-descent pairwise swap optimisation ──────────────
    active_lqs = [lq for lq in logical_qubits if interaction_neighbors[lq]]
    n_active = len(active_lqs)

    def delta_swap(lq_a, lq_b, m):
        """O(deg(a)+deg(b)) change in cost if lq_a and lq_b exchange positions."""
        p_a, p_b = m[lq_a], m[lq_b]
        delta = 0.0
        for lq_c, w in interaction_neighbors[lq_a].items():
            if lq_c != lq_b:
                p_c = m[lq_c]
                delta += w * (self.distance_matrix[p_b][p_c]
                               - self.distance_matrix[p_a][p_c])
        for lq_c, w in interaction_neighbors[lq_b].items():
            if lq_c != lq_a:
                p_c = m[lq_c]
                delta += w * (self.distance_matrix[p_a][p_c]
                               - self.distance_matrix[p_b][p_c])
        return delta

    def apply_swap(lq_a, lq_b, m, rm):
        p_a, p_b = m[lq_a], m[lq_b]
        m[lq_a], m[lq_b] = p_b, p_a
        rm[p_a], rm[p_b] = lq_b, lq_a

    def steepest_descent(m, rm):
        while True:
            best_d, best_pair = 0.0, None
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    d = delta_swap(active_lqs[i], active_lqs[j], m)
                    if d < best_d:
                        best_d, best_pair = d, (active_lqs[i], active_lqs[j])
            if best_pair is None:
                break
            apply_swap(*best_pair, m, rm)
        return m, rm

    mapping_dict, reverse_mapping_dict = steepest_descent(
        mapping_dict, reverse_mapping_dict
    )
    best_cost = total_cost(mapping_dict)
    best_mapping = mapping_dict[:]
    best_reverse = reverse_mapping_dict[:]

    # ── Phase 3: perturbation restarts ────────────────────────────────────
    # Build sorted conflict pair list from the current best mapping.
    # Each restart shifts the window by 2 pairs so different subsets are
    # perturbed, providing basin diversity without full randomness.
    active_set = set(active_lqs)

    def sorted_conflict_pairs(m):
        pairs = []
        for (q1, q2), w in interaction_weight.items():
            if q1 in active_set and q2 in active_set:
                d = self.distance_matrix[m[q1]][m[q2]]
                if d == float('inf'):
                    d = 1e9
                pairs.append((w * d, q1, q2))
        pairs.sort(reverse=True)
        return pairs

    # Adaptive restart budget: more restarts for smaller active sets (cheaper)
    n_restarts = min(20, max(5, 60 // max(1, n_active // 10)))

    for r in range(n_restarts):
        m_new = best_mapping[:]
        rm_new = best_reverse[:]

        pairs = sorted_conflict_pairs(m_new)
        if not pairs:
            break

        # Shift start index to diversify which pairs get perturbed each restart
        n_pool = min(len(pairs), max(6, n_active // 4))
        shift = (r * 2) % n_pool
        rotated = pairs[shift:n_pool] + pairs[:shift]

        # Apply 2, 3, or 4 perturbation swaps (cycle through 3 magnitudes)
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

        m_new, rm_new = steepest_descent(m_new, rm_new)
        c = total_cost(m_new)
        if c < best_cost:
            best_cost = c
            best_mapping = m_new[:]
            best_reverse = rm_new[:]

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)