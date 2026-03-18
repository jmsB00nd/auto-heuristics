def init_mapping(self):
    from collections import defaultdict
    import math

    # --- Step 1: Build TEMPORALLY-WEIGHTED interaction graph ---
    # Gates earlier in the circuit receive exponentially higher weight.
    # Rationale: the initial mapping affects early gates most critically;
    # early 2-qubit gates that must run first benefit most from co-location.
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    gate_list = list(self.access.items())
    total_gates = len(gate_list)

    for idx, (gate, qubits) in enumerate(gate_list):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            # Base count + temporal bonus: exp decay from idx=0 (highest) to idx=T-1 (lowest)
            t_norm = idx / max(total_gates - 1, 1)
            interaction_weight[key] += 1.0 + math.exp(-3.0 * t_norm)

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

    # --- Step 2: Precompute mean BFS distance for physical centrality ---
    mean_dist_cache = {}
    for p in physical_qubits:
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        mean_dist_cache[p] = sum(finite) / len(finite) if finite else float('inf')

    # --- Step 3: Anchor selection ---
    # Logical anchor: highest temporally-weighted degree (most critical qubit)
    anchor_logical = max(logical_qubits, key=lambda q: weighted_degree[q])
    # Physical anchor: most central (lowest mean BFS distance)
    anchor_physical = min(physical_qubits, key=lambda p: mean_dist_cache[p])

    # --- Step 4: Greedy BFS placement with tie-breaking ---
    lq_to_phys = {anchor_logical: anchor_physical}
    placed_phys = {anchor_physical}
    unplaced = [lq for lq in logical_qubits if lq != anchor_logical]

    while unplaced:
        # Select next logical: max interaction weight with placed qubits;
        # tie-break by total weighted degree (most connected overall first)
        next_lq = max(
            unplaced,
            key=lambda lq: (
                sum(interaction_neighbors[lq].get(p_lq, 0) for p_lq in lq_to_phys),
                weighted_degree[lq]
            )
        )

        # Candidate physical qubits: unoccupied hardware neighbors of placed qubits
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

        # Placement cost = weighted distance to already-placed interaction partners
        # + lookahead penalty: peripherality cost for top-3 unplaced neighbors
        top_unplaced_neighbors = sorted(
            [(interaction_neighbors[next_lq].get(ulq, 0), ulq)
             for ulq in unplaced if ulq != next_lq],
            reverse=True
        )[:3]

        def placement_cost(phys_c, _next=next_lq, _top=top_unplaced_neighbors):
            cost = 0.0
            for placed_lq, placed_phys_q in lq_to_phys.items():
                w = interaction_neighbors[_next].get(placed_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][placed_phys_q]
                    cost += w * (d if d != float('inf') else 1e9)
            # Lookahead: penalize peripheral placements for future strongly-coupled qubits
            for w_future, _ in _top:
                if w_future > 0:
                    cost += 0.25 * w_future * mean_dist_cache[phys_c]
            return cost

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # Remaining isolated qubits: fill by hardware degree descending
    remaining_phys = sorted(
        [p for p in physical_qubits if p not in placed_phys],
        key=lambda p: len(self.backend[p]),
        reverse=True
    )
    isolated = [lq for lq in logical_qubits if lq not in lq_to_phys]
    for lq, phys in zip(isolated, remaining_phys):
        lq_to_phys[lq] = phys

    # --- Step 5: Build strict 1-to-1 bijection via in-place swaps ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    # --- Step 6: LOCAL SWAP REFINEMENT ---
    # Iteratively swap physical assignments of any two logical qubits when it
    # strictly reduces total weighted routing distance. This escapes greedy-order
    # artifacts and finds a local optimum of the placement.
    #
    # delta_swap(lq_a, lq_b): change in sum_{(i,j)} w_ij * dist(phys_i, phys_j)
    # when lq_a (at pa) and lq_b (at pb) exchange physical positions.
    # The (lq_a, lq_b) edge distance is invariant under this swap (dist is symmetric).

    active_lqs = sorted(lq_to_phys.keys())

    def delta_swap(lq_a, lq_b):
        pa, pb = mapping_dict[lq_a], mapping_dict[lq_b]
        delta = 0.0
        for other_lq in active_lqs:
            if other_lq == lq_a or other_lq == lq_b:
                continue
            op = mapping_dict[other_lq]
            wa = interaction_neighbors[lq_a].get(other_lq, 0)
            wb = interaction_neighbors[lq_b].get(other_lq, 0)
            if wa > 0:
                d_old = self.distance_matrix[pa][op]
                d_new = self.distance_matrix[pb][op]
                delta += wa * (
                    (d_new if d_new != float('inf') else 1e9) -
                    (d_old if d_old != float('inf') else 1e9)
                )
            if wb > 0:
                d_old = self.distance_matrix[pb][op]
                d_new = self.distance_matrix[pa][op]
                delta += wb * (
                    (d_new if d_new != float('inf') else 1e9) -
                    (d_old if d_old != float('inf') else 1e9)
                )
        return delta

    improved = True
    max_rounds = 10
    round_count = 0

    while improved and round_count < max_rounds:
        improved = False
        round_count += 1
        for i in range(len(active_lqs)):
            for j in range(i + 1, len(active_lqs)):
                lq1, lq2 = active_lqs[i], active_lqs[j]
                if delta_swap(lq1, lq2) < -1e-9:
                    p1, p2 = mapping_dict[lq1], mapping_dict[lq2]
                    mapping_dict[lq1] = p2
                    mapping_dict[lq2] = p1
                    reverse_mapping_dict[p1] = lq2
                    reverse_mapping_dict[p2] = lq1
                    improved = True

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)