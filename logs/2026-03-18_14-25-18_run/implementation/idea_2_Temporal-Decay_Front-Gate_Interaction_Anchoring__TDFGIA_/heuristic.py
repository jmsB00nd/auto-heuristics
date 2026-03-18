def init_mapping(self):
    from collections import defaultdict
    import math

    # --- Step 1: Build temporally-biased weighted interaction graph ---
    # τ is set proportional to the number of gates so that the first third of
    # the circuit retains high influence and later gates decay toward zero.
    total_gates = len(self.access)
    tau = max(1.0, total_gates / 3.0)

    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    # Sort gate IDs to obtain a canonical temporal ordering (0, 1, 2, ...)
    # regardless of whether gate IDs are dense or sparse.
    for gate_pos, (_, qubits) in enumerate(sorted(self.access.items())):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            # Temporal decay: gates at position t receive weight exp(-t / τ).
            # Front-layer interactions (small t) dominate the weighted graph,
            # making the anchor placement directly optimise early routing cost.
            decay_weight = math.exp(-gate_pos / tau)
            interaction_weight[key] += decay_weight

    weighted_degree = defaultdict(float)
    interaction_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        weighted_degree[q1] += w
        weighted_degree[q2] += w
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0) + w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback: trivial identity mapping when the circuit contains no gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Anchor logical qubit = highest temporally-weighted degree ---
    # The qubit most heavily involved in early gates is the hardest to route
    # and benefits most from a central hardware placement.
    anchor_logical = max(logical_qubits, key=lambda q: weighted_degree[q])

    # --- Step 3: Most globally-central physical qubit = lowest mean BFS distance ---
    def mean_bfs_dist(p):
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(finite) / len(finite) if finite else float('inf')

    anchor_physical = min(physical_qubits, key=mean_bfs_dist)

    # --- Step 4: Place anchor ---
    lq_to_phys = {anchor_logical: anchor_physical}
    placed_phys = {anchor_physical}
    unplaced = [lq for lq in logical_qubits if lq != anchor_logical]

    # --- Step 5: BFS expansion using temporally-biased interaction weights ---
    while unplaced:
        # Next logical qubit: the unplaced qubit with the highest total
        # temporal-decay interaction weight toward already-placed qubits.
        # This ensures qubits involved in early heavy interactions are
        # co-located with their partners before routing begins.
        next_lq = max(
            unplaced,
            key=lambda lq: sum(
                interaction_neighbors[lq].get(placed_lq, 0)
                for placed_lq in lq_to_phys
            )
        )

        # Candidate physical qubits: unoccupied hardware neighbours of placed qubits
        candidates = list({
            nb
            for phys in placed_phys
            for nb in self.backend[phys]
            if nb not in placed_phys
        })

        # Fallback: all remaining unoccupied physical qubits
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]

        if not candidates:
            break  # Should not occur in a valid circuit/backend pair

        # Select the candidate that minimises total temporally-weighted distance
        # to the physical locations of already-placed interaction partners.
        def placement_cost(phys_c):
            total = 0.0
            for placed_lq, placed_phys_q in lq_to_phys.items():
                w = interaction_neighbors[next_lq].get(placed_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][placed_phys_q]
                    total += w * (d if d != float('inf') else 1e9)
            return total

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # --- Step 6: Isolated logical qubits fill remaining physical qubits
    #             ordered by hardware degree (descending) ---
    remaining_phys = sorted(
        [p for p in physical_qubits if p not in placed_phys],
        key=lambda p: len(self.backend[p]),
        reverse=True
    )
    for lq, phys in zip(unplaced, remaining_phys):
        lq_to_phys[lq] = phys

    # --- Step 7: Build strict 1-to-1 bijection over all num_qubits indices ---
    # Start from the identity permutation and apply TDFGIA assignments via
    # in-place swaps, guaranteeing validity without any search overhead.
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

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)