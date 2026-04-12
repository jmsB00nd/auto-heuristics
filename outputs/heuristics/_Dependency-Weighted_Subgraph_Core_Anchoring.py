def init_mapping(self):
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    # --- Step 0: Identity baseline ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    # --- Step 1: Build interaction graph ---
    logical_qubit_set = set()
    interaction_weight = defaultdict(float)
    weighted_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0
            weighted_degree[q1] += 1.0
            weighted_degree[q2] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n_logical = len(logical_qubits)

    if n_logical <= 1:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Build adjacency for logical qubits
    logical_neighbors = defaultdict(dict)  # logical_neighbors[q1][q2] = weight
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # --- Step 2: Identify core (top-K by weighted degree, K ~ 60%) ---
    K = max(2, int(0.6 * n_logical))
    sorted_by_degree = sorted(logical_qubits, key=lambda q: weighted_degree.get(q, 0), reverse=True)
    core_logical = sorted_by_degree[:K]
    core_logical_set = set(core_logical)
    non_core_logical = [q for q in logical_qubits if q not in core_logical_set]

    # --- Step 3: Find best K-node hardware subgraph (minimum diameter) ---
    n_phys = len(physical_qubits)
    dist = self.distance_matrix

    # For each physical qubit, take K nearest by distance, score by max pairwise distance
    best_hw_core = None
    best_hw_score = float('inf')

    # Sample centers to keep it tractable
    max_centers = min(n_phys, 50)
    # Pick centers with highest connectivity first
    center_candidates = sorted(physical_qubits, key=lambda p: len(self.backend[p]), reverse=True)[:max_centers]

    for center in center_candidates:
        # Get K nearest physical qubits to this center
        neighbors_sorted = sorted(physical_qubits, key=lambda p: dist[center][p])
        candidate_set = neighbors_sorted[:K]

        # Score: max pairwise distance (diameter)
        max_dist = 0
        for i in range(len(candidate_set)):
            for j in range(i + 1, len(candidate_set)):
                d = dist[candidate_set[i]][candidate_set[j]]
                if d > max_dist:
                    max_dist = d
                    if max_dist >= best_hw_score:
                        break
            if max_dist >= best_hw_score:
                break

        if max_dist < best_hw_score:
            best_hw_score = max_dist
            best_hw_core = candidate_set

    hw_core = best_hw_core
    hw_core_set = set(hw_core)

    # --- Step 4: Hungarian on K×K core assignment (2-pass bootstrap) ---
    for pass_num in range(2):
        cost_matrix = []
        for i, lq in enumerate(core_logical):
            row = []
            for j, pq in enumerate(hw_core):
                cost = 0.0
                for neighbor_lq, w in logical_neighbors.get(lq, {}).items():
                    if neighbor_lq in core_logical_set:
                        if pass_num == 0:
                            # First pass: use distance to closest hw_core qubit
                            min_d = min(dist[pq][hp] for hp in hw_core)
                            cost += w * min_d
                        else:
                            # Second pass: use distance to the assigned physical qubit of neighbor
                            assigned_pq = core_assignment.get(neighbor_lq, None)
                            if assigned_pq is not None:
                                cost += w * dist[pq][assigned_pq]
                            else:
                                min_d = min(dist[pq][hp] for hp in hw_core)
                                cost += w * min_d
                row.append(cost)
            cost_matrix.append(row)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        core_assignment = {}
        for r, c in zip(row_ind, col_ind):
            core_assignment[core_logical[r]] = hw_core[c]

    # --- Step 5: Apply core assignment ---
    occupied = set()
    lq_to_phys = {}
    for lq, pq in core_assignment.items():
        lq_to_phys[lq] = pq
        occupied.add(pq)

    # --- Step 6: Greedy placement for non-core qubits (gravity pull) ---
    for lq in sorted(non_core_logical, key=lambda q: weighted_degree.get(q, 0), reverse=True):
        best_pq = None
        best_score = float('inf')

        for pq in physical_qubits:
            if pq in occupied:
                continue
            score = 0.0
            for neighbor_lq, w in logical_neighbors.get(lq, {}).items():
                if neighbor_lq in lq_to_phys:
                    score += w * dist[pq][lq_to_phys[neighbor_lq]]
            if score < best_score:
                best_score = score
                best_pq = pq

        if best_pq is None:
            # Fallback: pick any unoccupied
            for pq in physical_qubits:
                if pq not in occupied:
                    best_pq = pq
                    break

        lq_to_phys[lq] = best_pq
        occupied.add(best_pq)

    # --- Step 7: Build strict bijection via swaps ---
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