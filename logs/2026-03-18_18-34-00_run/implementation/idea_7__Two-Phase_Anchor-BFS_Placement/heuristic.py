def init_mapping(self):
    from collections import defaultdict, deque

    # --- Step 1: Build F[i][j]: 2-qubit gate count per logical qubit pair ---
    F = defaultdict(float)
    logical_qubit_set = set()

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            F[key] += 1

    # Build interaction neighbor map and weighted degree for each logical qubit
    interaction_neighbors = defaultdict(dict)
    weighted_degree = defaultdict(float)
    for (q1, q2), w in F.items():
        weighted_degree[q1] += w
        weighted_degree[q2] += w
        interaction_neighbors[q1][q2] = w
        interaction_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n_logical = len(logical_qubits)

    # Fallback: trivial identity mapping if no logical qubits
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Phase 1: Anchor Selection & Placement ---
    # Select top-k anchors by total weighted interaction (k = sqrt(n_logical))
    k = max(1, int(math.sqrt(n_logical)))
    anchors = sorted(logical_qubits, key=lambda q: weighted_degree[q], reverse=True)[:k]

    # Hardware qubit degree (number of edges in backend coupling graph)
    hw_degree = {p: len(self.backend[p]) for p in physical_qubits}
    hw_by_degree = sorted(physical_qubits, key=lambda p: hw_degree[p], reverse=True)

    # Anchor_0 → highest-degree hw qubit; each subsequent anchor → highest-degree hw qubit
    # maximally far (max-min-distance) from all already-placed anchors
    lq_to_phys = {}
    assigned_physical = set()

    for i, anchor_lq in enumerate(anchors):
        if i == 0:
            chosen_phys = hw_by_degree[0]
        else:
            best_phys = None
            best_score = (-1, -1)  # (min_dist_to_placed_anchors, hw_degree)
            for p in hw_by_degree:
                if p in assigned_physical:
                    continue
                min_dist = min(
                    self.distance_matrix[p][lq_to_phys[a]]
                    for a in anchors[:i]
                )
                score = (min_dist, hw_degree[p])
                if score > best_score:
                    best_score = score
                    best_phys = p
            chosen_phys = best_phys

        lq_to_phys[anchor_lq] = chosen_phys
        assigned_physical.add(chosen_phys)

    # --- Phase 2: Satellite Assignment via BFS ---
    unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

    while unplaced:
        # Select satellite with highest max-interaction to any already-placed qubit
        next_lq = max(
            unplaced,
            key=lambda lq: max(
                (interaction_neighbors[lq].get(p, 0) for p in lq_to_phys),
                default=0
            )
        )
        unplaced.remove(next_lq)

        # Find best-interacting already-placed neighbor n*
        best_neighbor_lq = max(
            lq_to_phys.keys(),
            key=lambda p: interaction_neighbors[next_lq].get(p, 0)
        )
        anchor_phys = lq_to_phys[best_neighbor_lq]

        # BFS outward from anchor_phys on hardware to find nearest free physical qubit
        visited = {anchor_phys}
        queue = deque([anchor_phys])
        found_phys = None
        while queue:
            curr = queue.popleft()
            if curr not in assigned_physical:
                found_phys = curr
                break
            for nb in self.backend[curr]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        # Fallback: any unassigned physical qubit
        if found_phys is None:
            remaining = [p for p in physical_qubits if p not in assigned_physical]
            found_phys = remaining[0] if remaining else None

        if found_phys is not None:
            lq_to_phys[next_lq] = found_phys
            assigned_physical.add(found_phys)

    # --- Build strict 1-to-1 bijection over all num_qubits indices ---
    # Start from identity, apply assignments via in-place swaps to preserve bijectivity
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