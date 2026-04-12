def init_mapping(self):
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # -------------------------------------------------------------------
    # Step 1: Build DAG and extract 2-qubit gate info + layers
    # -------------------------------------------------------------------
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)
    access2q_local = {}

    for node in sorted(self.access.keys()):
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]

        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)

        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

        if len(self.access[node]) == 2:
            access2q_local[node] = self.access[node]

    # Extract layers via topological BFS (for simulation)
    all_2q_gates = set(access2q_local.keys())
    # Build 2q-only predecessor counts
    pred_2q = defaultdict(set)
    succ_2q = defaultdict(set)
    for g in all_2q_gates:
        for s in successors.get(g, set()):
            if s in all_2q_gates:
                succ_2q[g].add(s)
                pred_2q[s].add(g)

    # Layer extraction: gates with no 2q predecessors form layer 0, etc.
    layers = []
    in_deg = {g: len(pred_2q[g]) for g in all_2q_gates}
    current_layer = [g for g in all_2q_gates if in_deg[g] == 0]
    remaining_gates = set(all_2q_gates)

    while current_layer:
        layers.append(current_layer)
        next_layer_candidates = set()
        for g in current_layer:
            remaining_gates.discard(g)
            for s in succ_2q[g]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    next_layer_candidates.add(s)
        current_layer = list(next_layer_candidates)

    # -------------------------------------------------------------------
    # Step 2: Build interaction graph with weights
    # -------------------------------------------------------------------
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    total_layers = max(len(layers), 1)
    for li, layer in enumerate(layers):
        decay = 1.0 / (1.0 + li)
        for gate in layer:
            q1, q2 = access2q_local[gate]
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += decay
            logical_degree[q1] += decay
            logical_degree[q2] += decay

    # Also include single-qubit gate qubits
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_set.add(q)

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = sorted(
        [q for q in logical_qubits if logical_degree.get(q, 0) > 0],
        key=lambda q: logical_degree[q], reverse=True
    )

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # -------------------------------------------------------------------
    # Step 3: Lightweight routing simulation cost function
    # -------------------------------------------------------------------
    K_LAYERS = min(5, len(layers))  # simulate first K layers
    sim_layers = layers[:K_LAYERS]
    sim_gates = []
    for layer in sim_layers:
        for g in layer:
            sim_gates.append(access2q_local[g])

    def simulate_swap_cost(m):
        """Count SWAPs needed to route first K layers with greedy nearest-neighbor."""
        if not sim_gates:
            return 0
        # Work on a copy of the mapping
        sim_m = list(m)
        sim_rm = [0] * num_q
        for lq in range(num_q):
            sim_rm[sim_m[lq]] = lq

        swap_count = 0
        for (q1, q2) in sim_gates:
            pq1 = sim_m[q1]
            pq2 = sim_m[q2]
            # Greedily swap toward each other
            while self.distance_matrix[pq1][pq2] > 1:
                # Find neighbor of pq1 closest to pq2
                best_n = None
                best_d = self.distance_matrix[pq1][pq2]
                for n in self.backend[pq1]:
                    d = self.distance_matrix[n][pq2]
                    if d < best_d:
                        best_d = d
                        best_n = n
                if best_n is None:
                    break
                # Perform SWAP between pq1 and best_n
                lq_at_n = sim_rm[best_n]
                lq_at_pq1 = sim_rm[pq1]
                sim_m[lq_at_pq1] = best_n
                sim_m[lq_at_n] = pq1
                sim_rm[pq1] = lq_at_n
                sim_rm[best_n] = lq_at_pq1
                pq1 = best_n
                swap_count += 1
        return swap_count

    # -------------------------------------------------------------------
    # Phase 1: Critical-path backbone anchoring
    # -------------------------------------------------------------------
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    placed = set()
    used_phys = set()

    if interacting_logical and interaction_weight:
        # Find most heavily interacting pair
        best_pair = max(interaction_weight.keys(), key=lambda k: interaction_weight[k])
        lq_a, lq_b = best_pair

        # Find best adjacent physical pair (most central)
        phys_centrality = {}
        for pq in physical_qubits:
            phys_centrality[pq] = sum(
                self.distance_matrix[pq][pq2]
                for pq2 in physical_qubits
                if self.distance_matrix[pq][pq2] < float('inf')
            )

        best_score = float('inf')
        best_pa, best_pb = None, None
        for (pa, pb) in self.backend_connections:
            score = phys_centrality.get(pa, float('inf')) + phys_centrality.get(pb, float('inf'))
            if score < best_score:
                best_score = score
                best_pa, best_pb = pa, pb

        if best_pa is not None:
            mapping_dict[lq_a] = best_pa
            mapping_dict[lq_b] = best_pb
            reverse_mapping_dict[best_pa] = lq_a
            reverse_mapping_dict[best_pb] = lq_b
            placed.add(lq_a)
            placed.add(lq_b)
            used_phys.add(best_pa)
            used_phys.add(best_pb)

    # -------------------------------------------------------------------
    # Phase 2: Simulation-guided greedy placement
    # -------------------------------------------------------------------
    # Order remaining interacting qubits by degree
    remaining_interacting = [q for q in interacting_logical if q not in placed]

    for lq in remaining_interacting:
        # Find candidate physical positions
        free_phys = [pq for pq in physical_qubits if pq not in used_phys]
        if not free_phys:
            break

        # Pre-filter candidates: pick top candidates by distance heuristic first
        # to limit expensive simulation calls
        if len(free_phys) > 20 and placed:
            # Score by weighted distance to placed neighbors
            neighbor_scores = []
            for pq in free_phys:
                score = 0.0
                for plq in placed:
                    w = logical_neighbors[lq].get(plq, 0.0)
                    if w > 0:
                        score += w * self.distance_matrix[pq][mapping_dict[plq]]
                neighbor_scores.append((score, pq))
            neighbor_scores.sort()
            candidates = [pq for _, pq in neighbor_scores[:20]]
        else:
            candidates = free_phys

        best_pq = candidates[0]
        best_swap_count = float('inf')

        for pq in candidates:
            # Tentatively place lq -> pq
            mapping_dict[lq] = pq
            reverse_mapping_dict[pq] = lq

            # Build a full temporary mapping for simulation
            temp_m = list(mapping_dict)
            temp_rm = list(reverse_mapping_dict)
            # Fill unplaced with arbitrary free qubits for simulation
            temp_free = [p for p in range(num_q) if temp_rm[p] == -1 and p != pq]
            temp_unplaced = [q for q in range(num_q) if temp_m[q] == -1]
            for uq, fp in zip(temp_unplaced, temp_free):
                temp_m[uq] = fp
                temp_rm[fp] = uq

            cost = simulate_swap_cost(temp_m)
            if cost < best_swap_count:
                best_swap_count = cost
                best_pq = pq

            # Undo tentative placement
            mapping_dict[lq] = -1
            reverse_mapping_dict[pq] = -1

        # Commit best placement
        mapping_dict[lq] = best_pq
        reverse_mapping_dict[best_pq] = lq
        placed.add(lq)
        used_phys.add(best_pq)

    # Place remaining non-interacting qubits
    unmapped = [q for q in range(num_q) if mapping_dict[q] == -1]
    free = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    for lq, pq in zip(unmapped, free):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # -------------------------------------------------------------------
    # Phase 3: 2-opt refinement using layer-simulation cost
    # -------------------------------------------------------------------
    if sim_gates and len(interacting_logical) > 1:
        current_cost = simulate_swap_cost(mapping_dict)
        improved = True
        max_rounds = 3
        round_count = 0

        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            for i in range(len(interacting_logical)):
                for j in range(i + 1, len(interacting_logical)):
                    lq_a = interacting_logical[i]
                    lq_b = interacting_logical[j]
                    pq_a = mapping_dict[lq_a]
                    pq_b = mapping_dict[lq_b]

                    # Swap
                    mapping_dict[lq_a] = pq_b
                    mapping_dict[lq_b] = pq_a
                    reverse_mapping_dict[pq_a] = lq_b
                    reverse_mapping_dict[pq_b] = lq_a

                    new_cost = simulate_swap_cost(mapping_dict)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                    else:
                        # Undo swap
                        mapping_dict[lq_a] = pq_a
                        mapping_dict[lq_b] = pq_b
                        reverse_mapping_dict[pq_a] = lq_a
                        reverse_mapping_dict[pq_b] = lq_b

    # -------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------
    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)