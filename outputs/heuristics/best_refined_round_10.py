def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # ---------------------------------------------------------------
    # Step 1: Build DAG and compute topological rank for temporal decay
    # ---------------------------------------------------------------
    schedule = sorted(self.access.keys())

    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
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

    # Kahn's topological sort
    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_rank = {}
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    # ---------------------------------------------------------------
    # Step 2: Build temporal-decay weighted interaction graph
    # ---------------------------------------------------------------
    alpha = 2.5
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubits_set)

    # ---------------------------------------------------------------
    # Step 3: Seed placement — highest temporal-degree qubit at most
    # central physical qubit (min sum of hop distances)
    # [From Parent 2: direct centrality, no MDS]
    # ---------------------------------------------------------------
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    used_physical = set()
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    if logical_qubits:
        start_lq = max(logical_qubits, key=lambda q: logical_degree.get(q, 0))
        start_pq = min(physical_qubits, key=lambda pq: phys_centrality[pq])
        mapping_dict[start_lq] = start_pq
        reverse_mapping_dict[start_pq] = start_lq
        used_physical.add(start_pq)

        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        # -----------------------------------------------------------
        # Step 4: Greedy placement using actual distance matrix
        # [From Parent 2: direct hop-distance scoring]
        # [From Parent 1: temporal-decay weighted traversal order]
        # -----------------------------------------------------------
        while remaining:
            best_lq = None
            best_weight = -1.0
            for lq in remaining:
                w = sum(logical_neighbors[lq].get(plq, 0.0) for plq in placed)
                if w > best_weight:
                    best_weight = w
                    best_lq = lq

            neighbors_placed = {plq: logical_neighbors[best_lq].get(plq, 0.0)
                                for plq in placed if plq in logical_neighbors[best_lq]}

            if neighbors_placed:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_physical:
                        score = 0.0
                        for plq, iw in neighbors_placed.items():
                            score += iw * self.distance_matrix[pq][mapping_dict[plq]]
                        if score < best_score:
                            best_score = score
                            best_pq = pq
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_physical:
                        score = sum(self.distance_matrix[pq][pq2]
                                    for pq2 in physical_qubits if pq2 not in used_physical)
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            mapping_dict[best_lq] = best_pq
            reverse_mapping_dict[best_pq] = best_lq
            used_physical.add(best_pq)
            placed.add(best_lq)
            remaining.discard(best_lq)

    # ---------------------------------------------------------------
    # Step 5: Fill remaining unmapped qubits
    # ---------------------------------------------------------------
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]

    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # ---------------------------------------------------------------
    # Step 6: Post-placement pairwise swap refinement
    # CROSSOVER NOVELTY: Iteratively swap physical assignments of
    # logical qubit pairs when it reduces total temporal-decay-weighted
    # interaction distance. Corrects greedy errors using efficient
    # delta-cost computation (only edges incident to swap pair change).
    # ---------------------------------------------------------------
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    if len(interacting_logical) > 1:
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

                    # Delta cost: only edges incident to lq_a or lq_b
                    delta = 0.0
                    affected_qubits = set()
                    affected_qubits.update(logical_neighbors[lq_a].keys())
                    affected_qubits.update(logical_neighbors[lq_b].keys())

                    for q in affected_qubits:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = mapping_dict[q]

                        w_a = logical_neighbors[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])

                        w_b = logical_neighbors[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                    if delta < -1e-12:
                        mapping_dict[lq_a] = pq_b
                        mapping_dict[lq_b] = pq_a
                        reverse_mapping_dict[pq_a] = lq_b
                        reverse_mapping_dict[pq_b] = lq_a
                        improved = True

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)