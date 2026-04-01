def init_mapping(self):
    """
    Qubit Influence Propagation Placement

    Defines an "influence score" for each logical qubit based on how many
    other qubits it transitively interacts with through the DAG, weighted
    by proximity in the DAG. Places qubits in decreasing order of influence
    score, using an influence-aware position selection that considers not
    just current neighbors but the propagation of future routing pressure.
    """
    from collections import defaultdict, deque

    # -------------------------------------------------------------- #
    # 1. Build logical interaction graph and DAG structure             #
    # -------------------------------------------------------------- #
    logical_qubit_set = set()
    interaction = defaultdict(lambda: defaultdict(int))
    gate_list_2q = []

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            logical_qubit_set.update([q1, q2])
            interaction[q1][q2] += 1
            interaction[q2][q1] += 1
            gate_list_2q.append((gate, q1, q2))
        elif len(qubits) == 1:
            logical_qubit_set.add(qubits[0])

    logical_qubits = sorted(logical_qubit_set)

    # Trivial fallback
    if not logical_qubits or not gate_list_2q:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    gate_list_2q.sort(key=lambda x: x[0])

    adj = defaultdict(set)
    for q in logical_qubits:
        for nb in interaction[q]:
            adj[q].add(nb)

    # -------------------------------------------------------------- #
    # 2. Compute influence scores via BFS on interaction graph         #
    #    weighted by DAG proximity                                     #
    # -------------------------------------------------------------- #
    gate_count = defaultdict(int)
    for _, q1, q2 in gate_list_2q:
        gate_count[q1] += 1
        gate_count[q2] += 1

    influence = {}
    for q in logical_qubits:
        score = 0.0
        visited = {q}
        queue = deque([(q, 0)])
        while queue:
            curr, dist = queue.popleft()
            if dist > 0:
                score += interaction[q].get(curr, 0) / dist
                if curr not in interaction[q]:
                    score += gate_count[curr] / (dist * dist)
            for nb in adj[curr]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        influence[q] = score * (1.0 + gate_count[q])

    placement_order = sorted(logical_qubits, key=lambda q: -influence[q])

    # -------------------------------------------------------------- #
    # 3. Compute hardware centrality (closeness)                       #
    # -------------------------------------------------------------- #
    phys_nodes = sorted(self.backend.keys())
    centrality = {}
    for p in phys_nodes:
        centrality[p] = sum(
            self.distance_matrix[p][p2]
            for p2 in phys_nodes
            if self.distance_matrix[p][p2] < float('inf')
        )

    connectivity = {p: len(self.backend[p]) for p in phys_nodes}
    phys_by_centrality = sorted(
        phys_nodes,
        key=lambda p: (centrality[p], -connectivity[p])
    )

    # -------------------------------------------------------------- #
    # 4. Influence-aware placement                                     #
    # -------------------------------------------------------------- #
    lq_to_phys = {}
    used_phys = set()
    unplaced = set(logical_qubits)

    for lq in placement_order:
        unplaced.discard(lq)
        neighbors_placed = [nb for nb in adj[lq] if nb in lq_to_phys]
        neighbors_unplaced = [nb for nb in adj[lq] if nb in unplaced]

        if not neighbors_placed:
            for p in phys_by_centrality:
                if p not in used_phys:
                    best_phys = p
                    break
        else:
            best_phys = None
            best_score = float('inf')

            future_pressure = defaultdict(float)
            if neighbors_unplaced:
                for unp_nb in neighbors_unplaced:
                    unp_placed_nbs = [
                        nb2 for nb2 in adj[unp_nb]
                        if nb2 in lq_to_phys and nb2 != lq
                    ]
                    if unp_placed_nbs:
                        for nb2 in unp_placed_nbs:
                            pp = lq_to_phys[nb2]
                            weight = interaction[lq][unp_nb] * interaction[unp_nb][nb2]
                            future_pressure[pp] += weight

            for p in phys_nodes:
                if p in used_phys:
                    continue

                direct_cost = sum(
                    interaction[lq][nb] * self.distance_matrix[p][lq_to_phys[nb]]
                    for nb in neighbors_placed
                )

                pressure_cost = 0.0
                if future_pressure:
                    for pp, weight in future_pressure.items():
                        pressure_cost += weight * self.distance_matrix[p][pp]

                unplaced_influence = sum(
                    influence.get(nb, 0) for nb in neighbors_unplaced
                )
                alpha = 0.3 if unplaced_influence > 0 else 0.0

                score = direct_cost + alpha * pressure_cost

                if score < best_score or (score == best_score and centrality[p] < centrality.get(best_phys, float('inf'))):
                    best_score = score
                    best_phys = p

        lq_to_phys[lq] = best_phys
        used_phys.add(best_phys)

    # -------------------------------------------------------------- #
    # 5. Build strict 1-to-1 bijection via in-place swap               #
    # -------------------------------------------------------------- #
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