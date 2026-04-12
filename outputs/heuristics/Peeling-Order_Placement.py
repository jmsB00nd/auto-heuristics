def init_mapping(self):
    """
    Peeling-Order Placement

    Repeatedly peels the lowest-degree vertex from the logical interaction
    graph (k-core decomposition) to determine placement order. The last
    qubits remaining (highest core number) are placed first onto the most
    central hardware qubits. Placement proceeds in reverse peeling order,
    each qubit placed near its already-placed neighbors.
    """
    from collections import defaultdict, deque

    # -------------------------------------------------------------- #
    # 1. Build logical interaction graph from 2-qubit gates           #
    # -------------------------------------------------------------- #
    logical_qubit_set = set()
    interaction = defaultdict(lambda: defaultdict(int))

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            logical_qubit_set.update([q1, q2])
            interaction[q1][q2] += 1
            interaction[q2][q1] += 1

    # Also collect single-qubit-only logical qubits
    for gate, qubits in self.access.items():
        if len(qubits) == 1:
            logical_qubit_set.add(qubits[0])

    logical_qubits = sorted(logical_qubit_set)

    # If no logical qubits or no 2-qubit gates, use trivial mapping
    if not logical_qubits or not interaction:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Build adjacency sets for the interaction graph
    adj = defaultdict(set)
    for q in logical_qubits:
        for nb in interaction[q]:
            adj[q].add(nb)

    # -------------------------------------------------------------- #
    # 2. K-core decomposition (peeling)                               #
    # -------------------------------------------------------------- #
    # Repeatedly remove the minimum-degree vertex, recording the order
    degree = {q: len(adj[q]) for q in logical_qubits}
    remaining = set(logical_qubits)
    peeling_order = []  # first peeled = lowest core number

    while remaining:
        # Find vertex with minimum degree among remaining
        min_q = min(remaining, key=lambda q: degree[q])
        peeling_order.append(min_q)
        remaining.remove(min_q)
        # Update degrees of neighbors
        for nb in adj[min_q]:
            if nb in remaining:
                degree[nb] -= 1

    # Reverse peeling order: most interconnected (highest core) first
    placement_order = list(reversed(peeling_order))

    # -------------------------------------------------------------- #
    # 3. Compute hardware centrality (closeness)                      #
    # -------------------------------------------------------------- #
    # Centrality = sum of distances to all other physical qubits
    # Lower sum = more central
    phys_nodes = sorted(self.backend.keys())
    centrality = {}
    for p in phys_nodes:
        centrality[p] = sum(
            self.distance_matrix[p][p2]
            for p2 in phys_nodes
            if self.distance_matrix[p][p2] < float('inf')
        )

    # Sort physical qubits by centrality (most central first)
    phys_by_centrality = sorted(phys_nodes, key=lambda p: centrality[p])

    # -------------------------------------------------------------- #
    # 4. Place qubits in reverse peeling order                        #
    # -------------------------------------------------------------- #
    lq_to_phys = {}
    used_phys = set()

    for lq in placement_order:
        neighbors_placed = [
            nb for nb in adj[lq] if nb in lq_to_phys
        ]

        if not neighbors_placed:
            # No placed neighbors yet — pick most central available physical qubit
            for p in phys_by_centrality:
                if p not in used_phys:
                    best_phys = p
                    break
        else:
            # Score each available physical qubit by weighted distance
            # to already-placed neighbors (weighted by interaction count)
            best_phys = None
            best_score = float('inf')
            for p in phys_nodes:
                if p in used_phys:
                    continue
                score = sum(
                    interaction[lq][nb] * self.distance_matrix[p][lq_to_phys[nb]]
                    for nb in neighbors_placed
                )
                if score < best_score:
                    best_score = score
                    best_phys = p

        lq_to_phys[lq] = best_phys
        used_phys.add(best_phys)

    # -------------------------------------------------------------- #
    # 5. Build strict 1-to-1 bijection via in-place swap              #
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