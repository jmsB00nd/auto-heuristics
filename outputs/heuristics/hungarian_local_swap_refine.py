def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n = self.num_qubits
    dm_size = len(self.distance_matrix)

    logical_qubits = set()
    for q1 in self.qubit_interaction_graph:
        logical_qubits.add(q1)
        for q2 in self.qubit_interaction_graph[q1]:
            logical_qubits.add(q2)

    if not logical_qubits:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    def safe_dist(p1, p2):
        if p1 < dm_size and p2 < dm_size:
            return self.distance_matrix[p1][p2]
        return dm_size

    sorted_logical = sorted(logical_qubits,
                            key=lambda q: self.logical_activity.get(q, 0),
                            reverse=True)
    sorted_physical = sorted(range(n),
                             key=lambda p: self.physical_centrality.get(p, 0),
                             reverse=True)

    best_p = {}
    for i, lq in enumerate(sorted_logical):
        best_p[lq] = sorted_physical[i]

    cost_matrix = np.zeros((n, n))
    for l in range(n):
        neighbors = self.qubit_interaction_graph.get(l, {})
        if not neighbors:
            continue
        for p in range(n):
            cost = 0.0
            for l_prime, weight in neighbors.items():
                p_prime = best_p.get(l_prime, l_prime)
                cost += weight * safe_dist(p, p_prime)
            cost_matrix[l][p] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = list(range(n))
    reverse_mapping = list(range(n))
    for l, p in zip(row_ind, col_ind):
        mapping[l] = p
        reverse_mapping[p] = l

    interaction_pairs = []
    seen = set()
    for q1 in self.qubit_interaction_graph:
        for q2, weight in self.qubit_interaction_graph[q1].items():
            if (q1, q2) not in seen:
                seen.add((q1, q2))
                seen.add((q2, q1))
                interaction_pairs.append((q1, q2, weight))

    def total_cost():
        c = 0.0
        for q1, q2, w in interaction_pairs:
            c += w * safe_dist(mapping[q1], mapping[q2])
        return c

    def swap_delta(i, j):
        delta = 0.0
        pi, pj = mapping[i], mapping[j]
        for q1, q2, w in interaction_pairs:
            if q1 != i and q1 != j and q2 != i and q2 != j:
                continue
            old_d = safe_dist(mapping[q1], mapping[q2])
            m1 = pj if q1 == i else (pi if q1 == j else mapping[q1])
            m2 = pj if q2 == i else (pi if q2 == j else mapping[q2])
            new_d = safe_dist(m1, m2)
            delta += w * (new_d - old_d)
        return delta

    active_qubits = sorted(logical_qubits)
    all_qubits = list(range(n))

    improved = True
    while improved:
        improved = False
        for idx_a in range(len(all_qubits)):
            for idx_b in range(idx_a + 1, len(all_qubits)):
                i, j = all_qubits[idx_a], all_qubits[idx_b]
                d = swap_delta(i, j)
                if d < -1e-12:
                    pi, pj = mapping[i], mapping[j]
                    mapping[i], mapping[j] = pj, pi
                    reverse_mapping[pi], reverse_mapping[pj] = j, i
                    improved = True

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)