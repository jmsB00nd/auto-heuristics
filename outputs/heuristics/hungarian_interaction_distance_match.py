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

    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))
    for l, p in zip(row_ind, col_ind):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)