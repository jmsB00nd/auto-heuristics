def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n = self.num_qubits
    num_physical = len(self.distance_matrix)

    temporal_weights = np.zeros(n)
    for i in range(n):
        if i in self.temporal_interaction_graph:
            temporal_weights[i] = sum(self.temporal_interaction_graph[i].values())

    dist_sums = np.zeros(n)
    for p in range(num_physical):
        dist_sums[p] = sum(self.distance_matrix[p])
    if num_physical < n:
        pad_val = max(dist_sums[:num_physical]) if num_physical > 0 else 0.0
        for p in range(num_physical, n):
            dist_sums[p] = pad_val

    cost_matrix = np.outer(temporal_weights, dist_sums)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    self.mapping_dict = [0] * n
    self.reverse_mapping_dict = [0] * n
    for i, p in zip(row_ind, col_ind):
        self.mapping_dict[i] = p
        self.reverse_mapping_dict[p] = i

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)