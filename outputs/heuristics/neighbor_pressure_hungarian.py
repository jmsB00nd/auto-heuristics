def init_mapping(self):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    num_p = self.num_qubits
    dist = self.distance_matrix
    qig = self.qubit_interaction_graph

    logical_qubits = sorted(set(qig.keys()))
    num_l = len(logical_qubits)

    if num_l == 0:
        self.mapping_dict = list(range(num_p))
        self.reverse_mapping_dict = list(range(num_p))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    log_idx = {q: i for i, q in enumerate(logical_qubits)}
    num_phys = len(dist)
    phys_qubits = list(range(num_phys))

    dim = max(num_l, num_phys)

    def build_base_cost():
        cost = np.zeros((dim, dim), dtype=np.float64)
        for i, lq in enumerate(logical_qubits):
            for j, pq in enumerate(phys_qubits):
                total = 0.0
                for neighbor, weight in qig[lq].items():
                    if neighbor in log_idx:
                        for j2, pq2 in enumerate(phys_qubits):
                            total += weight * dist[pq][pq2] * (1.0 / num_phys)
                cost[i][j] = total
        return cost

    base_cost = np.zeros((dim, dim), dtype=np.float64)
    for i, lq in enumerate(logical_qubits):
        for j, pq in enumerate(phys_qubits):
            w_sum = sum(qig[lq].values())
            base_cost[i][j] = w_sum * sum(dist[pq]) / num_phys

    row_ind, col_ind = linear_sum_assignment(base_cost)
    prev_assignment = dict(zip(row_ind[:num_l], col_ind[:num_l]))

    max_iter = 20
    for iteration in range(max_iter):
        cost = np.full((dim, dim), 1e9, dtype=np.float64)

        for i, lq in enumerate(logical_qubits):
            for j, pq in enumerate(phys_qubits):
                pressure = 0.0
                for neighbor, weight in qig[lq].items():
                    if neighbor in log_idx:
                        ni = log_idx[neighbor]
                        assigned_pq = prev_assignment.get(ni, 0)
                        if assigned_pq < num_phys and pq < num_phys:
                            pressure += weight * dist[pq][assigned_pq]
                cost[i][j] = pressure

        for i in range(num_l, dim):
            for j in range(dim):
                cost[i][j] = 0.0
        for i in range(dim):
            for j in range(num_phys, dim):
                if i >= num_l:
                    cost[i][j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)
        new_assignment = dict(zip(row_ind[:num_l], col_ind[:num_l]))

        if new_assignment == prev_assignment:
            break
        prev_assignment = new_assignment

    self.mapping_dict = list(range(num_p))
    self.reverse_mapping_dict = list(range(num_p))

    used_physical = set()
    for i, lq in enumerate(logical_qubits):
        pq = prev_assignment.get(i, None)
        if pq is not None and pq < num_phys:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)

    free_physical = [p for p in range(num_p) if p not in used_physical]
    unmapped_logical = [q for q in range(num_p) if q not in logical_qubits]
    for lq, pq in zip(unmapped_logical, free_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)