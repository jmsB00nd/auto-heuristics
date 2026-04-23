def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    n = self.num_qubits

    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    if self.access2q is None:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_qubits_set = set()
    interactions = defaultdict(lambda: defaultdict(float))
    criticality = defaultdict(lambda: defaultdict(float))

    for gate, qubits in self.access2q.items():
        if len(qubits) != 2:
            continue
        q1, q2 = qubits
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        interactions[q1][q2] += 1.0
        interactions[q2][q1] += 1.0
        dep_count = self.dag_dependencies_count.get(gate, 0) if hasattr(self, 'dag_dependencies_count') and self.dag_dependencies_count is not None else 0
        crit = dep_count + 1.0
        criticality[q1][q2] += crit
        criticality[q2][q1] += crit

    if not logical_qubits_set:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_list = sorted(logical_qubits_set)
    num_logical = len(logical_list)
    log_to_idx = {q: i for i, q in enumerate(logical_list)}

    dist = np.array(self.distance_matrix)

    current_phys = list(range(n))

    for _iteration in range(3):
        cost = np.zeros((n, n), dtype=np.float64)

        for i_log in logical_list:
            idx_i = i_log
            for k_log, weight in interactions[i_log].items():
                crit_val = criticality[i_log][k_log]
                phys_k = current_phys[k_log]
                for j_phys in range(n):
                    if j_phys < len(dist) and phys_k < len(dist):
                        cost[idx_i][j_phys] += weight * crit_val * dist[j_phys][phys_k]

        row_ind, col_ind = linear_sum_assignment(cost)

        new_mapping = list(range(n))
        used_physical = set()

        for r, c in zip(row_ind, col_ind):
            new_mapping[r] = c
            used_physical.add(c)

        current_phys = new_mapping[:]

    self.mapping_dict = current_phys[:]
    self.reverse_mapping_dict = [0] * n
    for logical_q in range(n):
        self.reverse_mapping_dict[self.mapping_dict[logical_q]] = logical_q

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)