def init_mapping(self):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n = self.num_qubits

    # Step 1: Build interaction matrix W from 2-qubit gates
    W = [[0] * n for _ in range(n)]
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            W[q1][q2] += 1
            W[q2][q1] += 1

    # Collect logical qubits that participate in 2-qubit interactions
    interacting_logical = set()
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                interacting_logical.add(i)
                interacting_logical.add(j)

    # Step 2: Initialize with trivial mapping
    mapping = list(range(n))  # mapping[logical] = physical

    # Step 3: Iterative min-cost bipartite matching
    T = 20  # max iterations
    dist = self.distance_matrix

    for _ in range(T):
        # Compute cost matrix C[i][p] = sum_j W[i][j] * dist(p, mapping[j])
        C = np.zeros((n, n))
        for i in range(n):
            for p in range(n):
                cost = 0.0
                for j in range(n):
                    if W[i][j] > 0:
                        cost += W[i][j] * dist[p][mapping[j]]
                C[i][p] = cost

        # Solve min-cost bipartite matching
        row_ind, col_ind = linear_sum_assignment(C)

        new_mapping = list(range(n))
        for i, p in zip(row_ind, col_ind):
            new_mapping[i] = p

        # Check convergence
        if new_mapping == mapping:
            break

        mapping = new_mapping

    # Populate results
    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * n
    for logical, physical in enumerate(mapping):
        self.reverse_mapping_dict[physical] = logical

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)