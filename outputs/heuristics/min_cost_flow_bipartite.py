def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    N = self.num_qubits

    # Collect logical qubits involved in 2-qubit gates and build interaction weights
    logical_qubits = set()
    if self.access2q is not None:
        for gate, qubits in self.access2q.items():
            if len(qubits) == 2:
                logical_qubits.update(qubits)
    else:
        for gate, qubits in self.access.items():
            if len(qubits) == 2:
                logical_qubits.update(qubits)

    access2q = self.access2q if self.access2q is not None else self.access

    if not logical_qubits:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    W = np.zeros((N, N), dtype=np.float64)
    for gate, qubits in access2q.items():
        if len(qubits) == 2:
            i, j = qubits[0], qubits[1]
            W[i][j] += 1.0
            W[j][i] += 1.0

    D = np.zeros((N, N), dtype=np.float64)
    for p in range(N):
        for q in range(N):
            if p in self.distance_matrix and q in self.distance_matrix[p]:
                D[p][q] = self.distance_matrix[p][q]
            elif q in self.distance_matrix and p in self.distance_matrix[q]:
                D[p][q] = self.distance_matrix[q][p]

    logical_list = sorted(logical_qubits)
    n_logical = len(logical_list)
    log_idx = {q: idx for idx, q in enumerate(logical_list)}

    # Degree of each logical qubit (sum of interaction weights)
    degree = np.array([W[q].sum() for q in logical_list])

    # Initial cost matrix: degree-weighted centroid approach
    # For each logical qubit j, compute a "centroid" physical location
    # as the physical qubit minimizing sum of D[p, *] weighted by neighbor interactions.
    # First pass: use uniform spread assumption — centroid(j) ≈ argmin_p Σ_k W[j,k] * D[p, k_approx]
    # Since we don't know assignments yet, approximate by weighting D rows by neighbor degrees.

    n_phys = N
    cost = np.zeros((n_logical, n_phys), dtype=np.float64)

    # Initial heuristic: for logical qubit i placed at physical p,
    # cost = Σ_j W[i,j] * (average distance from p to all physical qubits, weighted by degree of j)
    # This simplifies to: cost[i,p] = Σ_j W[i,j] * mean(D[p,:])  which is uniform,
    # so instead use: cost[i,p] = Σ_j W[i,j] * D[p, centroid_j]
    # where centroid_j = physical qubit closest to graph center weighted by j's neighbors' degrees.

    # Better initial heuristic: spectral-like approach
    # Place high-interaction logical qubits near each other on hardware.
    # cost[i,p] = Σ_j W[i,j] * D[p, degree_centroid]
    # degree_centroid = physical qubit minimizing Σ_q D[p,q] (graph center)

    # Compute graph center of hardware
    dist_sum = D.sum(axis=1)
    center = int(np.argmin(dist_sum))

    # First pass: cost proportional to interaction weight * distance from center
    # High-degree logical qubits should be near center
    for idx_i, qi in enumerate(logical_list):
        for p in range(n_phys):
            # Weighted distance: neighbors of qi should ideally be close to p
            # Without assignment info, use distance to center as proxy
            interaction_sum = 0.0
            for idx_j, qj in enumerate(logical_list):
                if W[qi][qj] > 0:
                    interaction_sum += W[qi][qj]
            # High-interaction qubits should be placed centrally
            cost[idx_i, p] = interaction_sum * D[p][center]
            # Add small tie-breaking based on distance spread
            for idx_j, qj in enumerate(logical_list):
                if W[qi][qj] > 0:
                    # Penalize placement far from other high-degree qubits' ideal positions
                    cost[idx_i, p] += W[qi][qj] * dist_sum[p] / n_phys * 0.1

    # Solve initial assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = {}
    for r, c in zip(row_ind, col_ind):
        assignment[logical_list[r]] = c

    # Iterative refinement: 2 more rounds
    for _round in range(2):
        cost = np.zeros((n_logical, n_phys), dtype=np.float64)
        for idx_i, qi in enumerate(logical_list):
            for p in range(n_phys):
                c = 0.0
                for idx_j, qj in enumerate(logical_list):
                    if W[qi][qj] > 0 and qj in assignment:
                        c += W[qi][qj] * D[p][assignment[qj]]
                cost[idx_i, p] = c

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {}
        for r, c in zip(row_ind, col_ind):
            assignment[logical_list[r]] = c

    # Build full mapping
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    used_physical = set(assignment.values())
    available_physical = [p for p in range(N) if p not in used_physical]
    unmapped_logical = [q for q in range(N) if q not in assignment]

    for lq, pq in assignment.items():
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    av_idx = 0
    for lq in unmapped_logical:
        self.mapping_dict[lq] = available_physical[av_idx]
        self.reverse_mapping_dict[available_physical[av_idx]] = lq
        av_idx += 1

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)