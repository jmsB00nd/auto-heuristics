def init_mapping(self):
    """
    Iterative Proportional Fitting Placement

    Constructs a doubly-stochastic affinity matrix between logical and physical
    qubits using Sinkhorn-Knopp normalization on a raw compatibility matrix.
    The compatibility matrix encodes how well each logical-physical pair fits
    based on interaction patterns and hardware distances. The converged
    doubly-stochastic matrix is then rounded to a permutation via Hungarian.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # -------------------------------------------------------------- #
    # 1. Collect logical qubits and build interaction weights         #
    # -------------------------------------------------------------- #
    logical_qubit_set = set()
    interaction_weight = defaultdict(float)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)
    lq_idx = {q: i for i, q in enumerate(logical_qubits)}
    pq_idx = {q: i for i, q in enumerate(physical_qubits)}

    # -------------------------------------------------------------- #
    # 2. Build logical interaction matrix W (n_lq x n_lq)            #
    # -------------------------------------------------------------- #
    W = np.zeros((n_lq, n_lq))
    for (q1, q2), w in interaction_weight.items():
        i, j = lq_idx[q1], lq_idx[q2]
        W[i][j] = w
        W[j][i] = w

    # -------------------------------------------------------------- #
    # 3. Build hardware distance matrix D (n_pq x n_pq)              #
    # -------------------------------------------------------------- #
    D = np.zeros((n_pq, n_pq))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i][j] = self.distance_matrix[p1][p2]

    # -------------------------------------------------------------- #
    # 4. Build raw compatibility matrix C (n_lq x n_pq)              #
    #                                                                  #
    #    C[i,j] = affinity of placing logical qubit i on physical j.  #
    #    Combines interaction-degree * centrality with a pairwise     #
    #    refinement based on top-K proximity matching.                 #
    # -------------------------------------------------------------- #

    # Proximity: convert distances to affinities via exp(-alpha * d)
    D_max = D.max()
    alpha = 1.0 / D_max if D_max > 0 else 1.0
    proximity = np.exp(-alpha * D)  # n_pq x n_pq

    # Interaction degree per logical qubit
    W_row = W.sum(axis=1)  # n_lq

    # Mean proximity (centrality proxy) per physical qubit
    mean_prox = proximity.mean(axis=1)  # n_pq

    # Base compatibility: interaction degree * centrality
    C = np.outer(W_row, mean_prox)  # n_lq x n_pq

    # Pairwise refinement: for each logical qubit with K partners,
    # add bonus proportional to how well physical qubit j's top-K
    # nearest neighbors can host those partners
    for i in range(n_lq):
        partners = np.where(W[i] > 0)[0]
        n_partners = len(partners)
        if n_partners == 0:
            continue
        total_partner_weight = W[i, partners].sum()
        for j in range(n_pq):
            top_k = min(n_partners, n_pq)
            prox_sorted = np.partition(proximity[j], -top_k)[-top_k:]
            C[i, j] += total_partner_weight * prox_sorted.mean()

    C = np.clip(C, 1e-12, None)

    # -------------------------------------------------------------- #
    # 5. Pad to square matrix (n_pq x n_pq) if needed                #
    # -------------------------------------------------------------- #
    if n_lq < n_pq:
        pad = np.full((n_pq - n_lq, n_pq), C.mean() * 0.1)
        C_sq = np.vstack([C, pad])
    elif n_lq == n_pq:
        C_sq = C.copy()
    else:
        C_sq = C[:n_pq, :]

    # -------------------------------------------------------------- #
    # 6. Sinkhorn-Knopp: iterative proportional fitting               #
    #    Alternately normalize rows and columns until convergence     #
    #    to a doubly-stochastic matrix                                #
    # -------------------------------------------------------------- #
    M = C_sq.copy()
    M = np.clip(M, 1e-12, None)

    for iteration in range(200):
        M = M / M.sum(axis=1, keepdims=True)
        M = M / M.sum(axis=0, keepdims=True)
        M = np.clip(M, 1e-12, None)

        if iteration % 20 == 19:
            row_err = np.max(np.abs(M.sum(axis=1) - 1.0))
            col_err = np.max(np.abs(M.sum(axis=0) - 1.0))
            if row_err < 1e-8 and col_err < 1e-8:
                break

    # -------------------------------------------------------------- #
    # 7. Round to permutation via Hungarian (maximize affinity)       #
    # -------------------------------------------------------------- #
    cost_matrix = -M[:n_lq, :]
    if n_lq < n_pq:
        pad_cost = np.zeros((n_pq - n_lq, n_pq))
        cost_sq = np.vstack([cost_matrix, pad_cost])
    else:
        cost_sq = cost_matrix

    row_ind, col_ind = linear_sum_assignment(cost_sq)

    lq_to_phys = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_lq:
            lq_to_phys[logical_qubits[r]] = physical_qubits[c]

    # -------------------------------------------------------------- #
    # 8. Build strict 1-to-1 bijection via in-place swaps             #
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