def init_mapping(self):
    import numpy as np
    from scipy.sparse.linalg import eigsh
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict, deque
    import random
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix
    d = 5  # embedding dimension

    # ── Step 0: Build DAG for critical-path weights ──
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)
    all_gates = set(self.access.keys())

    for node in sorted(all_gates):
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]
        for q in read_qubits:
            if q in latest_writer:
                pred = latest_writer[q]
                if pred != node:
                    successors[pred].add(node)
                    predecessors[node].add(pred)
        for q in write_qubits:
            if q in latest_writer:
                pred = latest_writer[q]
                if pred != node:
                    successors[pred].add(node)
                    predecessors[node].add(pred)
            for reader in active_readers.get(q, set()):
                if reader != node:
                    successors[reader].add(node)
                    predecessors[node].add(reader)
            active_readers[q] = set()
            latest_writer[q] = node
        for q in read_qubits:
            active_readers[q].add(node)

    # Topological order + critical path length
    in_deg = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_deg[g] == 0))
    topo_order = []
    cp_forward = {}
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        cp_forward[g] = 1 + max((cp_forward.get(p, 0) for p in predecessors.get(g, set())), default=0)
        for s in successors.get(g, set()):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    max_cp = max(cp_forward.values()) if cp_forward else 1

    # ── Step 1: Build critical-path-weighted interaction graph ──
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)
    interacting_logical = set()

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            interacting_logical.add(q1)
            interacting_logical.add(q2)
            cp = cp_forward.get(gate, 1)
            w = (max_cp - cp + 1)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w

    interacting_logical = sorted(interacting_logical)
    n_logical = len(interacting_logical)

    if n_logical == 0:
        # No 2-qubit gates: trivial mapping
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # ── Step 2: Build Interaction Graph Laplacian ──
    log_idx = {q: i for i, q in enumerate(interacting_logical)}
    n = n_logical
    L_I = np.zeros((n, n))
    for (q1, q2), w in interaction_weight.items():
        i, j = log_idx[q1], log_idx[q2]
        L_I[i, j] -= w
        L_I[j, i] -= w
        L_I[i, i] += w
        L_I[j, j] += w

    # ── Step 3: Build Hardware Graph Laplacian ──
    m_hw = len(physical_qubits)
    phys_idx = {pq: i for i, pq in enumerate(physical_qubits)}
    L_H = np.zeros((m_hw, m_hw))
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            if pq < pq2:
                i, j = phys_idx[pq], phys_idx[pq2]
                L_H[i, j] -= 1.0
                L_H[j, i] -= 1.0
                L_H[i, i] += 1.0
                L_H[j, j] += 1.0

    # ── Step 4: Compute spectral embeddings ──
    actual_d = min(d, n - 1, m_hw - 1)
    if actual_d < 1:
        actual_d = 1

    # Use smallest eigenvectors (skip the trivial zero eigenvector)
    k_I = min(actual_d + 1, n)
    try:
        eigvals_I, eigvecs_I = eigsh(L_I, k=k_I, which='SM')
        # Sort by eigenvalue
        order = np.argsort(eigvals_I)
        eigvecs_I = eigvecs_I[:, order]
        # Skip first (constant) eigenvector
        U_I = eigvecs_I[:, 1:actual_d + 1]
    except Exception:
        U_I = np.random.randn(n, actual_d)

    k_H = min(actual_d + 1, m_hw)
    try:
        eigvals_H, eigvecs_H = eigsh(L_H, k=k_H, which='SM')
        order = np.argsort(eigvals_H)
        eigvecs_H = eigvecs_H[:, order]
        U_H = eigvecs_H[:, 1:actual_d + 1]
    except Exception:
        U_H = np.random.randn(m_hw, actual_d)

    # Pad if dimensions don't match
    if U_I.shape[1] < actual_d:
        U_I = np.hstack([U_I, np.zeros((U_I.shape[0], actual_d - U_I.shape[1]))])
    if U_H.shape[1] < actual_d:
        U_H = np.hstack([U_H, np.zeros((U_H.shape[0], actual_d - U_H.shape[1]))])

    # ── Step 5: Orthogonal Procrustes + Hungarian ──
    # Solve R* = argmin ||U_I R - U_H_subset||_F
    # For each possible subset this is intractable, so we use relaxation:
    # R* from SVD of U_I^T @ U_H (using all of U_H averaged)
    # Then compute pairwise distances and solve Hungarian

    # Compute R via SVD of U_I^T @ (mean-centered U_H projected)
    # Use the Procrustes approach: for the n x m assignment, compute
    # cost matrix after optimal rotation
    # R* = V @ U^T from SVD of U_I^T @ U_H_candidate
    # Since we don't know the subset, we compute R from the full cross-matrix

    # Build n x m cost matrix
    # For each physical qubit p, compute distance to each logical qubit l
    # after optimal rotation alignment

    # Procrustes: M = U_I^T @ U_H_full won't work directly (different sizes)
    # Instead, try all sign flips of eigenvectors (eigenvector sign ambiguity)
    # and pick best assignment

    best_cost = float('inf')
    best_assignment = None

    # Try sign combinations for first few eigenvectors
    sign_combos = []
    for bits in range(1 << min(actual_d, 4)):
        signs = np.array([1 - 2 * ((bits >> i) & 1) for i in range(actual_d)])
        sign_combos.append(signs)

    for signs in sign_combos:
        U_I_signed = U_I * signs[np.newaxis, :]

        # Compute pairwise distance matrix: D[l, p] = ||U_I_signed[l] - U_H[p]||^2
        # Using broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d)
        diff = U_I_signed[:, np.newaxis, :] - U_H[np.newaxis, :, :]
        D = np.sum(diff ** 2, axis=2)  # n x m_hw

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(D)
        cost = D[row_ind, col_ind].sum()

        if cost < best_cost:
            best_cost = cost
            best_assignment = (row_ind, col_ind)

    # Build initial mapping from spectral assignment
    mapping = [-1] * num_q
    rev_mapping = [-1] * num_q
    used_phys = set()

    row_ind, col_ind = best_assignment
    for r, c in zip(row_ind, col_ind):
        lq = interacting_logical[r]
        pq = physical_qubits[c]
        mapping[lq] = pq
        rev_mapping[pq] = lq
        used_phys.add(pq)

    # Fill non-interacting logical qubits
    free_phys = [pq for pq in physical_qubits if pq not in used_phys]
    unmapped_lq = [q for q in range(num_q) if mapping[q] == -1]
    random.shuffle(free_phys)
    for lq, pq in zip(unmapped_lq, free_phys):
        mapping[lq] = pq
        rev_mapping[pq] = lq

    # ── Step 6: Compute cost function ──
    def compute_cost(m):
        c = 0.0
        for (q1, q2), w in interaction_weight.items():
            c += w * dist[m[q1]][m[q2]]
        return c

    # ── Step 7: ILS + SA refinement ──
    current_m = mapping[:]
    current_rm = rev_mapping[:]
    current_cost = compute_cost(current_m)
    best_m = current_m[:]
    best_rm = current_rm[:]
    best_total_cost = current_cost

    inter_list = list(interacting_logical)
    n_inter = len(inter_list)

    if n_inter < 2:
        self.mapping_dict = best_m
        self.reverse_mapping_dict = best_rm
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Precompute neighbor weights for fast delta
    def delta_swap(m, lq_a, lq_b):
        pq_a, pq_b = m[lq_a], m[lq_b]
        delta = 0.0
        nbrs_a = logical_neighbors.get(lq_a, {})
        nbrs_b = logical_neighbors.get(lq_b, {})
        for q, w in nbrs_a.items():
            if q == lq_b:
                continue
            pq_q = m[q]
            delta += w * (dist[pq_b][pq_q] - dist[pq_a][pq_q])
        for q, w in nbrs_b.items():
            if q == lq_a:
                continue
            pq_q = m[q]
            delta += w * (dist[pq_a][pq_q] - dist[pq_b][pq_q])
        return delta

    # SA parameters
    T = current_cost / (n_inter + 1) * 0.3
    T_min = T * 0.001
    alpha_cool = 0.97
    max_sa_iters = min(n_inter * n_inter * 8, 50000)

    # ILS outer loop
    n_restarts = min(5, max(1, 200 // max(n_inter, 1)))

    for restart in range(n_restarts):
        # SA inner loop
        temp = T
        sa_m = current_m[:]
        sa_rm = current_rm[:]
        sa_cost = current_cost

        for it in range(max_sa_iters):
            if temp < T_min:
                break

            # Pick two interacting logical qubits
            idx_a = random.randint(0, n_inter - 1)
            idx_b = random.randint(0, n_inter - 2)
            if idx_b >= idx_a:
                idx_b += 1
            lq_a = inter_list[idx_a]
            lq_b = inter_list[idx_b]

            d_cost = delta_swap(sa_m, lq_a, lq_b)

            if d_cost < 0 or random.random() < math.exp(-d_cost / max(temp, 1e-15)):
                pq_a, pq_b = sa_m[lq_a], sa_m[lq_b]
                sa_m[lq_a], sa_m[lq_b] = pq_b, pq_a
                sa_rm[pq_a], sa_rm[pq_b] = lq_b, lq_a
                sa_cost += d_cost

                if sa_cost < best_total_cost - 1e-9:
                    best_total_cost = sa_cost
                    best_m = sa_m[:]
                    best_rm = sa_rm[:]

            temp *= alpha_cool

        # Perturbation for next restart: random swaps from best
        current_m = best_m[:]
        current_rm = best_rm[:]
        current_cost = best_total_cost

        # Perturb: do a few random swaps
        n_perturb = max(2, n_inter // 5)
        for _ in range(n_perturb):
            idx_a = random.randint(0, n_inter - 1)
            idx_b = random.randint(0, n_inter - 2)
            if idx_b >= idx_a:
                idx_b += 1
            lq_a = inter_list[idx_a]
            lq_b = inter_list[idx_b]
            pq_a, pq_b = current_m[lq_a], current_m[lq_b]
            current_m[lq_a], current_m[lq_b] = pq_b, pq_a
            current_rm[pq_a], current_rm[pq_b] = lq_b, lq_a
        current_cost = compute_cost(current_m)

    # ── Step 8: Final 2-opt local search on best ──
    improved = True
    rounds = 0
    while improved and rounds < 5:
        improved = False
        rounds += 1
        for i in range(n_inter):
            for j in range(i + 1, n_inter):
                lq_a = inter_list[i]
                lq_b = inter_list[j]
                d_cost = delta_swap(best_m, lq_a, lq_b)
                if d_cost < -1e-12:
                    pq_a, pq_b = best_m[lq_a], best_m[lq_b]
                    best_m[lq_a], best_m[lq_b] = pq_b, pq_a
                    best_rm[pq_a], best_rm[pq_b] = lq_b, lq_a
                    best_total_cost += d_cost
                    improved = True

    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)