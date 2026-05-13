def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits

    # --- 1. Collect logical qubits and weighted interactions ---
    logical_set = set()
    pair_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 1:
            logical_set.add(qubits[0])
        elif len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_set.add(a)
            logical_set.add(b)
            if a != b:
                key = (a, b) if a < b else (b, a)
                pair_weight[key] += 1

    # Augment with prebuilt QIG (authoritative source of weights)
    for u, nbrs in self.qubit_interaction_graph.items():
        logical_set.add(u)
        for v, w in nbrs.items():
            logical_set.add(v)
            if u < v:
                pair_weight[(u, v)] = max(pair_weight[(u, v)], w)

    logicals = sorted(logical_set)
    # Cap logicals to those that fit on hardware; extras get identity later
    if len(logicals) > N:
        logicals = logicals[:N]

    # --- 2. Build logical adjacency (weighted) ---
    log_adj = defaultdict(dict)
    for (a, b), w in pair_weight.items():
        if a in logical_set and b in logical_set:
            log_adj[a][b] = w
            log_adj[b][a] = w

    def logical_role(q, K):
        # weighted hop-1: each neighbor contributes (deg(neighbor) * weight)
        hop1 = []
        for nb, w in log_adj[q].items():
            d = len(log_adj[nb])
            hop1.append(float(d) * float(w))
        # hop-2: neighbors of neighbors (weighted by product of edge weights)
        hop2 = []
        for nb, w1 in log_adj[q].items():
            for nb2, w2 in log_adj[nb].items():
                if nb2 == q:
                    continue
                d = len(log_adj[nb2])
                hop2.append(float(d) * float(w1) * float(w2))
        hop1.sort(reverse=True)
        hop2.sort(reverse=True)

        def fit(vec, k):
            if len(vec) >= k:
                return vec[:k]
            return vec + [0.0] * (k - len(vec))

        return np.array(fit(hop1, K) + fit(hop2, K), dtype=float)

    def physical_role(p, K):
        hop1 = []
        for nb in self.backend[p]:
            d = len(self.backend[nb])
            hop1.append(float(d))
        hop2 = []
        for nb in self.backend[p]:
            for nb2 in self.backend[nb]:
                if nb2 == p:
                    continue
                d = len(self.backend[nb2])
                hop2.append(float(d))
        hop1.sort(reverse=True)
        hop2.sort(reverse=True)

        def fit(vec, k):
            if len(vec) >= k:
                return vec[:k]
            return vec + [0.0] * (k - len(vec))

        return np.array(fit(hop1, K) + fit(hop2, K), dtype=float)

    # --- 3. Determine descriptor length K from max degree across both sides ---
    max_log_deg = max((len(log_adj[q]) for q in logicals), default=1)
    max_phys_deg = max((len(self.backend[p]) for p in range(N)), default=1)
    K = max(1, max(max_log_deg, max_phys_deg))

    log_vecs = [logical_role(q, K) for q in logicals]
    phys_vecs = [physical_role(p, K) for p in range(N)]

    # --- 4. Build square padded cost matrix (N x N) ---
    PAD = 1e6  # large but finite cost for padded logical rows
    cost = np.zeros((N, N), dtype=float)
    for i in range(N):
        if i < len(logicals):
            lv = log_vecs[i]
            for j in range(N):
                diff = lv - phys_vecs[j]
                cost[i, j] = float(np.sqrt(np.dot(diff, diff)))
        else:
            cost[i, :] = PAD  # padded rows: any assignment equally bad

    # --- 5. Hungarian assignment ---
    assignment = None
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        # Greedy fallback
        used_phys = set()
        assignment = []
        order = sorted(range(N), key=lambda i: -np.sum(cost[i] if i < len(logicals) else 0))
        for i in order:
            best_j, best_c = -1, float("inf")
            for j in range(N):
                if j in used_phys:
                    continue
                c = cost[i, j]
                if c < best_c:
                    best_c = c
                    best_j = j
            if best_j >= 0:
                used_phys.add(best_j)
                assignment.append((i, best_j))

    # --- 6. Materialize mapping lists ---
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    used_phys = set()
    for i, j in assignment:
        if i < len(logicals):
            L = logicals[i]
            if 0 <= L < N and self.mapping_dict[L] == -1 and j not in used_phys:
                self.mapping_dict[L] = j
                self.reverse_mapping_dict[j] = L
                used_phys.add(j)

    # Identity / fill fallback for any logical id not yet mapped
    free_phys = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for L in range(N):
        if self.mapping_dict[L] == -1:
            if L not in used_phys:
                self.mapping_dict[L] = L
                self.reverse_mapping_dict[L] = L
                used_phys.add(L)
            else:
                while fp_idx < len(free_phys) and free_phys[fp_idx] in used_phys:
                    fp_idx += 1
                if fp_idx < len(free_phys):
                    p = free_phys[fp_idx]
                    fp_idx += 1
                    self.mapping_dict[L] = p
                    self.reverse_mapping_dict[p] = L
                    used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)