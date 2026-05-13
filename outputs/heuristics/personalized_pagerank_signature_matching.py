def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits

    # ---- gather logical interactions from self.access (access2q may be None)
    log_adj = defaultdict(lambda: defaultdict(float))
    active = set()
    for _, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            log_adj[a][b] += 1.0
            log_adj[b][a] += 1.0
            active.add(a); active.add(b)

    # If QIG already provides richer info, prefer it
    try:
        for u, nbrs in self.qubit_interaction_graph.items():
            for v, w in nbrs.items():
                if u == v or w <= 0:
                    continue
                if log_adj[u][v] < w:
                    log_adj[u][v] = float(w)
                    log_adj[v][u] = float(w)
                active.add(u); active.add(v)
    except Exception:
        pass

    # ---- build dense logical/physical transition matrices over N nodes
    def build_transition(adj_neighbors_fn, n):
        P = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            nbrs = adj_neighbors_fn(i)
            tot = 0.0
            for j, w in nbrs:
                if 0 <= j < n and j != i and w > 0:
                    P[i, j] = w
                    tot += w
            if tot > 0:
                P[i, :] /= tot
            else:
                # dangling: uniform teleport
                P[i, :] = 1.0 / n
        return P

    def log_neighbors(i):
        return [(j, w) for j, w in log_adj.get(i, {}).items()]

    def phys_neighbors(i):
        out = []
        try:
            for j in self.backend.get(i, set()):
                if 0 <= j < N and j != i:
                    out.append((j, 1.0))
        except Exception:
            pass
        return out

    P_log = build_transition(log_neighbors, N)
    P_phys = build_transition(phys_neighbors, N)

    # ---- personalized PageRank by power iteration; rows of (1-a)(I - a P^T)^{-1}
    # We compute PPR via iteration: r_{k+1} = a * P^T r_k + (1-a) e_s, batched over all s.
    def ppr_matrix(P, alpha=0.15, iters=40, tol=1e-6):
        n = P.shape[0]
        Pt = P.T
        # R[:, s] = PPR vector for source s
        R = np.full((n, n), 1.0 / n, dtype=np.float64)
        E = np.eye(n, dtype=np.float64) * alpha
        one_minus_a = 1.0 - alpha
        for _ in range(iters):
            R_new = one_minus_a * (Pt @ R) + E
            if np.max(np.abs(R_new - R)) < tol:
                R = R_new
                break
            R = R_new
        return R  # column s = PPR(s)

    try:
        PPR_log = ppr_matrix(P_log)
        PPR_phys = ppr_matrix(P_phys)

        # ---- fingerprint = sorted (descending) PPR vector, fixed length L
        L = min(N, 32)
        def fingerprints(PPR):
            F = np.zeros((N, L), dtype=np.float64)
            for s in range(N):
                v = np.sort(PPR[:, s])[::-1]
                F[s, :] = v[:L]
            # normalize each fingerprint to unit L1 to make comparable
            sums = F.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            return F / sums
        F_log = fingerprints(PPR_log)
        F_phys = fingerprints(PPR_phys)

        # ---- weights: logical activity (idle logicals get small weight)
        weights = np.ones(N, dtype=np.float64)
        try:
            for q, w in self.logical_activity.items():
                if 0 <= q < N:
                    weights[q] = 1.0 + float(w)
        except Exception:
            pass
        for q in active:
            if 0 <= q < N and weights[q] < 1.0:
                weights[q] = 1.0

        # ---- cost matrix: weighted L1 distance between fingerprints
        # C[i, j] = w_i * sum_k |F_log[i,k] - F_phys[j,k]|
        diff = np.abs(F_log[:, None, :] - F_phys[None, :, :]).sum(axis=2)
        C = diff * weights[:, None]

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(C)

        mapping = [-1] * N
        for r, c in zip(row_ind, col_ind):
            mapping[int(r)] = int(c)

        # safety: ensure no -1 and injection
        used = set(p for p in mapping if p >= 0)
        free = [p for p in range(N) if p not in used]
        for i in range(N):
            if mapping[i] == -1:
                mapping[i] = free.pop()

        if len(set(mapping)) != N:
            raise RuntimeError("non-injective ppr mapping")

        self.mapping_dict = mapping
        self.reverse_mapping_dict = [0] * N
        for l, p in enumerate(mapping):
            self.reverse_mapping_dict[p] = l

    except Exception:
        # Fallback: structure-aware initial mapping then identity completion
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            md, rmd = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, N
            )
            self.mapping_dict = list(md)
            self.reverse_mapping_dict = list(rmd)
        except Exception:
            self.mapping_dict = list(range(N))
            self.reverse_mapping_dict = list(range(N))

        used = set()
        for i, p in enumerate(self.mapping_dict):
            if not isinstance(p, int) or p < 0 or p >= N or p in used:
                self.mapping_dict[i] = -1
            else:
                used.add(p)
        free = [p for p in range(N) if p not in used]
        for i in range(N):
            if self.mapping_dict[i] == -1:
                self.mapping_dict[i] = free.pop()
        self.reverse_mapping_dict = [0] * N
        for l, p in enumerate(self.mapping_dict):
            self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)