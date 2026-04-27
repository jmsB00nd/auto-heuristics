def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = []
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                continue
            interactions.append((a, b))
            logical_set.add(a)
            logical_set.add(b)
        elif len(qubits) == 1:
            logical_set.add(int(qubits[0]))

    max_logical = max(logical_set) if logical_set else -1
    L_size = max(N, max_logical + 1)

    log_weight = defaultdict(float)
    for (a, b) in interactions:
        if a >= L_size or b >= L_size:
            continue
        log_weight[(a, b)] += 1.0
        log_weight[(b, a)] += 1.0

    def build_transition(size, edge_weights):
        P = np.zeros((size, size), dtype=np.float64)
        for (u, v), w in edge_weights.items():
            P[u, v] += w
        row_sums = P.sum(axis=1)
        for i in range(size):
            if row_sums[i] > 0.0:
                P[i] /= row_sums[i]
            else:
                P[i, i] = 1.0
        return P

    log_size = L_size
    P_log = build_transition(log_size, log_weight)

    phys_weight = defaultdict(float)
    try:
        for (u, v) in self.backend_connections:
            if 0 <= u < N and 0 <= v < N and u != v:
                phys_weight[(int(u), int(v))] += 1.0
    except Exception:
        for u in range(N):
            try:
                for v in self.backend[u]:
                    if 0 <= v < N and u != v:
                        phys_weight[(u, int(v))] += 1.0
            except Exception:
                pass
    P_phys = build_transition(N, phys_weight)

    def ppr_signatures(P, size, alpha=0.15, iters=50, tol=1e-9):
        sigs = np.zeros((size, size), dtype=np.float64)
        I = np.eye(size, dtype=np.float64)
        Pt = P.T
        for v in range(size):
            r = np.zeros(size, dtype=np.float64)
            r[v] = 1.0
            x = r.copy()
            for _ in range(iters):
                x_new = (1.0 - alpha) * (Pt @ x) + alpha * r
                if np.linalg.norm(x_new - x, ord=1) < tol:
                    x = x_new
                    break
                x = x_new
            sigs[v] = np.sort(x)[::-1]
        return sigs

    log_sigs = ppr_signatures(P_log, log_size)
    phys_sigs = ppr_signatures(P_phys, N)

    def sig_key(sig):
        return tuple(round(float(x), 10) for x in sig)

    logical_qubits_to_place = sorted(logical_set)
    log_keys = [(sig_key(log_sigs[q]), q) for q in logical_qubits_to_place]
    log_keys.sort(reverse=True)

    phys_keys = [(sig_key(phys_sigs[p]), p) for p in range(N)]
    phys_keys.sort(reverse=True)

    used_phys = set()
    assigned_log = set()

    for (lk, lq), (pk, pq) in zip(log_keys, phys_keys):
        if lq >= N:
            continue
        if pq in used_phys or lq in assigned_log:
            continue
        self.mapping_dict[lq] = pq
        used_phys.add(pq)
        assigned_log.add(lq)

    free_phys = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for lq in range(N):
        if self.mapping_dict[lq] != -1:
            continue
        if lq < N and lq not in used_phys:
            self.mapping_dict[lq] = lq
            used_phys.add(lq)
        else:
            while fp_idx < len(free_phys) and free_phys[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx < len(free_phys):
                self.mapping_dict[lq] = free_phys[fp_idx]
                used_phys.add(free_phys[fp_idx])
                fp_idx += 1

    free_phys = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            if fp_idx < len(free_phys):
                self.mapping_dict[lq] = free_phys[fp_idx]
                used_phys.add(free_phys[fp_idx])
                fp_idx += 1

    for lq in range(N):
        pq = self.mapping_dict[lq]
        if 0 <= pq < N:
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)