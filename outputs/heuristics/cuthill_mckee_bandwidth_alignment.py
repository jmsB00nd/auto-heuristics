def init_mapping(self):
    import numpy as np
    from collections import defaultdict
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        scipy_ok = True
    except Exception:
        scipy_ok = False

    N = self.num_qubits

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_qubits = set()
    logical_edges = defaultdict(float)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                logical_qubits.add(a)
                continue
            logical_qubits.add(a)
            logical_qubits.add(b)
            u, v = (a, b) if a < b else (b, a)
            try:
                w = float(self.qubit_interaction_graph[a][b])
            except Exception:
                w = 1.0
            if w <= 0:
                w = 1.0
            logical_edges[(u, v)] = max(logical_edges[(u, v)], w)

    for q in range(N):
        logical_qubits.add(q)
    logical_list = sorted(logical_qubits)
    L = len(logical_list)
    log_to_idx = {q: i for i, q in enumerate(logical_list)}

    def rcm_order_logical():
        if not scipy_ok or L == 0:
            return list(logical_list)
        rows, cols, data = [], [], []
        for (u, v), w in logical_edges.items():
            if u in log_to_idx and v in log_to_idx:
                iu, iv = log_to_idx[u], log_to_idx[v]
                rows.append(iu); cols.append(iv); data.append(w)
                rows.append(iv); cols.append(iu); data.append(w)
        if not rows:
            activity = []
            for q in logical_list:
                act = 0.0
                try:
                    act = float(self.logical_activity.get(q, 0))
                except Exception:
                    act = 0.0
                activity.append((-act, q))
            activity.sort()
            return [q for _, q in activity]
        try:
            mat = csr_matrix((data, (rows, cols)), shape=(L, L))
            perm = reverse_cuthill_mckee(mat, symmetric_mode=True)
            return [logical_list[int(i)] for i in perm]
        except Exception:
            return list(logical_list)

    def rcm_order_physical():
        if not scipy_ok:
            return list(range(N))
        rows, cols, data = [], [], []
        for p, neighbors in self.backend.items():
            if not (0 <= p < N):
                continue
            for nb in neighbors:
                if 0 <= nb < N and nb != p:
                    rows.append(p); cols.append(nb); data.append(1.0)
        if not rows:
            cent = []
            for p in range(N):
                try:
                    c = float(self.physical_centrality.get(p, 0.0))
                except Exception:
                    c = 0.0
                cent.append((-c, p))
            cent.sort()
            return [p for _, p in cent]
        try:
            mat = csr_matrix((data, (rows, cols)), shape=(N, N))
            perm = reverse_cuthill_mckee(mat, symmetric_mode=True)
            return [int(i) for i in perm]
        except Exception:
            return list(range(N))

    rcm_log = rcm_order_logical()
    rcm_phys = rcm_order_physical()

    used_phys = set()
    assigned_log = set()
    K = min(len(rcm_log), len(rcm_phys))
    for i in range(K):
        lq = rcm_log[i]
        pq = rcm_phys[i]
        if lq in assigned_log or pq in used_phys:
            continue
        if not (0 <= lq < N and 0 <= pq < N):
            continue
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        assigned_log.add(lq)
        used_phys.add(pq)

    remaining_logical = [q for q in range(N) if q not in assigned_log]
    remaining_physical = [p for p in range(N) if p not in used_phys]
    try:
        remaining_physical.sort(
            key=lambda p: -float(self.physical_centrality.get(p, 0.0))
        )
    except Exception:
        pass

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        assigned_log.add(lq)
        used_phys.add(pq)

    if -1 in self.mapping_dict:
        leftover_phys = [p for p in range(N) if p not in used_phys]
        idx = 0
        for lq in range(N):
            if self.mapping_dict[lq] == -1:
                if idx < len(leftover_phys):
                    pq = leftover_phys[idx]
                    idx += 1
                else:
                    pq = lq
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_phys.add(pq)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)