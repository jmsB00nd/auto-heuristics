def init_mapping(self):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # 1. Collect logical interactions from self.access
    logical_qubits = set()
    edges_logical = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits.add(q)
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a != b and 0 <= a < N and 0 <= b < N:
                edges_logical.add((min(a, b), max(a, b)))

    # 2. Build sparse logical adjacency (weighted by QIG)
    rows_l, cols_l, data_l = [], [], []
    for (a, b) in edges_logical:
        w = 1
        try:
            w = max(1, int(self.qubit_interaction_graph[a][b]))
        except Exception:
            w = 1
        rows_l.append(a); cols_l.append(b); data_l.append(w)
        rows_l.append(b); cols_l.append(a); data_l.append(w)
    if not rows_l:
        # No interactions: fall back to identity
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_csr = csr_matrix((data_l, (rows_l, cols_l)), shape=(N, N))

    # 3. Build sparse physical adjacency from self.backend
    rows_p, cols_p, data_p = [], [], []
    for u, neighbors in self.backend.items():
        if not (0 <= u < N):
            continue
        for v in neighbors:
            if 0 <= v < N and u != v:
                rows_p.append(u); cols_p.append(v); data_p.append(1)
    if not rows_p:
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    physical_csr = csr_matrix((data_p, (rows_p, cols_p)), shape=(N, N))

    # 4. Run RCM on both graphs
    try:
        logical_perm = list(reverse_cuthill_mckee(logical_csr, symmetric_mode=True))
    except Exception:
        logical_perm = list(range(N))
    try:
        physical_perm = list(reverse_cuthill_mckee(physical_csr, symmetric_mode=True))
    except Exception:
        physical_perm = list(range(N))

    # 5. Build full orderings: active logicals first (in RCM order), then idle logicals
    active = logical_qubits
    seen_l = set()
    logical_order = []
    for q in logical_perm:
        if q in active and q not in seen_l:
            logical_order.append(q); seen_l.add(q)
    for q in logical_perm:
        if q not in seen_l:
            logical_order.append(q); seen_l.add(q)
    for q in range(N):
        if q not in seen_l:
            logical_order.append(q); seen_l.add(q)

    seen_p = set()
    physical_order = []
    for p in physical_perm:
        if p not in seen_p and 0 <= p < N:
            physical_order.append(p); seen_p.add(p)
    for p in range(N):
        if p not in seen_p:
            physical_order.append(p); seen_p.add(p)

    # 6. Position-wise alignment
    used_phys = set()
    for i in range(N):
        L = logical_order[i]
        P = physical_order[i]
        if self.mapping_dict[L] == -1 and P not in used_phys:
            self.mapping_dict[L] = P
            self.reverse_mapping_dict[P] = L
            used_phys.add(P)

    # 7. Fallback: assign any leftover logicals to leftover physicals
    leftover_phys = [p for p in range(N) if p not in used_phys]
    li = 0
    for L in range(N):
        if self.mapping_dict[L] == -1:
            if li < len(leftover_phys):
                P = leftover_phys[li]; li += 1
                self.mapping_dict[L] = P
                self.reverse_mapping_dict[P] = L
            else:
                for P in range(N):
                    if P not in used_phys:
                        self.mapping_dict[L] = P
                        self.reverse_mapping_dict[P] = L
                        used_phys.add(P)
                        break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)