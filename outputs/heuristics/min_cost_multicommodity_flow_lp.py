def init_mapping(self):
    import numpy as np
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None

    N = int(self.num_qubits)

    # Default identity (acts as universal fallback before any work)
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    # 1) Collect logical qubits appearing in 2-qubit gates
    logical_set = set()
    for _, qubits in self.access.items():
        if len(qubits) == 2:
            logical_set.add(qubits[0])
            logical_set.add(qubits[1])

    if not logical_set:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_qubits = sorted(q for q in logical_set if 0 <= q < N)
    L_count = len(logical_qubits)
    if L_count == 0:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    log_row = {l: i for i, l in enumerate(logical_qubits)}

    # 2) Logical activity (gate-frequency proxy)
    activity = {}
    for l in logical_qubits:
        s = 0.0
        nbrs = self.qubit_interaction_graph.get(l, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[l]
        for _, w in nbrs.items():
            s += float(w)
        activity[l] = s

    # 3) Structural prior prov(v): rank logicals by activity desc, physicals by centrality desc
    cent = self.physical_centrality if isinstance(self.physical_centrality, dict) else {p: float(self.physical_centrality[p]) for p in range(N)}
    physicals_by_cent = sorted(range(N), key=lambda p: -float(cent.get(p, 0.0)))

    # Include all logicals that appear at least as a neighbor too (for prov coverage)
    all_neighbor_logicals = set(logical_qubits)
    for l in logical_qubits:
        nbrs = self.qubit_interaction_graph.get(l, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[l]
        for v in nbrs.keys():
            if 0 <= v < N:
                all_neighbor_logicals.add(v)

    nbr_act = {}
    for v in all_neighbor_logicals:
        if v in activity:
            nbr_act[v] = activity[v]
        else:
            s = 0.0
            nbrs = self.qubit_interaction_graph.get(v, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[v]
            for _, w in nbrs.items():
                s += float(w)
            nbr_act[v] = s

    ranked_logicals = sorted(all_neighbor_logicals, key=lambda l: (-nbr_act.get(l, 0.0), l))
    prov = {}
    for i, l in enumerate(ranked_logicals):
        if i < N:
            prov[l] = physicals_by_cent[i]

    # 4) Build cost matrix C[L][P] = sum_v w(L,v) * dist(P, prov(v))
    C = np.zeros((L_count, N), dtype=float)
    for l in logical_qubits:
        i = log_row[l]
        nbrs = self.qubit_interaction_graph.get(l, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[l]
        for v, w in nbrs.items():
            if v not in prov:
                continue
            pv = prov[v]
            wf = float(w)
            if wf == 0.0:
                continue
            for p in range(N):
                d = self.distance_matrix[p][pv]
                C[i, p] += wf * float(d)

    # Pad to square N x N (dummy commodities cost 0 — they will absorb idle physicals)
    if L_count < N:
        pad = np.zeros((N - L_count, N), dtype=float)
        C_full = np.vstack([C, pad])
    else:
        C_full = C

    # 5) Solve LP relaxation (assignment polytope is integral → exact rounded optimum)
    new_mapping = [None] * N
    used_phys = set()
    if linear_sum_assignment is not None:
        try:
            row_ind, col_ind = linear_sum_assignment(C_full)
            for r, c in zip(row_ind, col_ind):
                c = int(c)
                if r < L_count:
                    l = logical_qubits[int(r)]
                    new_mapping[l] = c
                    used_phys.add(c)
        except Exception:
            pass

    # Capacity repair: unmatched logicals → unused physicals (greedy by activity desc)
    unmatched_logicals = [l for l in range(N) if new_mapping[l] is None]
    unmatched_logicals.sort(key=lambda l: -activity.get(l, 0.0))
    unused_physicals = [p for p in physicals_by_cent if p not in used_phys]
    for l, p in zip(unmatched_logicals, unused_physicals):
        new_mapping[l] = p
        used_phys.add(p)

    # Final safety: any None left → identity-fill
    if any(m is None for m in new_mapping):
        remaining_phys = [p for p in range(N) if p not in used_phys]
        ri = 0
        for l in range(N):
            if new_mapping[l] is None:
                if ri < len(remaining_phys):
                    new_mapping[l] = remaining_phys[ri]
                    ri += 1
                else:
                    new_mapping[l] = l

    self.mapping_dict = [int(p) for p in new_mapping]
    self.reverse_mapping_dict = [0] * N
    for l, p in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)