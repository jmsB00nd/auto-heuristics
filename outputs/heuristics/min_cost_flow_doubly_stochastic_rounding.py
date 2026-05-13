def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_qubits.add(q)

    if not logical_qubits:
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logicals = sorted(q for q in logical_qubits if 0 <= q < N)

    activity = {}
    for L in logicals:
        a = int(self.logical_activity.get(L, 0)) if hasattr(self, "logical_activity") else 0
        activity[L] = max(1, a)

    centrality = {}
    for P in range(N):
        c = 0.0
        if hasattr(self, "physical_centrality"):
            c = float(self.physical_centrality.get(P, 0.0))
        centrality[P] = c
    max_c = max(centrality.values()) if centrality else 0.0
    if max_c <= 0:
        for P in range(N):
            centrality[P] = 1.0
        max_c = 1.0
    floor_c = max_c * 1e-3
    for P in range(N):
        if centrality[P] < floor_c:
            centrality[P] = floor_c

    SCALE = 1000

    def cost(L, P):
        return int(round(activity[L] / centrality[P] * SCALE / max_c))

    pref = {L: {} for L in logicals}
    flow_ok = False
    try:
        import networkx as nx
        G = nx.DiGraph()
        SRC, SNK = "__SRC__", "__SNK__"
        total_supply = sum(activity[L] for L in logicals)
        per_phys_cap = (total_supply + N - 1) // N
        if per_phys_cap < 1:
            per_phys_cap = 1
        for L in logicals:
            G.add_edge(SRC, ("L", L), capacity=activity[L], weight=0)
            for P in range(N):
                G.add_edge(("L", L), ("P", P),
                           capacity=activity[L], weight=cost(L, P))
        for P in range(N):
            G.add_edge(("P", P), SNK, capacity=per_phys_cap, weight=0)
        flow_dict = nx.max_flow_min_cost(G, SRC, SNK)
        for L in logicals:
            ld = flow_dict.get(("L", L), {})
            for key, f in ld.items():
                if isinstance(key, tuple) and key[0] == "P" and f > 0:
                    pref[L][key[1]] = f
        flow_ok = True
    except Exception:
        flow_ok = False

    if not flow_ok:
        for L in logicals:
            for P in range(N):
                pref[L][P] = activity[L] * centrality[P]

    candidates = []
    for L in logicals:
        for P, f in pref[L].items():
            candidates.append((-f, -activity[L], L, P))
    candidates.sort()

    used_phys = set()
    assigned = set()
    for _, _, L, P in candidates:
        if L in assigned or P in used_phys:
            continue
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L
        assigned.add(L)
        used_phys.add(P)
        if len(assigned) == len(logicals):
            break

    free_phys_iter = (p for p in range(N) if p not in used_phys)
    for L in logicals:
        if L in assigned:
            continue
        for P in free_phys_iter:
            self.mapping_dict[L] = P
            self.reverse_mapping_dict[P] = L
            assigned.add(L)
            used_phys.add(P)
            break

    remaining = [p for p in range(N) if p not in used_phys]
    ri = 0
    for i in range(N):
        if self.mapping_dict[i] == -1:
            if ri < len(remaining):
                P = remaining[ri]
                ri += 1
                self.mapping_dict[i] = P
                self.reverse_mapping_dict[P] = i

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)