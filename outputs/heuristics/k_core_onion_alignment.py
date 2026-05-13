def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    # ---- Build weighted logical interaction graph from self.access ----
    log_adj = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            log_adj[a][b] += 1.0
            log_adj[b][a] += 1.0
            logical_qubits.add(a)
            logical_qubits.add(b)

    # ---- Build unit-weight physical graph from self.backend ----
    phys_adj = defaultdict(lambda: defaultdict(float))
    for p in range(N):
        for nb in self.backend.get(p, ()):  # set of neighbors
            if nb != p:
                phys_adj[p][nb] = 1.0

    # ---- Weighted k-core decomposition (Batagelj-Zaversnik, lazy heap) ----
    def weighted_core(nodes, adj):
        if not nodes:
            return {}
        wdeg = {n: sum(adj[n].values()) for n in nodes}
        heap = [(wdeg[n], n) for n in nodes]
        heapq.heapify(heap)
        remaining = set(nodes)
        core = {}
        cur = 0.0
        while heap and remaining:
            w, n = heapq.heappop(heap)
            if n not in remaining:
                continue
            if w != wdeg[n]:  # stale heap entry
                continue
            if w > cur:
                cur = w
            core[n] = cur
            remaining.discard(n)
            for nb, ew in adj[n].items():
                if nb in remaining:
                    wdeg[nb] -= ew
                    heapq.heappush(heap, (wdeg[nb], nb))
        return core

    log_nodes = list(logical_qubits)
    phys_nodes = list(range(N))
    log_core = weighted_core(log_nodes, log_adj)
    phys_core = weighted_core(phys_nodes, phys_adj)

    # ---- Shell grouping (innermost first) ----
    def shells_desc(core_dict):
        groups = defaultdict(list)
        for n, c in core_dict.items():
            groups[round(c, 9)].append(n)
        return sorted(groups.items(), key=lambda kv: -kv[0])

    activity = getattr(self, "logical_activity", {}) or {}
    centrality = getattr(self, "physical_centrality", {}) or {}

    log_shells = shells_desc(log_core)
    phys_shells = shells_desc(phys_core)

    log_order = []
    for _c, ns in log_shells:
        ns.sort(key=lambda l: (-activity.get(l, 0), l))
        log_order.extend(ns)

    phys_order = []
    for _c, ps in phys_shells:
        ps.sort(key=lambda p: (-centrality.get(p, 0.0), p))
        phys_order.extend(ps)

    # ---- Shell-by-shell assignment ----
    mapping = {}
    used_phys = set()
    pi = 0
    for L in log_order:
        while pi < len(phys_order) and phys_order[pi] in used_phys:
            pi += 1
        if pi >= len(phys_order):
            break
        P = phys_order[pi]
        mapping[L] = P
        used_phys.add(P)
        pi += 1

    # ---- Back-fill idle logicals onto remaining most-central physicals ----
    remaining_phys = [p for p in range(N) if p not in used_phys]
    remaining_phys.sort(key=lambda p: (-centrality.get(p, 0.0), p))
    idle_logicals = [l for l in range(N) if l not in mapping]
    idle_logicals.sort(key=lambda l: (-activity.get(l, 0), l))
    for L, P in zip(idle_logicals, remaining_phys):
        mapping[L] = P
        used_phys.add(P)

    # ---- Final safety fallback (any still-unmapped logicals -> identity-style) ----
    if len(mapping) < N:
        free = [p for p in range(N) if p not in used_phys]
        for L in range(N):
            if L not in mapping:
                if not free:
                    break
                P = free.pop(0)
                mapping[L] = P
                used_phys.add(P)

    # ---- Materialize as lists ----
    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N
    for L in range(N):
        P = mapping.get(L, L)
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)