def init_mapping(self):
    import math
    from collections import defaultdict, deque

    N = self.num_qubits

    # ---- Step 1: logical interaction graph from self.access ----
    logical_interactions = defaultdict(lambda: defaultdict(float))
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 == q2:
                continue
            logical_interactions[q1][q2] += 1.0
            logical_interactions[q2][q1] += 1.0

    logical_degree = [0] * N
    logical_wdeg = [0.0] * N
    logical_centrality = [0.0] * N
    for q in range(N):
        nbrs = logical_interactions.get(q, {})
        logical_degree[q] = len(nbrs)
        logical_wdeg[q] = float(sum(nbrs.values()))
        cent = 0.0
        for n, w in nbrs.items():
            cent += w * len(logical_interactions.get(n, {}))
        logical_centrality[q] = cent

    # ---- Step 2: physical structural features from self.backend ----
    backend = self.backend

    def phys_neighbors(p):
        try:
            nb = backend[p]
        except (TypeError, IndexError, KeyError):
            return []
        out = []
        seen = set()
        for x in nb:
            if isinstance(x, (list, tuple)):
                continue
            if 0 <= x < N and x != p and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    phys_degree = [0] * N
    phys_wdeg = [0.0] * N
    phys_centrality = [0.0] * N
    nbr_cache = [None] * N
    for p in range(N):
        nb = phys_neighbors(p)
        nbr_cache[p] = nb
        phys_degree[p] = len(nb)
        phys_wdeg[p] = float(len(nb))
    for p in range(N):
        cent = 0.0
        for n in nbr_cache[p]:
            cent += len(nbr_cache[n] if nbr_cache[n] is not None else phys_neighbors(n))
        phys_centrality[p] = cent

    # ---- Step 3: normalize features and define cost ----
    def normalize(vec):
        mx = max(vec) if len(vec) > 0 else 0.0
        if mx <= 0:
            return [0.0] * len(vec)
        return [v / mx for v in vec]

    ld = normalize(logical_degree)
    lw = normalize(logical_wdeg)
    lc = normalize(logical_centrality)
    pd = normalize(phys_degree)
    pw = normalize(phys_wdeg)
    pc = normalize(phys_centrality)

    def cost(l, p):
        return (ld[l] - pd[p]) ** 2 + (lw[l] - pw[p]) ** 2 + (lc[l] - pc[p]) ** 2

    # ---- Step 4: build preference lists ----
    log_pref = [None] * N
    for l in range(N):
        log_pref[l] = sorted(range(N), key=lambda p: (cost(l, p), p))

    phy_rank = [None] * N
    for p in range(N):
        order = sorted(range(N), key=lambda l: (cost(l, p), l))
        rank = [0] * N
        for pos, l in enumerate(order):
            rank[l] = pos
        phy_rank[p] = rank

    # ---- Step 5: Gale-Shapley deferred acceptance ----
    next_proposal = [0] * N
    phys_match = [-1] * N
    log_match = [-1] * N

    order_free = sorted(
        range(N),
        key=lambda l: (-(logical_wdeg[l] + logical_centrality[l] + logical_degree[l]), l),
    )
    free_queue = deque(order_free)

    while free_queue:
        l = free_queue.popleft()
        if next_proposal[l] >= N:
            continue
        p = log_pref[l][next_proposal[l]]
        next_proposal[l] += 1
        cur = phys_match[p]
        if cur == -1:
            phys_match[p] = l
            log_match[l] = p
        else:
            if phy_rank[p][l] < phy_rank[p][cur]:
                phys_match[p] = l
                log_match[l] = p
                log_match[cur] = -1
                free_queue.append(cur)
            else:
                free_queue.append(l)

    # ---- Step 6: safety fallback ----
    used_phys = set(p for p in log_match if p != -1)
    for l in range(N):
        if log_match[l] == -1:
            for p in range(N):
                if p not in used_phys:
                    log_match[l] = p
                    used_phys.add(p)
                    break

    self.mapping_dict = list(log_match)
    self.reverse_mapping_dict = [0] * N
    for l in range(N):
        self.reverse_mapping_dict[self.mapping_dict[l]] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)