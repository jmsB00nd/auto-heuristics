def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits

    # ---- Step 1: logical interaction graph from self.access ----
    interaction_count = defaultdict(int)
    logical_neighbors = defaultdict(set)
    logical_qubits = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if not (0 <= a < N and 0 <= b < N):
                continue
            logical_qubits.add(a); logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            interaction_count[key] += 1
            logical_neighbors[a].add(b)
            logical_neighbors[b].add(a)

    logical_degree = {l: len(logical_neighbors[l]) for l in logical_qubits}
    logical_weight = defaultdict(int)
    for (a, b), w in interaction_count.items():
        logical_weight[a] += w
        logical_weight[b] += w

    def _bfs2_logical(l):
        seen = {l}
        for n1 in logical_neighbors[l]:
            seen.add(n1)
            for n2 in logical_neighbors[n1]:
                seen.add(n2)
        return len(seen) - 1
    logical_bfs2 = {l: _bfs2_logical(l) for l in logical_qubits}

    # ---- Step 2: physical supply features ----
    phys_neighbors = [set() for _ in range(N)]
    for p in range(N):
        try:
            adj = self.backend[p]
        except Exception:
            adj = []
        if adj is None:
            adj = []
        for q in adj:
            if isinstance(q, int) and 0 <= q < N and q != p:
                phys_neighbors[p].add(q)
    phys_degree = [len(phys_neighbors[p]) for p in range(N)]
    phys_nbhd_weight = [sum(phys_degree[q] for q in phys_neighbors[p]) for p in range(N)]

    def _bfs2_phys(p):
        seen = {p}
        for n1 in phys_neighbors[p]:
            seen.add(n1)
            for n2 in phys_neighbors[n1]:
                seen.add(n2)
        return len(seen) - 1
    phys_bfs2 = [_bfs2_phys(p) for p in range(N)]

    def _dist(a, b):
        try:
            d = self.distance_matrix[a][b]
            return d if d is not None else N + 1
        except Exception:
            return N + 1

    # ---- Step 3: unary consistency to seed domains ----
    domains = {}
    for l in logical_qubits:
        d_l = logical_degree[l]; b_l = logical_bfs2[l]; w_l = logical_weight[l]
        cands = [p for p in range(N)
                 if phys_degree[p] >= d_l and phys_bfs2[p] >= b_l
                 and phys_nbhd_weight[p] >= min(w_l, phys_nbhd_weight[p])]
        if not cands:
            cands = [p for p in range(N) if phys_degree[p] >= d_l and phys_bfs2[p] >= b_l]
        if not cands:
            cands = [p for p in range(N) if phys_degree[p] >= d_l]
        if not cands:
            cands = list(range(N))
        cands.sort(key=lambda p: (-phys_nbhd_weight[p], -phys_degree[p], p))
        domains[l] = cands

    # ---- Step 4: AC-3-like binary pruning ----
    arcs = deque()
    in_queue = set()
    for l in logical_qubits:
        for l2 in logical_neighbors[l]:
            arcs.append((l, l2)); in_queue.add((l, l2))

    max_iter = 4 * (len(arcs) + 1) * max(1, len(logical_qubits))
    iters = 0
    DIST_OK = 3
    while arcs and iters < max_iter:
        iters += 1
        l, l2 = arcs.popleft()
        in_queue.discard((l, l2))
        dom_l2 = set(domains.get(l2, []))
        if not dom_l2 or l not in domains:
            continue
        new_dom = []
        for p in domains[l]:
            ok = False
            for p2 in dom_l2:
                if p2 == p:
                    continue
                if _dist(p, p2) <= DIST_OK:
                    ok = True
                    break
            if ok:
                new_dom.append(p)
        if not new_dom:
            continue  # never empty a domain
        if len(new_dom) < len(domains[l]):
            domains[l] = new_dom
            for l3 in logical_neighbors[l]:
                if l3 != l2 and (l3, l) not in in_queue:
                    arcs.append((l3, l)); in_queue.add((l3, l))

    # ---- Step 5: greedy instantiation, most-constrained-variable first ----
    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = set()

    order = sorted(logical_qubits,
                   key=lambda l: (len(domains[l]), -logical_weight[l], -logical_degree[l], l))

    def _score(l, p):
        s = 0.0
        for n in logical_neighbors[l]:
            pn = mapping[n]
            if pn != -1:
                w = interaction_count[(l, n) if l < n else (n, l)]
                s -= _dist(p, pn) * w
        s += 0.01 * phys_degree[p] + 0.001 * phys_nbhd_weight[p]
        return s

    for l in order:
        cands = [p for p in domains[l] if p not in used_phys]
        if not cands:
            cands = [p for p in range(N) if p not in used_phys]
        if not cands:
            break
        best = max(cands, key=lambda p: _score(l, p))
        mapping[l] = best
        reverse[best] = l
        used_phys.add(best)

    # ---- Step 6: identity fill for any remaining logicals / physicals ----
    for l in range(N):
        if mapping[l] == -1:
            if l not in used_phys:
                mapping[l] = l
                reverse[l] = l
                used_phys.add(l)
            else:
                for p in range(N):
                    if p not in used_phys:
                        mapping[l] = p
                        reverse[p] = l
                        used_phys.add(p)
                        break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)