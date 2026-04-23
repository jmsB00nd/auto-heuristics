def init_mapping(self):
    from itertools import permutations
    import random, math

    n = self.num_qubits

    interaction = {}
    log_q_set = set()
    for gid, qs in self.access.items():
        if len(qs) == 2:
            a, b = qs[0], qs[1]
            log_q_set.add(a)
            log_q_set.add(b)
            key = (min(a, b), max(a, b))
            interaction[key] = interaction.get(key, 0) + 1

    if not log_q_set:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    nbrs = {q: [] for q in log_q_set}
    for (a, b), w in interaction.items():
        nbrs[a].append((b, w))
        nbrs[b].append((a, w))
    dist = self.distance_matrix

    def hem(nodes, adj, mem):
        edges, seen = [], set()
        for u in nodes:
            for v, w in adj.get(u, {}).items():
                e = (min(u, v), max(u, v))
                if e not in seen:
                    seen.add(e)
                    edges.append((w, u, v))
        edges.sort(reverse=True)
        matched, pairs = set(), []
        for w, u, v in edges:
            if u not in matched and v not in matched:
                pairs.append((u, v))
                matched.add(u)
                matched.add(v)
        if not pairs:
            return None
        nm, nmem, nn = {}, {}, []
        for u, v in pairs:
            nm[u] = u
            nm[v] = u
            nmem[u] = mem[u] + mem[v]
            nn.append(u)
        for nd in nodes:
            if nd not in matched:
                nm[nd] = nd
                nmem[nd] = mem[nd]
                nn.append(nd)
        ea, proc = {}, set()
        for u in nodes:
            for v, w in adj[u].items():
                p = (min(u, v), max(u, v))
                if p not in proc:
                    proc.add(p)
                    nu, nv = nm[u], nm[v]
                    if nu != nv:
                        ek = (min(nu, nv), max(nu, nv))
                        ea[ek] = ea.get(ek, 0) + w
        na = {nd: {} for nd in nn}
        for (u, v), w in ea.items():
            na[u][v] = w
            na[v][u] = w
        return nn, na, nmem

    ln = sorted(log_q_set)
    la = {q: {} for q in ln}
    for (a, b), w in interaction.items():
        la[a][b] = w
        la[b][a] = w
    lm = {q: [q] for q in ln}
    num_coarsen_levels = 0

    while len(ln) > 8:
        r = hem(ln, la, lm)
        if r is None or len(r[0]) >= len(ln):
            break
        ln, la, lm = r
        num_coarsen_levels += 1

    hw_keys = sorted(self.backend.keys())
    hn = list(hw_keys)
    ha = {q: {} for q in hn}
    for u, v in self.backend_connections:
        if u in ha and v in ha:
            ha[u][v] = 1
            ha[v][u] = 1
    hm = {q: [q] for q in hn}
    tgt = max(len(ln), 8)
    while len(hn) > tgt:
        r = hem(hn, ha, hm)
        if r is None or len(r[0]) >= len(hn):
            break
        hn, ha, hm = r

    hd = {}
    for a in hn:
        hd[(a, a)] = 0
        for b in hn:
            if a != b and (a, b) not in hd:
                t = sum(dist[x][y] for x in hm[a] for y in hm[b])
                c = len(hm[a]) * len(hm[b])
                hd[(a, b)] = t / c if c else 0
                hd[(b, a)] = hd[(a, b)]

    ce = [(u, v, w) for u in ln for v, w in la[u].items() if u < v]
    nlc, nhc = len(ln), len(hn)

    def cfn(pm):
        asgn = {ln[i]: pm[i] for i in range(nlc)}
        return sum(w * hd.get((asgn[u], asgn[v]), 0) for u, v, w in ce)

    bc, bp = float('inf'), None
    try:
        tp = math.perm(nhc, nlc) if nlc <= nhc else float('inf')
    except (ValueError, OverflowError):
        tp = float('inf')

    if nlc <= nhc and tp <= 200000:
        for p in permutations(hn, nlc):
            c = cfn(p)
            if c < bc:
                bc, bp = c, p
    elif nlc <= nhc:
        random.seed(42)
        for _ in range(min(100000, max(10000, nhc * 100))):
            p = tuple(random.sample(hn, nlc))
            c = cfn(p)
            if c < bc:
                bc, bp = c, p

    if bp is None:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    mapping = [None] * n
    used = set()
    for i in range(nlc):
        pqs = hm[bp[i]]
        for j, lq in enumerate(lm[ln[i]]):
            if j < len(pqs):
                mapping[lq] = pqs[j]
                used.add(pqs[j])

    backend_set = set(hw_keys)
    ba = sorted(backend_set - used)
    bi = 0
    for q in sorted(log_q_set):
        if mapping[q] is None:
            if bi < len(ba):
                mapping[q] = ba[bi]
                used.add(ba[bi])
                bi += 1
            else:
                for p in sorted(set(range(n)) - used):
                    mapping[q] = p
                    used.add(p)
                    break
    avail = sorted(set(range(n)) - used)
    ai = 0
    for q in range(n):
        if mapping[q] is None:
            mapping[q] = avail[ai]
            ai += 1

    log_qs = sorted(log_q_set)
    valid = [q for q in range(n) if mapping[q] in backend_set]
    nlq, nv = len(log_qs), len(valid)

    def sgain(qi, qj):
        pi, pj = mapping[qi], mapping[qj]
        g = 0
        for o, w in nbrs.get(qi, []):
            if o == qj:
                continue
            g += w * (dist[pi][mapping[o]] - dist[pj][mapping[o]])
        for o, w in nbrs.get(qj, []):
            if o == qi:
                continue
            g += w * (dist[pj][mapping[o]] - dist[pi][mapping[o]])
        return g

    total_passes = min(50, max(5, 2000000 // (nlq * nv + 1)))
    passes_per_lvl = max(1, total_passes // max(num_coarsen_levels + 1, 1))

    for _lvl in range(num_coarsen_levels, -1, -1):
        for _ in range(passes_per_lvl):
            bg, bs = 0, None
            for qi in log_qs:
                for qj in valid:
                    if qj == qi:
                        continue
                    g = sgain(qi, qj)
                    if g > bg:
                        bg, bs = g, (qi, qj)
            if bs is None:
                break
            mapping[bs[0]], mapping[bs[1]] = mapping[bs[1]], mapping[bs[0]]

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * n
    for lq in range(n):
        self.reverse_mapping_dict[mapping[lq]] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)