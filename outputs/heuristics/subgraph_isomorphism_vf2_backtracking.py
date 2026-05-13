def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    backend = self.backend
    bconn = self.backend_connections
    centrality = self.physical_centrality

    qig = defaultdict(lambda: defaultdict(int))
    logical_set = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_set.add(a); logical_set.add(b)
            if a != b:
                qig[a][b] += 1
                qig[b][a] += 1
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    logicals = sorted(logical_set)

    log_deg = {q: len(qig[q]) for q in logicals}
    log_wgt = {q: sum(qig[q].values()) for q in logicals}
    phys_deg = {p: len(backend[p]) for p in range(N)}

    order = sorted(logicals, key=lambda q: (-log_deg[q], -log_wgt[q], q))

    phys_by_central = sorted(range(N), key=lambda p: (-centrality.get(p, 0.0), -phys_deg[p], p))

    def mapping_score(phi):
        s = 0
        seen = set()
        for u, nbrs in qig.items():
            if u not in phi:
                continue
            pu = phi[u]
            for v, w in nbrs.items():
                if v not in phi or v <= u and (v, u) in seen:
                    continue
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                pv = phi[v]
                if (pu, pv) in bconn or (pv, pu) in bconn:
                    s += w
        return s

    best = {"score": -1, "phi": {}}
    budget = [max(2000, 50 * max(1, len(order)) * max(1, N))]

    def feasible(u, p, phi, used):
        if p in used:
            return False
        if phys_deg[p] < log_deg[u]:
            unmapped_nbrs = sum(1 for v in qig[u] if v not in phi)
            mapped_nbrs = log_deg[u] - unmapped_nbrs
            if phys_deg[p] < mapped_nbrs:
                return False
            free_phys_nbrs = sum(1 for x in backend[p] if x not in used)
            if free_phys_nbrs < unmapped_nbrs - 1 and unmapped_nbrs > 0:
                return False
        for v, _w in qig[u].items():
            if v in phi:
                pv = phi[v]
                if (p, pv) not in bconn and (pv, p) not in bconn:
                    pass
        return True

    def candidates(u, phi, used):
        cand_set = set()
        for v in qig[u]:
            if v in phi:
                for x in backend[phi[v]]:
                    if x not in used:
                        cand_set.add(x)
        if not cand_set:
            for p in phys_by_central:
                if p not in used:
                    cand_set.add(p)
                    if len(cand_set) >= max(8, N // 4):
                        break
        scored = []
        for p in cand_set:
            adj_bonus = 0
            for v, w in qig[u].items():
                if v in phi:
                    pv = phi[v]
                    if (p, pv) in bconn or (pv, p) in bconn:
                        adj_bonus += w
            scored.append((-adj_bonus, -centrality.get(p, 0.0), -phys_deg[p], p))
        scored.sort()
        return [t[3] for t in scored]

    def backtrack(idx, phi, used, partial_score):
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if idx == len(order):
            if partial_score > best["score"]:
                best["score"] = partial_score
                best["phi"] = dict(phi)
            return
        u = order[idx]
        cands = candidates(u, phi, used)
        if not cands:
            if partial_score > best["score"]:
                best["score"] = partial_score
                best["phi"] = dict(phi)
            return
        for p in cands:
            if not feasible(u, p, phi, used):
                continue
            gain = 0
            for v, w in qig[u].items():
                if v in phi:
                    pv = phi[v]
                    if (p, pv) in bconn or (pv, p) in bconn:
                        gain += w
            phi[u] = p
            used.add(p)
            backtrack(idx + 1, phi, used, partial_score + gain)
            del phi[u]
            used.discard(p)
            if budget[0] <= 0:
                return

    if order:
        seeds = phys_by_central[:max(1, min(len(phys_by_central), 4))]
        u0 = order[0]
        for seed in seeds:
            if budget[0] <= 0:
                break
            phi0 = {u0: seed}
            used0 = {seed}
            backtrack(1, phi0, used0, 0)
    else:
        best["phi"] = {}
        best["score"] = 0

    phi = dict(best["phi"])

    if any(q not in phi for q in order):
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            fb_map, _fb_rev = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits
            )
            used_now = set(phi.values())
            for L in order:
                if L in phi:
                    continue
                cand = fb_map[L] if L < len(fb_map) else None
                if cand is not None and cand not in used_now:
                    phi[L] = cand
                    used_now.add(cand)
        except Exception:
            pass

    used_phys = set(phi.values())
    free_phys = [p for p in phys_by_central if p not in used_phys]

    mapping_list = [-1] * N
    for L, P in phi.items():
        if 0 <= L < N:
            mapping_list[L] = P

    free_logicals = [L for L in range(N) if mapping_list[L] == -1]
    fp_iter = iter(free_phys)
    for L in free_logicals:
        try:
            p = next(fp_iter)
        except StopIteration:
            break
        mapping_list[L] = p

    if any(x == -1 for x in mapping_list):
        used_phys2 = set(p for p in mapping_list if p != -1)
        leftover = [p for p in range(N) if p not in used_phys2]
        li = iter(leftover)
        for L in range(N):
            if mapping_list[L] == -1:
                try:
                    mapping_list[L] = next(li)
                except StopIteration:
                    for p in range(N):
                        if p not in used_phys2:
                            mapping_list[L] = p
                            used_phys2.add(p)
                            break

    reverse_list = [-1] * N
    for L in range(N):
        P = mapping_list[L]
        if 0 <= P < N:
            reverse_list[P] = L

    if any(x == -1 for x in reverse_list):
        used_log = set(L for L in reverse_list if L != -1)
        free_log = [L for L in range(N) if L not in used_log]
        fl_iter = iter(free_log)
        for P in range(N):
            if reverse_list[P] == -1:
                try:
                    reverse_list[P] = next(fl_iter)
                except StopIteration:
                    break

    self.mapping_dict = mapping_list
    self.reverse_mapping_dict = reverse_list

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)