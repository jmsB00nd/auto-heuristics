def init_mapping(self):
    from collections import defaultdict
    import math

    N = int(self.num_qubits)

    # ---- 1. collect active logical qubits and 2q interactions from self.access ----
    interactions = []  # list of (l1, l2, weight)
    active_logicals = set()
    pair_w = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                continue
            active_logicals.add(a); active_logicals.add(b)
            key = (a, b) if a < b else (b, a)
            pair_w[key] += 1

    # prefer the prebuilt QIG when present; else use pair_w
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig is not None:
        edges = []
        seen = set()
        for u, nbrs in qig.items():
            for v, w in nbrs.items():
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                if w > 0:
                    edges.append((key[0], key[1], int(w)))
                    active_logicals.add(int(key[0]))
                    active_logicals.add(int(key[1]))
        if not edges:
            edges = [(u, v, w) for (u, v), w in pair_w.items()]
    else:
        edges = [(u, v, w) for (u, v), w in pair_w.items()]

    edges.sort(key=lambda e: -e[2])
    # define "heavy" set: top half of edges, plus those with weight >= median*2
    if edges:
        weights_sorted = sorted([e[2] for e in edges])
        median_w = weights_sorted[len(weights_sorted)//2]
        heavy_cutoff = max(1, median_w)
        heavy_edges = [e for e in edges if e[2] >= heavy_cutoff]
        if not heavy_edges:
            heavy_edges = edges[:max(1, len(edges)//2)]
    else:
        heavy_edges = []

    # ---- 2. logical activity for ordering ----
    log_act = getattr(self, "logical_activity", None)
    def activity(L):
        if log_act is not None:
            return log_act.get(L, 0)
        return sum(w for (a, b, w) in edges if a == L or b == L)

    # ---- physical ordering by centrality ----
    pc = getattr(self, "physical_centrality", None)
    if pc:
        phys_ordered = sorted(range(N), key=lambda p: -pc.get(p, 0.0))
    else:
        # fallback: order by sum-of-distances (low = central)
        dm = self.distance_matrix
        phys_ordered = sorted(range(N), key=lambda p: sum(dm[p][q] for q in range(N)))

    # ---- 3. initialize domains ----
    logicals = sorted(active_logicals)
    if not logicals:
        # nothing to do: identity mapping
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    domains = {L: list(phys_ordered) for L in logicals}

    # diameter for budget scaling
    dm = self.distance_matrix
    try:
        diameter = max(dm[i][j] for i in range(N) for j in range(N))
        if diameter <= 0:
            diameter = N
    except Exception:
        diameter = N

    max_w = max((e[2] for e in edges), default=1)

    def dist_budget(w):
        # heavier edge -> tighter budget; weight 1 => full diameter, top weight => ~2
        frac = w / max_w
        budget = max(2, int(round(diameter * (1.0 - 0.65 * frac))))
        return budget

    # ---- 4. AC-3 arc consistency over heavy edges ----
    constraints = defaultdict(list)  # logical -> list of (neighbor, budget)
    for (u, v, w) in heavy_edges:
        if u in domains and v in domains:
            b = dist_budget(w)
            constraints[u].append((v, b))
            constraints[v].append((u, b))

    def revise(xi, xj, budget):
        removed = False
        keep = []
        Dj = domains[xj]
        if not Dj:
            return False
        Dj_set = set(Dj)
        for pi in domains[xi]:
            # need at least one pj in Dj with pj != pi and dist <= budget
            ok = False
            for pj in Dj_set:
                if pj == pi:
                    continue
                if dm[pi][pj] <= budget:
                    ok = True
                    break
            if ok:
                keep.append(pi)
            else:
                removed = True
        if removed:
            domains[xi] = keep
        return removed

    queue = []
    for u, lst in constraints.items():
        for (v, b) in lst:
            queue.append((u, v, b))

    ac_iters = 0
    ac_cap = 8 * (len(queue) + 1) + 4 * N
    while queue and ac_iters < ac_cap:
        ac_iters += 1
        xi, xj, b = queue.pop(0)
        if revise(xi, xj, b):
            if not domains[xi]:
                # empty domain -> relax: reseed with centrality order, stop AC
                domains[xi] = list(phys_ordered)
                queue = []
                break
            for (xk, bk) in constraints.get(xi, []):
                if xk != xj:
                    queue.append((xk, xi, bk))

    # ---- 5. backtracking DFS with MRV + activity ordering ----
    assignment = {}
    used = set()
    node_cap = [20000]  # mutable counter

    def order_vars():
        unassigned = [L for L in logicals if L not in assignment]
        unassigned.sort(key=lambda L: (len(domains[L]), -activity(L)))
        return unassigned

    def value_order(L):
        # prefer physicals that minimize sum of distances to already-assigned partners
        cands = [p for p in domains[L] if p not in used]
        # build partner list weight pairs
        partners = []
        for (a, b, w) in edges:
            if a == L and b in assignment:
                partners.append((assignment[b], w))
            elif b == L and a in assignment:
                partners.append((assignment[a], w))
        if not partners:
            return cands  # already centrality-ordered
        def cost(p):
            c = 0.0
            for (pp, w) in partners:
                c += w * dm[p][pp]
            # break ties with centrality (lower idx in phys_ordered = better)
            return c
        return sorted(cands, key=cost)

    def consistent(L, p):
        for (a, b, w) in heavy_edges:
            if a == L and b in assignment:
                if dm[p][assignment[b]] > dist_budget(w):
                    return False
            elif b == L and a in assignment:
                if dm[p][assignment[a]] > dist_budget(w):
                    return False
        return True

    def dfs():
        if node_cap[0] <= 0:
            return False
        node_cap[0] -= 1
        remaining = order_vars()
        if not remaining:
            return True
        L = remaining[0]
        for p in value_order(L):
            if p in used:
                continue
            if not consistent(L, p):
                continue
            assignment[L] = p
            used.add(p)
            if dfs():
                return True
            del assignment[L]
            used.discard(p)
        return False

    success = dfs()

    # ---- 6. fallback warm start if CSP failed ----
    if not success or len(assignment) < len(logicals):
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            warm_map, warm_rev = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits
            )
            assignment = {}
            used = set()
            for L in logicals:
                p = warm_map[L] if L < len(warm_map) else None
                if p is not None and p not in used and 0 <= p < N:
                    assignment[L] = p
                    used.add(p)
        except Exception:
            pass

    # ---- final sweep: ensure full-length lists and injection ----
    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = set()
    for L, p in assignment.items():
        if 0 <= L < N and 0 <= p < N and p not in used_phys:
            mapping[L] = p
            reverse[p] = L
            used_phys.add(p)

    free_phys = [p for p in phys_ordered if p not in used_phys]
    fp_idx = 0
    for L in range(N):
        if mapping[L] == -1:
            while fp_idx < len(free_phys) and free_phys[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx < len(free_phys):
                p = free_phys[fp_idx]; fp_idx += 1
                mapping[L] = p
                reverse[p] = L
                used_phys.add(p)
            else:
                # last-resort identity (shouldn't trigger given sets above)
                for p in range(N):
                    if p not in used_phys:
                        mapping[L] = p
                        reverse[p] = L
                        used_phys.add(p)
                        break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)