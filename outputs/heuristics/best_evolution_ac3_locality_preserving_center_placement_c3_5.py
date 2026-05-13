def init_mapping(self):
    import math
    from collections import defaultdict, deque

    N = self.num_qubits
    dm = self.distance_matrix

    # 1. Weighted logical interaction graph
    logical_adj = defaultdict(lambda: defaultdict(float))
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a][b] += 1.0
            logical_adj[b][a] += 1.0
            active_logicals.add(a)
            active_logicals.add(b)
        elif len(qubits) == 1:
            active_logicals.add(qubits[0])

    def wdeg(q):
        return sum(logical_adj[q].values())

    # 2. Physical neighbors + eccentricity/closeness/diameter
    def phys_neighbors(p):
        try:
            nb = self.backend[p]
        except Exception:
            return []
        out = []
        for x in nb:
            if isinstance(x, (list, tuple)):
                continue
            if isinstance(x, int) and 0 <= x < N and x != p:
                out.append(x)
        return out

    nbr_cache = [phys_neighbors(p) for p in range(N)]
    BIG = N * N

    phys_eccentricity = [0] * N
    phys_closeness = [0.0] * N
    diameter = 0
    for p in range(N):
        s = 0.0
        ecc = 0
        row = dm[p]
        for q in range(N):
            if p == q:
                continue
            d = row[q]
            if d > 0:
                s += d
                if d > ecc:
                    ecc = d
            else:
                s += BIG
                ecc = BIG
        phys_closeness[p] = s
        phys_eccentricity[p] = ecc
        if 0 < ecc < BIG and ecc > diameter:
            diameter = ecc
    if diameter <= 0:
        diameter = max(2, int(math.ceil(math.log2(max(N, 2)))))

    # 3. AC3 setup — initial domains constrained by physical degree.
    # HYPOTHESIS APPLIED: shrink each logical's candidate physical set up
    # front to nodes whose physical degree can plausibly support that
    # logical's interaction valence, so coupling-graph locality is enforced
    # before any assignment.
    log_degree = {L: len(set(logical_adj[L].keys()) & active_logicals)
                  for L in active_logicals}
    phys_degree = [len(nbr_cache[p]) for p in range(N)]

    domains = {}
    for L in active_logicals:
        ld = log_degree[L]
        if ld <= 1:
            cand = set(range(N))
        else:
            cand = {p for p in range(N) if phys_degree[p] >= max(1, ld - 1)}
            if len(cand) < ld + 1:
                cand = set(range(N))
        domains[L] = cand

    log_neighbors = {L: (set(logical_adj[L].keys()) & active_logicals)
                     for L in active_logicals}

    # Tight per-edge distance budgets: heavier interactions get tighter slack.
    base_slack = max(1, diameter // 3)

    def edge_budget(weight):
        b = base_slack - int(math.floor(math.log2(weight + 1)))
        return max(1, b)

    mapping = [None] * N
    reverse = [None] * N
    placed = set()
    used_phys = set()

    # 4. Seed: heaviest-weighted-degree logical -> graph center
    seed_logical = None
    seed_physical = None
    if active_logicals:
        seed_logical = max(
            active_logicals,
            key=lambda q: (wdeg(q), len(logical_adj[q]), -q),
        )
        seed_physical = min(
            range(N),
            key=lambda p: (
                phys_eccentricity[p],
                phys_closeness[p],
                -phys_degree[p],
                p,
            ),
        )
        mapping[seed_logical] = seed_physical
        reverse[seed_physical] = seed_logical
        placed.add(seed_logical)
        used_phys.add(seed_physical)
        domains[seed_logical] = {seed_physical}

    # Distance-to-center field for tiebreaking
    if seed_physical is not None:
        center_row = dm[seed_physical]
        dist_to_center = [
            (center_row[p] if center_row[p] > 0 or p == seed_physical else BIG)
            for p in range(N)
        ]
    else:
        dist_to_center = [0] * N

    # 5. AC3 arc-consistency propagation — central mechanism for hypothesis.
    # After every commitment, propagate constraints transitively until no
    # domain shrinks further, eliminating any physical that cannot satisfy
    # the per-edge distance budget to some feasible neighbor placement.
    def ac3_propagate(L_just_placed, P_just_placed):
        # Strip the assigned physical from every other domain (alldiff).
        for L2 in domains:
            if L2 in placed:
                continue
            domains[L2].discard(P_just_placed)

        queue = deque()
        # Seed the queue with arcs from the freshly-placed logical.
        for L2 in log_neighbors.get(L_just_placed, ()):
            if L2 in placed:
                continue
            queue.append((L2, L_just_placed))

        # Also reconsider arcs incident on any domain that contained
        # P_just_placed (it may have lost its only consistent support).
        for L2 in list(domains.keys()):
            if L2 in placed:
                continue
            for L3 in log_neighbors.get(L2, ()):
                if L3 in placed:
                    continue
                queue.append((L2, L3))

        while queue:
            Li, Lj = queue.popleft()
            if Li in placed:
                continue
            w = logical_adj[Li].get(Lj, 0.0)
            if w <= 0:
                continue
            budget = edge_budget(w)

            # Domain of Lj for support lookup
            if Lj in placed:
                support_set = {mapping[Lj]}
            else:
                support_set = domains.get(Lj, set())

            if not support_set:
                continue

            new_dom = set()
            for pi in domains[Li]:
                if pi in used_phys and pi != mapping[Li]:
                    continue
                row_i = dm[pi]
                feasible = False
                # Try strict budget first.
                for pj in support_set:
                    if pj == pi:
                        continue
                    d = row_i[pj]
                    if 0 < d <= budget:
                        feasible = True
                        break
                if feasible:
                    new_dom.add(pi)

            # If pruning would empty the domain, relax budget gradually
            # rather than wipe out (keep the propagation safe).
            if not new_dom:
                relaxed = budget
                while not new_dom and relaxed < diameter * 2:
                    relaxed += 1
                    for pi in domains[Li]:
                        if pi in used_phys and pi != mapping[Li]:
                            continue
                        row_i = dm[pi]
                        for pj in support_set:
                            if pj == pi:
                                continue
                            d = row_i[pj]
                            if 0 < d <= relaxed:
                                new_dom.add(pi)
                                break
                if not new_dom:
                    new_dom = {p for p in domains[Li] if p not in used_phys}
                    if not new_dom:
                        continue

            if new_dom != domains[Li]:
                domains[Li] = new_dom
                # Re-enqueue arcs into Li so transitive shrinkage propagates.
                for Lk in log_neighbors.get(Li, ()):
                    if Lk in placed or Lk == Lj:
                        continue
                    queue.append((Lk, Li))

    if seed_logical is not None:
        ac3_propagate(seed_logical, seed_physical)

    # 6. Sequential placement: most-constrained-variable on AC3-pruned domains
    interaction_to_placed = defaultdict(float)
    if seed_logical is not None:
        for nb, w in logical_adj[seed_logical].items():
            interaction_to_placed[nb] += w

    remaining = set(active_logicals) - placed

    while remaining:
        # MRV with frontier-connectivity tiebreak — domains have already
        # been AC3-pruned, so this picks the logical whose hardware-feasible
        # placement choices are scarcest first, embodying the hypothesis
        # that pre-assignment pruning yields locality-preserving choices.
        best_l = None
        best_score = None
        for l in remaining:
            connectivity = interaction_to_placed.get(l, 0.0)
            dom = domains.get(l)
            dom_size = len([p for p in dom if p not in used_phys]) if dom else N
            if dom_size == 0:
                dom_size = N
            score = (dom_size, -connectivity, -wdeg(l), l)
            if best_score is None or score < best_score:
                best_score = score
                best_l = l

        candidate_phys = [p for p in domains.get(best_l, set(range(N)))
                          if p not in used_phys]
        if not candidate_phys:
            candidate_phys = [p for p in range(N) if p not in used_phys]

        has_placed_nbr = any(nb in placed for nb in logical_adj[best_l])
        best_p = None
        best_c = float("inf")
        for p in candidate_phys:
            if has_placed_nbr:
                c = 0.0
                for nb, w in logical_adj[best_l].items():
                    if nb in placed:
                        pp = mapping[nb]
                        d = dm[p][pp]
                        c += w * (d if d > 0 else BIG)
                c += 1e-3 * phys_eccentricity[p]
                c += 1e-6 * dist_to_center[p]
            else:
                c = (phys_eccentricity[p] * BIG
                     + dist_to_center[p] * (diameter + 1)
                     + phys_closeness[p])
            if c < best_c:
                best_c = c
                best_p = p

        if best_p is None:
            for p in range(N):
                if p not in used_phys:
                    best_p = p
                    break

        mapping[best_l] = best_p
        reverse[best_p] = best_l
        placed.add(best_l)
        used_phys.add(best_p)
        remaining.discard(best_l)
        domains[best_l] = {best_p}
        for nb, w in logical_adj[best_l].items():
            if nb not in placed:
                interaction_to_placed[nb] += w
        ac3_propagate(best_l, best_p)

    # 7. Fill inactive logicals into remaining physicals near the center
    free_phys = [p for p in range(N) if p not in used_phys]
    free_phys.sort(key=lambda p: (dist_to_center[p], phys_eccentricity[p], p))
    fi = 0
    for L in range(N):
        if mapping[L] is None and fi < len(free_phys):
            p = free_phys[fi]
            mapping[L] = p
            reverse[p] = L
            fi += 1

    used_phys = set(p for p in mapping if p is not None)
    free_phys = [p for p in range(N) if p not in used_phys]
    free_phys.sort(key=lambda p: (dist_to_center[p], phys_eccentricity[p], p))
    fi = 0
    for L in range(N):
        if mapping[L] is None and fi < len(free_phys):
            p = free_phys[fi]
            mapping[L] = p
            reverse[p] = L
            fi += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)