def init_mapping(self):
    import random
    import math

    N = self.num_qubits
    dist = self.distance_matrix

    # --- collect 2-qubit interactions and logical weights ---
    weights = {}
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_set.add(a)
            logical_set.add(b)
            key = (a, b) if a < b else (b, a)
            weights[key] = weights.get(key, 0) + 1
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    # ensure all logicals up to max id are tracked
    if logical_set:
        max_logical = max(logical_set)
    else:
        max_logical = -1

    # active logicals = those touched by any 2q interaction (priority for ACO)
    active_logicals = set()
    for (a, b) in weights.keys():
        active_logicals.add(a)
        active_logicals.add(b)
    active_logicals = sorted(active_logicals,
                             key=lambda q: -self.logical_activity.get(q, 0))

    physicals = list(range(N))

    # if no interactions, fall back to identity
    if not active_logicals:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # --- ACO parameters ---
    num_ants = max(8, min(20, N))
    num_iter = max(10, min(30, N))
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 1.0
    elite_weight = 2.0

    # pheromone: dict keyed by logical -> list over physical
    tau0 = 1.0
    tau = {L: [tau0] * N for L in active_logicals}

    # visibility: prefer central physicals for active logicals
    cent = [self.physical_centrality.get(p, 0.0) for p in range(N)]
    cmax = max(cent) if cent else 1.0
    if cmax <= 0:
        cmax = 1.0
    eta = {}
    for L in active_logicals:
        act = float(self.logical_activity.get(L, 1))
        row = []
        for p in range(N):
            v = (cent[p] / cmax) * (act + 1.0)
            if v <= 0:
                v = 1e-6
            row.append(v)
        eta[L] = row

    def construct_one():
        used_phys = set()
        mapping = {}
        # randomize ties slightly: small perturbation to ordering
        order = list(active_logicals)
        for L in order:
            tau_row = tau[L]
            eta_row = eta[L]
            scores = []
            cands = []
            total = 0.0
            for p in range(N):
                if p in used_phys:
                    continue
                s = (tau_row[p] ** alpha) * (eta_row[p] ** beta)
                if s <= 0:
                    s = 1e-12
                scores.append(s)
                cands.append(p)
                total += s
            if not cands:
                break
            r = random.random() * total
            acc = 0.0
            chosen = cands[-1]
            for p, s in zip(cands, scores):
                acc += s
                if acc >= r:
                    chosen = p
                    break
            mapping[L] = chosen
            used_phys.add(chosen)
        return mapping, used_phys

    def cost_of(mapping):
        c = 0.0
        for (a, b), w in weights.items():
            if a in mapping and b in mapping:
                c += w * dist[mapping[a]][mapping[b]]
        return c

    best_mapping = None
    best_cost = float('inf')

    for _ in range(num_iter):
        iter_best_mapping = None
        iter_best_cost = float('inf')
        ant_results = []
        for _a in range(num_ants):
            m, _u = construct_one()
            c = cost_of(m)
            ant_results.append((c, m))
            if c < iter_best_cost:
                iter_best_cost = c
                iter_best_mapping = m
            if c < best_cost:
                best_cost = c
                best_mapping = dict(m)

        # evaporation
        for L in tau:
            row = tau[L]
            for p in range(N):
                row[p] *= (1.0 - rho)
                if row[p] < 1e-9:
                    row[p] = 1e-9

        # deposit from all ants (proportional to 1/cost)
        for c, m in ant_results:
            if c <= 0:
                deposit = Q
            else:
                deposit = Q / (1.0 + c)
            for L, p in m.items():
                if L in tau:
                    tau[L][p] += deposit

        # elite reinforcement on iteration best
        if iter_best_mapping is not None:
            if iter_best_cost <= 0:
                ed = Q * elite_weight
            else:
                ed = (Q * elite_weight) / (1.0 + iter_best_cost)
            for L, p in iter_best_mapping.items():
                if L in tau:
                    tau[L][p] += ed

    if best_mapping is None:
        best_mapping = {}

    # --- materialize into list-based mapping_dict / reverse_mapping_dict ---
    mapping_list = [-1] * N
    reverse_list = [-1] * N
    used_phys = set()

    for L, P in best_mapping.items():
        if 0 <= L < N and 0 <= P < N and reverse_list[P] == -1 and mapping_list[L] == -1:
            mapping_list[L] = P
            reverse_list[P] = L
            used_phys.add(P)

    # back-fill remaining logicals (idle ones) to most-central unused physicals
    remaining_phys = [p for p in range(N) if p not in used_phys]
    remaining_phys.sort(key=lambda p: -self.physical_centrality.get(p, 0.0))
    rp_idx = 0
    for L in range(N):
        if mapping_list[L] == -1:
            # prefer identity if free, else next central
            if L not in used_phys:
                mapping_list[L] = L
                reverse_list[L] = L
                used_phys.add(L)
                if L in remaining_phys:
                    remaining_phys.remove(L)
            else:
                while rp_idx < len(remaining_phys) and remaining_phys[rp_idx] in used_phys:
                    rp_idx += 1
                if rp_idx < len(remaining_phys):
                    p = remaining_phys[rp_idx]
                    rp_idx += 1
                    mapping_list[L] = p
                    reverse_list[p] = L
                    used_phys.add(p)

    # final safety: any leftover -1 slots
    leftover_phys = [p for p in range(N) if p not in used_phys]
    li = 0
    for L in range(N):
        if mapping_list[L] == -1:
            while li < len(leftover_phys) and leftover_phys[li] in used_phys:
                li += 1
            if li < len(leftover_phys):
                p = leftover_phys[li]
                li += 1
                mapping_list[L] = p
                reverse_list[p] = L
                used_phys.add(p)

    self.mapping_dict = mapping_list
    self.reverse_mapping_dict = reverse_list

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)