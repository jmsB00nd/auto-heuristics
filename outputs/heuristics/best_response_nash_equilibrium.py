def init_mapping(self):
    import math
    from collections import defaultdict
    from src.mapping.initial_mapping import generate_structure_aware_initial_mapping

    N = self.num_qubits
    dist = self.distance_matrix

    active = set()
    interactions = []
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            interactions.append((a, b))
            active.add(a); active.add(b)
        elif len(qubits) == 1:
            active.add(qubits[0])

    w = defaultdict(lambda: defaultdict(float))
    if getattr(self, "qubit_interaction_graph", None):
        for a in self.qubit_interaction_graph:
            for b, c in self.qubit_interaction_graph[a].items():
                if a != b and c:
                    w[a][b] = float(c)
    else:
        for a, b in interactions:
            w[a][b] += 1.0
            w[b][a] += 1.0

    try:
        seed_map, seed_rev = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, N
        )
        pos = list(seed_map)
        rev = list(seed_rev)
    except Exception:
        pos = list(range(N))
        rev = list(range(N))

    if len(pos) != N:
        pos = (list(pos) + list(range(N)))[:N]
    if len(rev) != N:
        rev = (list(rev) + list(range(N)))[:N]

    used_phys = set(pos)
    free_phys = [p for p in range(N) if p not in used_phys]
    centrality = getattr(self, "physical_centrality", {}) or {}
    free_phys.sort(key=lambda p: -centrality.get(p, 0.0))
    seen = set()
    fixed_pos = []
    for L, p in enumerate(pos):
        if p in seen or p < 0 or p >= N:
            new_p = free_phys.pop() if free_phys else next(x for x in range(N) if x not in seen)
            fixed_pos.append(new_p)
            seen.add(new_p)
        else:
            fixed_pos.append(p)
            seen.add(p)
    pos = fixed_pos
    rev = [-1] * N
    for L, p in enumerate(pos):
        rev[p] = L

    def player_cost(L, p_L, pos_local):
        s = 0.0
        row = w.get(L)
        if not row:
            return 0.0
        for L2, weight in row.items():
            if L2 == L or L2 >= N:
                continue
            s += weight * dist[p_L][pos_local[L2]]
        return s

    movers = sorted(active, key=lambda L: -sum(w[L].values()) if L in w else 0.0)
    if not movers:
        movers = list(range(N))

    max_sweeps = max(4, int(math.log2(N + 2)) * 4)
    for _ in range(max_sweeps):
        improved = False
        for L in movers:
            if L >= N:
                continue
            p_L = pos[L]
            base = player_cost(L, p_L, pos)
            best_delta = 0.0
            best_target = None
            for p_T in range(N):
                if p_T == p_L:
                    continue
                L2 = rev[p_T]
                if L2 == -1 or L2 < 0 or L2 >= N:
                    new_cost_L = player_cost(L, p_T, pos)
                    delta = new_cost_L - base
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_target = (p_T, None)
                else:
                    if L2 == L:
                        continue
                    base_L2 = player_cost(L2, p_T, pos)
                    pos_swapped = list(pos)
                    pos_swapped[L] = p_T
                    pos_swapped[L2] = p_L
                    new_L = player_cost(L, p_T, pos_swapped)
                    new_L2 = player_cost(L2, p_L, pos_swapped)
                    delta = (new_L + new_L2) - (base + base_L2)
                    if delta < best_delta - 1e-12 and new_L < base - 1e-12:
                        best_delta = delta
                        best_target = (p_T, L2)
            if best_target is not None:
                p_T, L2 = best_target
                if L2 is None:
                    rev[p_L] = -1
                    pos[L] = p_T
                    rev[p_T] = L
                else:
                    pos[L], pos[L2] = p_T, p_L
                    rev[p_T], rev[p_L] = L, L2
                improved = True
        if not improved:
            break

    occupied = set()
    for L in range(N):
        p = pos[L]
        if p < 0 or p >= N or p in occupied:
            for q in range(N):
                if q not in occupied:
                    pos[L] = q
                    occupied.add(q)
                    break
        else:
            occupied.add(p)

    self.mapping_dict = list(pos)
    self.reverse_mapping_dict = [0] * N
    for L, p in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)