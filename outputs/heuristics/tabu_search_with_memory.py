def init_mapping(self):
    import math
    import random
    from collections import deque, defaultdict

    N = self.num_qubits

    # ----- collect logical 2-qubit interactions with weights -----
    pair_weight = defaultdict(int)
    logical_qubits_in_use = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_qubits_in_use.add(a)
            logical_qubits_in_use.add(b)
            key = (a, b) if a < b else (b, a)
            # prefer the prebuilt QIG weight if present, else count gates
            w = self.qubit_interaction_graph[a].get(b, 0) if hasattr(self, "qubit_interaction_graph") else 0
            pair_weight[key] = w if w > 0 else pair_weight[key] + 1
    interactions = [(a, b, w) for (a, b), w in pair_weight.items()]

    # ----- warm-start mapping -----
    mapping = list(range(N))
    reverse = list(range(N))
    try:
        from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
        m0, r0 = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, N
        )
        if m0 is not None and len(m0) == N and len(set(m0)) == N:
            mapping = list(m0)
            reverse = list(r0)
    except Exception:
        pass

    D = self.distance_matrix

    def cost_of(mp):
        s = 0
        for a, b, w in interactions:
            s += w * D[mp[a]][mp[b]]
        return s

    def delta_swap(mp, la, lb):
        # cost change if we swap physical assignments of logical la and lb
        pa, pb = mp[la], mp[lb]
        before = 0
        after = 0
        for a, b, w in interactions:
            if a == la or b == la or a == lb or b == lb:
                pA, pB = mp[a], mp[b]
                before += w * D[pA][pB]
                nA = pb if a == la else (pa if a == lb else pA)
                nB = pb if b == la else (pa if b == lb else pB)
                after += w * D[nA][nB]
        return after - before

    # ----- candidate logical-pair moves derived from physically-adjacent swaps -----
    # For each physical edge (p, q): swapping logical(p) with logical(q) is a unit-distance move.
    phys_edges = set()
    for p in range(N):
        for q in self.backend.get(p, ()):
            if p < q:
                phys_edges.add((p, q))
    # Also include random logical-pair moves for diversification.
    active_logicals = sorted(logical_qubits_in_use) if logical_qubits_in_use else list(range(N))

    def neighbor_moves():
        moves = set()
        for (p, q) in phys_edges:
            la, lb = reverse[p], reverse[q]
            if la != lb:
                moves.add((la, lb) if la < lb else (lb, la))
        # a few random long-range moves to escape plateaus
        if len(active_logicals) >= 2:
            for _ in range(min(2 * N, 64)):
                la, lb = random.sample(active_logicals, 2)
                moves.add((la, lb) if la < lb else (lb, la))
        return list(moves)

    # ----- tabu search loop -----
    if not interactions:
        # nothing to optimize; commit current mapping
        self.mapping_dict = list(mapping)
        self.reverse_mapping_dict = list(reverse)
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    rng = random.Random(0xC0FFEE)
    random.seed(0xC0FFEE)

    cur_cost = cost_of(mapping)
    best_mapping = list(mapping)
    best_cost = cur_cost

    tabu_tenure = max(4, int(math.sqrt(max(N, 1)) * 2))
    tabu = deque(maxlen=tabu_tenure)
    tabu_set = set()

    max_iters = max(40, 4 * N)
    no_improve = 0
    plateau_limit = max(20, N)

    for it in range(max_iters):
        moves = neighbor_moves()
        if not moves:
            break
        best_move = None
        best_move_delta = None
        best_was_tabu = False
        # scan neighborhood
        for (la, lb) in moves:
            d = delta_swap(mapping, la, lb)
            key = (la, lb)
            is_tabu = key in tabu_set
            new_cost = cur_cost + d
            # aspiration: tabu allowed if it would beat global best
            if is_tabu and not (new_cost < best_cost):
                continue
            if best_move_delta is None or d < best_move_delta:
                best_move_delta = d
                best_move = (la, lb)
                best_was_tabu = is_tabu

        if best_move is None:
            # all moves were tabu and none satisfied aspiration -> clear oldest
            if tabu:
                old = tabu.popleft()
                tabu_set.discard(old)
                continue
            else:
                break

        la, lb = best_move
        pa, pb = mapping[la], mapping[lb]
        mapping[la], mapping[lb] = pb, pa
        reverse[pa], reverse[pb] = lb, la
        cur_cost += best_move_delta

        # record tabu (reverse-move equivalent is the same unordered pair)
        if len(tabu) == tabu.maxlen and tabu:
            old = tabu[0]
            tabu_set.discard(old)
        tabu.append(best_move)
        tabu_set.add(best_move)

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_mapping = list(mapping)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= plateau_limit:
            # diversify: jump back to best and perturb tenure
            mapping = list(best_mapping)
            reverse = [0] * N
            for l, p in enumerate(mapping):
                reverse[p] = l
            cur_cost = best_cost
            tabu_tenure = max(4, int(math.sqrt(max(N, 1)) * 3))
            tabu = deque(maxlen=tabu_tenure)
            tabu_set = set()
            no_improve = 0
            # random kick
            if len(active_logicals) >= 2:
                for _ in range(2):
                    la, lb = rng.sample(active_logicals, 2)
                    pa, pb = mapping[la], mapping[lb]
                    mapping[la], mapping[lb] = pb, pa
                    reverse[pa], reverse[pb] = lb, la
                cur_cost = cost_of(mapping)

    # commit best-known mapping
    self.mapping_dict = list(best_mapping)
    self.reverse_mapping_dict = [0] * N
    for l, p in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[p] = l
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)