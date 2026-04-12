def init_mapping(self):
    import math
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build DAG, topological sort, critical path
    # ---------------------------------------------------------------
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in sorted(self.access.keys()):
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]
        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)
        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_order = []
    topo_rank = {}
    gate_layer = {g: 0 for g in all_gates}
    rank = 0
    temp_in = dict(in_degree)

    while queue:
        g = queue.popleft()
        topo_order.append(g)
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ---------------------------------------------------------------
    # Step 2: Build 2Q DAG for routing simulation
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)

    logical_qubits_set = set()
    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        static_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    last_2q_on_qubit = {}
    gates_2q = {}
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate

    topo_2q = []
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    q2 = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2:
        g = q2.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2.append(s)

    dep_count_2q = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count_2q[g] += dep_count_2q[s] + 1

    # Precompute which gates each logical qubit participates in (for delta eval)
    qubit_to_gates_2q = defaultdict(set)
    for g, (gq1, gq2) in gates_2q.items():
        qubit_to_gates_2q[gq1].add(g)
        qubit_to_gates_2q[gq2].add(g)

    # ---------------------------------------------------------------
    # Step 3: Physical graph properties
    # ---------------------------------------------------------------
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}

    # Seed selection
    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda qq: logical_degree[qq], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    # Degree matching
    logical_degree_ranked = sorted(interacting_logical, key=lambda qq: logical_degree[qq], reverse=True)
    logical_degree_rank = {qq: i for i, qq in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    # ---------------------------------------------------------------
    # Step 4: SABRE-Lite Router - the TRUE objective
    # ---------------------------------------------------------------
    def sabre_lite(m, rm, max_layers_K):
        """Run SABRE-style routing on first K layers, return SWAP count."""
        if not gates_2q:
            return 0

        sim_m = list(m)
        sim_rm = list(rm)
        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set()
        for g in gates_2q:
            if pred_remaining[g] == 0:
                front.add(g)

        swap_count = 0
        layers_done = 0
        sim_decay = [1.0] * num_q

        while front and layers_done < max_layers_K:
            # Execute ready gates
            executable = []
            for g in front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = sim_m[gq1], sim_m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executable.append(g)

            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag2q_succ[g]:
                        pred_remaining[s] -= 1
                        if pred_remaining[s] == 0:
                            front.add(s)
                layers_done += 1
                sim_decay = [1.0] * num_q
                continue

            # Build extended set for lookahead
            ext_set = set()
            ext_layer_idx = {}
            bfs_q = deque()
            for g in front:
                for s in dag2q_succ[g]:
                    if s not in front and s not in ext_set:
                        ext_set.add(s)
                        ext_layer_idx[s] = 1
                        bfs_q.append(s)
            while bfs_q and len(ext_set) < len(front) * 3:
                g = bfs_q.popleft()
                for s in dag2q_succ[g]:
                    if s not in front and s not in ext_set:
                        ext_set.add(s)
                        ext_layer_idx[s] = ext_layer_idx[g] + 1
                        bfs_q.append(s)

            # Find best SWAP
            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])

            candidates = set()
            for pq in active_phys:
                for nb in hw_adj[pq]:
                    candidates.add((min(pq, nb), max(pq, nb)))

            best_swap = None
            best_score = float('inf')

            for (s1, s2) in candidates:
                l1 = sim_rm[s1]
                l2 = sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])

                f_score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1 = sim_m[gq1]
                    p2 = sim_m[gq2]
                    if gq1 == l1:
                        p1 = s2
                    elif gq1 == l2:
                        p1 = s1
                    if gq2 == l1:
                        p2 = s2
                    elif gq2 == l2:
                        p2 = s1
                    deps = dep_count_2q.get(g, 0) + 1
                    f_score += deps * self.distance_matrix[p1][p2]

                e_score = 0.0
                for g in ext_set:
                    gq1, gq2 = gates_2q[g]
                    p1 = sim_m[gq1]
                    p2 = sim_m[gq2]
                    if gq1 == l1:
                        p1 = s2
                    elif gq1 == l2:
                        p1 = s1
                    if gq2 == l1:
                        p2 = s2
                    elif gq2 == l2:
                        p2 = s1
                    layer_f = ext_layer_idx.get(g, 1)
                    e_score += self.distance_matrix[p1][p2] / layer_f

                front_sz = max(len(front), 1)
                ext_sz = max(len(ext_set), 1)
                score = max_d * (f_score / front_sz + 0.5 * e_score / ext_sz)

                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1 = sim_rm[s1]
            l2 = sim_rm[s2]
            sim_m[l1] = s2
            sim_m[l2] = s1
            sim_rm[s1] = l2
            sim_rm[s2] = l1
            sim_decay[s1] += 0.001
            sim_decay[s2] += 0.001
            swap_count += 1

        return swap_count

    # ---------------------------------------------------------------
    # Step 5: Greedy construction with connectivity matching
    # ---------------------------------------------------------------
    def build_neighbors_from_weights(weights):
        neighbors = defaultdict(dict)
        for (q1, q2), w in weights.items():
            neighbors[q1][q2] = w
            neighbors[q2][q1] = w
        return neighbors

    logical_nbrs = build_neighbors_from_weights(static_weight)

    def run_greedy_placement(start_lq, start_pq):
        max_iw = max((w for nbrs in logical_nbrs.values() for w in nbrs.values()), default=1.0)

        def dyn_bonus(iw):
            return 0.90 - 0.10 * (iw / max_iw)

        used_phys = {start_pq}
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq = None
            best_w = -1.0
            for lq in remaining:
                w = sum(logical_nbrs[lq].get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: logical_nbrs[best_lq].get(plq, 0.0)
                           for plq in placed if plq in logical_nbrs[best_lq]}

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        dist = self.distance_matrix[pq][m[plq]]
                        cost = iw * dist
                        if m[plq] in hw_adj[pq]:
                            cost *= dyn_bonus(iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    cands = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(cands) > 1 and best_lq in logical_degree_rank:
                        lq_rank_norm = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(cands, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rank_norm), x[0]))[1]
                    else:
                        best_pq = min(cands, key=lambda x: x[0])[1]
                elif near_ties:
                    best_pq = min(near_ties, key=lambda x: x[0])[1]
                else:
                    best_pq = None
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        score = phys_centrality[pq]
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            m[best_lq] = best_pq
            rm[best_pq] = best_lq
            used_phys.add(best_pq)
            placed.add(best_lq)
            remaining.discard(best_lq)

        return m, rm

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    def compute_proxy_cost(m):
        cost = 0.0
        for (q1, q2), w in static_weight.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    # ---------------------------------------------------------------
    # Step 6: Local refinement (swap-based hill climbing using proxy for speed)
    # ---------------------------------------------------------------
    def run_swap_refinement(m, rm, max_rounds):
        if len(interacting_logical) <= 1:
            return
        improved = True
        round_count = 0
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            for i in range(len(interacting_logical)):
                for j in range(i + 1, len(interacting_logical)):
                    lq_a = interacting_logical[i]
                    lq_b = interacting_logical[j]
                    pq_a = m[lq_a]
                    pq_b = m[lq_b]
                    delta = 0.0
                    affected = set()
                    affected.update(logical_nbrs[lq_a].keys())
                    affected.update(logical_nbrs[lq_b].keys())
                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = m[q]
                        w_a = logical_nbrs[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                        w_b = logical_nbrs[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])
                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    # ---------------------------------------------------------------
    # Step 7: Multi-seed construction
    # ---------------------------------------------------------------
    greedy_candidates = []
    if seed_lqs and seed_pqs:
        for s_lq in seed_lqs:
            for s_pq in seed_pqs:
                m, rm = run_greedy_placement(s_lq, s_pq)
                fill_unmapped(m, rm)
                run_swap_refinement(m, rm, max_rounds=3)
                greedy_candidates.append((m, rm))

    if not greedy_candidates:
        m = list(range(num_q))
        rm = list(range(num_q))
        greedy_candidates.append((m, rm))

    # ---------------------------------------------------------------
    # Step 8: Evaluate greedy candidates with SABRE-Lite (true objective)
    # ---------------------------------------------------------------
    K_init = 15
    best_swaps = float('inf')
    best_m = None
    best_rm = None

    for m, rm in greedy_candidates:
        sc = sabre_lite(m, rm, K_init)
        if sc < best_swaps:
            best_swaps = sc
            best_m = list(m)
            best_rm = list(rm)

    # ---------------------------------------------------------------
    # Step 9: ILS + SA with SABRE-Lite objective & bandit-adaptive perturbation
    # ---------------------------------------------------------------
    if len(interacting_logical) >= 4 and gates_2q:
        import random as rng_mod
        rng_seed = int(best_swaps * 1000 + len(gates_2q)) % (2**31)
        rng_mod.seed(rng_seed)

        # Bandit: track which perturbation modes produce improvements
        num_modes = 4
        mode_successes = [1.0] * num_modes
        mode_trials = [1.0] * num_modes

        current_m = list(best_m)
        current_rm = list(best_rm)
        current_swaps = best_swaps

        # Adaptive K: start small, grow as search progresses
        num_ils_iters = min(12, max(6, len(interacting_logical) // 3))
        T_init = max(best_swaps * 0.15, 2.0)
        T_min = 0.5

        # Cache: store (tuple(m restricted to interacting)) -> swap_count
        cache = {}
        cache_key_fn = lambda mm: tuple(mm[q] for q in interacting_logical)
        cache[cache_key_fn(best_m)] = best_swaps

        for it in range(num_ils_iters):
            progress = it / max(num_ils_iters - 1, 1)
            T = T_init * (T_min / T_init) ** progress
            K = int(10 + 20 * progress)  # Adaptive: 10 -> 30

            # Bandit: select mode via UCB1
            total_trials = sum(mode_trials)
            ucb_scores = []
            for mi in range(num_modes):
                exploit = mode_successes[mi] / mode_trials[mi]
                explore = math.sqrt(2.0 * math.log(total_trials + 1) / mode_trials[mi])
                ucb_scores.append(exploit + explore)
            mode = ucb_scores.index(max(ucb_scores))

            m_try = list(current_m)
            rm_try = list(current_rm)

            if mode == 0:
                # Edge-targeted perturbation: swap qubits on highest-cost edge
                edge_costs = []
                for (eq1, eq2), w in static_weight.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                        edge_costs.append((c, eq1, eq2))
                edge_costs.sort(reverse=True)
                done = False
                for _, eq1, eq2 in edge_costs[:3]:
                    pq1, pq2 = m_try[eq1], m_try[eq2]
                    best_delta = 0.0
                    best_sp = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_dist = self.distance_matrix[adj_pq][pq2]
                        old_dist = self.distance_matrix[pq1][pq2]
                        if new_dist - old_dist < best_delta:
                            best_delta = new_dist - old_dist
                            best_sp = (eq1, occ, pq1, adj_pq)
                    if best_sp:
                        lq_a, lq_b, pq_a, pq_b = best_sp
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        done = True
                        break
                if not done:
                    idx = rng_mod.sample(range(len(interacting_logical)), 2)
                    lq_a, lq_b = interacting_logical[idx[0]], interacting_logical[idx[1]]
                    pq_a, pq_b = m_try[lq_a], m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            elif mode == 1:
                # Cost-driven neighborhood swap
                qcost = defaultdict(float)
                for (eq1, eq2), w in static_weight.items():
                    c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                    qcost[eq1] += c
                    qcost[eq2] += c
                cost_pairs = sorted(((qcost.get(q, 0.0), q) for q in interacting_logical), reverse=True)
                swapped = False
                for _, lq_a in cost_pairs[:3]:
                    pq_a = m_try[lq_a]
                    adj_cands = []
                    for adj_pq in hw_adj[pq_a]:
                        occ = rm_try[adj_pq]
                        if occ >= 0:
                            adj_cands.append((qcost.get(occ, 0.0), occ, adj_pq))
                    if adj_cands:
                        adj_cands.sort()
                        lq_b = adj_cands[0][1]
                        pq_b = adj_cands[0][2]
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        swapped = True
                        break
                if not swapped:
                    idx = rng_mod.sample(range(len(interacting_logical)), min(4, len(interacting_logical)))
                    for s in range(0, len(idx) - 1, 2):
                        lq_x, lq_y = interacting_logical[idx[s]], interacting_logical[idx[s + 1]]
                        pq_x, pq_y = m_try[lq_x], m_try[lq_y]
                        m_try[lq_x] = pq_y
                        m_try[lq_y] = pq_x
                        rm_try[pq_x] = lq_y
                        rm_try[pq_y] = lq_x

            elif mode == 2 and len(interacting_logical) >= 6:
                # 3-opt chain rotation
                qcost = defaultdict(float)
                for (eq1, eq2), w in static_weight.items():
                    c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                    qcost[eq1] += c
                    qcost[eq2] += c
                cost_pairs = sorted(((qcost.get(q, 0.0), q) for q in interacting_logical), reverse=True)
                top3 = [q for _, q in cost_pairs[:5]]
                if len(top3) >= 3:
                    sel = top3[:3]
                    pqs = [m_try[lq] for lq in sel]
                    m_try[sel[0]] = pqs[1]
                    m_try[sel[1]] = pqs[2]
                    m_try[sel[2]] = pqs[0]
                    rm_try[pqs[0]] = sel[2]
                    rm_try[pqs[1]] = sel[0]
                    rm_try[pqs[2]] = sel[1]
            else:
                # Random perturbation
                n_swaps = min(2, len(interacting_logical) // 2)
                idx = rng_mod.sample(range(len(interacting_logical)), min(2 * n_swaps, len(interacting_logical)))
                for s in range(0, len(idx) - 1, 2):
                    lq_a, lq_b = interacting_logical[idx[s]], interacting_logical[idx[s + 1]]
                    pq_a, pq_b = m_try[lq_a], m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            # Local refinement with proxy (fast)
            run_swap_refinement(m_try, rm_try, max_rounds=3)

            # Check cache
            ck = cache_key_fn(m_try)
            if ck in cache:
                try_swaps = cache[ck]
            else:
                try_swaps = sabre_lite(m_try, rm_try, K)
                cache[ck] = try_swaps

            # SA acceptance using TRUE swap count
            delta = try_swaps - current_swaps
            mode_trials[mode] += 1.0

            if delta < 0 or (T > 1e-12 and rng_mod.random() < math.exp(-delta / max(T, 1e-12))):
                current_m = list(m_try)
                current_rm = list(rm_try)
                current_swaps = try_swaps
                if delta < 0:
                    mode_successes[mode] += 1.0

            if try_swaps < best_swaps:
                best_swaps = try_swaps
                best_m = list(m_try)
                best_rm = list(rm_try)

    # ---------------------------------------------------------------
    # Step 10: Final RSDIWR feedback pass - use routing sim to refine weights
    # ---------------------------------------------------------------
    if gates_2q and len(interacting_logical) >= 2:
        # Run routing simulation to get swap counts per qubit pair
        sim_m = list(best_m)
        sim_rm = list(best_rm)
        swap_counts = defaultdict(float)
        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_remaining[g] == 0)
        sim_decay = [1.0] * num_q
        layers_done = 0

        while front and layers_done < 30:
            executable = []
            for g in front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = sim_m[gq1], sim_m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executable.append(g)
            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag2q_succ[g]:
                        pred_remaining[s] -= 1
                        if pred_remaining[s] == 0:
                            front.add(s)
                layers_done += 1
                sim_decay = [1.0] * num_q
                continue

            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])
            candidates = set()
            for pq in active_phys:
                for nb in hw_adj[pq]:
                    candidates.add((min(pq, nb), max(pq, nb)))

            best_swap = None
            best_score = float('inf')
            for (s1, s2) in candidates:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1 = sim_m[gq1]
                    p2 = sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = dep_count_2q.get(g, 0) + 1
                    score += deps * self.distance_matrix[p1][p2]
                score *= max_d
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break
            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1] = s2
            sim_m[l2] = s1
            sim_rm[s1] = l2
            sim_rm[s2] = l1
            sim_decay[s1] += 0.001
            sim_decay[s2] += 0.001

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        # Blend static weights with routing feedback
        if swap_counts:
            max_s = max(static_weight.values()) if static_weight else 1.0
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max_s / max(max_sw, 1e-10)

            eff_weights = defaultdict(float)
            all_keys = set(static_weight.keys()) | set(swap_counts.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = 0.4 * w_s + 0.6 * w_r

            eff_nbrs = build_neighbors_from_weights(eff_weights)

            # Rebuild with refined weights
            refine_candidates = []
            for s_lq in seed_lqs[:2]:
                for s_pq in seed_pqs[:2]:
                    m2 = [-1] * num_q
                    rm2 = [-1] * num_q
                    m2[s_lq] = s_pq
                    rm2[s_pq] = s_lq
                    used_phys = {s_pq}
                    placed = {s_lq}
                    remaining = set(logical_qubits) - placed

                    while remaining:
                        best_lq2 = max(remaining, key=lambda lq: sum(eff_nbrs[lq].get(plq, 0.0) for plq in placed))
                        nbrs_placed = {plq: eff_nbrs[best_lq2].get(plq, 0.0) for plq in placed if plq in eff_nbrs[best_lq2]}

                        if nbrs_placed:
                            best_pq2 = None
                            best_sc = float('inf')
                            for pq in physical_qubits:
                                if pq in used_phys:
                                    continue
                                sc2 = sum(iw * self.distance_matrix[pq][m2[plq]] for plq, iw in nbrs_placed.items())
                                if sc2 < best_sc:
                                    best_sc = sc2
                                    best_pq2 = pq
                        else:
                            best_pq2 = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq])

                        m2[best_lq2] = best_pq2
                        rm2[best_pq2] = best_lq2
                        used_phys.add(best_pq2)
                        placed.add(best_lq2)
                        remaining.discard(best_lq2)

                    fill_unmapped(m2, rm2)
                    run_swap_refinement(m2, rm2, max_rounds=3)
                    sc2 = sabre_lite(m2, rm2, 25)
                    refine_candidates.append((sc2, m2, rm2))

            if refine_candidates:
                refine_candidates.sort(key=lambda x: x[0])
                if refine_candidates[0][0] < best_swaps:
                    best_swaps = refine_candidates[0][0]
                    best_m = refine_candidates[0][1]
                    best_rm = refine_candidates[0][2]

    # ---------------------------------------------------------------
    # Step 11: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        from src.utils.python_to_isl import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)