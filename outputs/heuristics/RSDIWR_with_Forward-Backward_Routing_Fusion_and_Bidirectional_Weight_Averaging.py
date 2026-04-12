def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build DAG and compute topological rank for temporal decay
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

    # Kahn's topological sort
    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_rank = {}
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    # ---------------------------------------------------------------
    # Step 2: Build STATIC temporal-decay weighted interaction graph
    # ---------------------------------------------------------------
    alpha_decay = 2.5
    logical_qubits_set = set()
    static_weight = defaultdict(float)
    logical_degree_static = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha_decay * r / total_gates)
            static_weight[key] += w
            logical_degree_static[q1] += w
            logical_degree_static[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree_static.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Precompute physical graph properties
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
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree_static[q], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree_static[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    # ---------------------------------------------------------------
    # Step 4: Build FORWARD and BACKWARD 2q-only DAGs
    # ---------------------------------------------------------------
    last_2q_on_qubit = {}
    gates_2q = {}
    dag2q_fwd_succ = defaultdict(set)
    dag2q_fwd_pred = defaultdict(set)

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_fwd_succ[prev].add(gate)
                        dag2q_fwd_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate

    # Backward DAG is just the reverse of the forward DAG
    dag2q_bwd_succ = dag2q_fwd_pred  # backward successors = forward predecessors
    dag2q_bwd_pred = dag2q_fwd_succ  # backward predecessors = forward successors

    # Forward dependency counts (transitive successors in forward direction)
    topo_fwd = []
    in_deg_fwd = {g: len(dag2q_fwd_pred[g]) for g in gates_2q}
    q_fwd = deque(sorted(g for g in gates_2q if in_deg_fwd[g] == 0))
    while q_fwd:
        g = q_fwd.popleft()
        topo_fwd.append(g)
        for s in dag2q_fwd_succ[g]:
            in_deg_fwd[s] -= 1
            if in_deg_fwd[s] == 0:
                q_fwd.append(s)

    dep_count_fwd = defaultdict(int)
    for g in reversed(topo_fwd):
        for s in dag2q_fwd_succ[g]:
            dep_count_fwd[g] += dep_count_fwd[s] + 1

    # Backward dependency counts (transitive successors in backward direction)
    dep_count_bwd = defaultdict(int)
    for g in topo_fwd:  # forward topo order = reverse of backward topo order
        for s in dag2q_bwd_succ[g]:
            dep_count_bwd[g] += dep_count_bwd[s] + 1

    # ---------------------------------------------------------------
    # Step 5: Helper functions
    # ---------------------------------------------------------------
    def build_neighbors_from_weights(weights):
        neighbors = defaultdict(dict)
        degree = defaultdict(float)
        for (q1, q2), w in weights.items():
            neighbors[q1][q2] = w
            neighbors[q2][q1] = w
            degree[q1] += w
            degree[q2] += w
        return neighbors, degree

    def run_greedy_placement(start_lq, start_pq, logical_nbrs, eff_degree):
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
                    candidates = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates) > 1 and best_lq in logical_degree_rank:
                        lq_rank_norm = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(candidates, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rank_norm),
                            x[0]
                        ))[1]
                    else:
                        best_pq = min(candidates, key=lambda x: x[0])[1]
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

    def compute_total_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    def run_swap_refinement(m, rm, logical_nbrs, max_rounds):
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

    def perturb_and_refine(m, rm, weights, logical_nbrs, num_perturbations=5):
        if len(interacting_logical) < 4:
            return m, rm
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m, weights)

        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        def compute_edge_costs(mc):
            edge_costs = []
            for (eq1, eq2), w in weights.items():
                if mc[eq1] >= 0 and mc[eq2] >= 0:
                    c = w * self.distance_matrix[mc[eq1]][mc[eq2]]
                    edge_costs.append((c, eq1, eq2))
            edge_costs.sort(reverse=True)
            return edge_costs

        for p_idx in range(num_perturbations):
            m_try = list(best_m)
            rm_try = list(best_rm)
            mode = p_idx % 3

            if mode == 0 and len(interacting_logical) >= 4:
                edge_costs = compute_edge_costs(m_try)
                for _, eq1, eq2 in edge_costs[:min(3, len(edge_costs))]:
                    pq1 = m_try[eq1]
                    pq2 = m_try[eq2]
                    current_dist = self.distance_matrix[pq1][pq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_dist = self.distance_matrix[adj_pq][pq2]
                        delta = new_dist - current_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    for adj_pq in hw_adj[pq2]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq1:
                            continue
                        new_dist = self.distance_matrix[pq1][adj_pq]
                        delta = new_dist - current_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_swap_pair = (eq2, occ, pq2, adj_pq)
                    if best_swap_pair is not None:
                        lq_a, lq_b, pq_a, pq_b = best_swap_pair
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        break
                indices = rng.choice(len(interacting_logical), size=2, replace=False)
                lq_c = interacting_logical[indices[0]]
                lq_d = interacting_logical[indices[1]]
                pq_c, pq_d = m_try[lq_c], m_try[lq_d]
                m_try[lq_c] = pq_d
                m_try[lq_d] = pq_c
                rm_try[pq_c] = lq_d
                rm_try[pq_d] = lq_c

            elif mode == 1 and len(interacting_logical) >= 4:
                qcost = defaultdict(float)
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                        qcost[eq1] += c
                        qcost[eq2] += c
                cost_pairs = [(qcost.get(q, 0.0), q) for q in interacting_logical]
                cost_pairs.sort(reverse=True)
                swapped = False
                for _, lq_a in cost_pairs[:min(3, len(cost_pairs))]:
                    pq_a = m_try[lq_a]
                    adj_candidates = []
                    for adj_pq in hw_adj[pq_a]:
                        occ = rm_try[adj_pq]
                        if occ >= 0:
                            adj_candidates.append((qcost.get(occ, 0.0), occ, adj_pq))
                    if adj_candidates:
                        adj_candidates.sort()
                        lq_b = adj_candidates[0][1]
                        pq_b = adj_candidates[0][2]
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        swapped = True
                        break
                if not swapped:
                    indices = rng.choice(len(interacting_logical),
                                         size=min(4, len(interacting_logical)),
                                         replace=False)
                    for s in range(0, len(indices) - 1, 2):
                        lq_x = interacting_logical[indices[s]]
                        lq_y = interacting_logical[indices[s + 1]]
                        pq_x, pq_y = m_try[lq_x], m_try[lq_y]
                        m_try[lq_x] = pq_y
                        m_try[lq_y] = pq_x
                        rm_try[pq_x] = lq_y
                        rm_try[pq_y] = lq_x

            else:
                n_swaps = min(2, len(interacting_logical) // 2)
                indices = rng.choice(len(interacting_logical),
                                     size=min(2 * n_swaps, len(interacting_logical)),
                                     replace=False)
                for s in range(0, len(indices) - 1, 2):
                    lq_a = interacting_logical[indices[s]]
                    lq_b = interacting_logical[indices[s + 1]]
                    pq_a = m_try[lq_a]
                    pq_b = m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            run_swap_refinement(m_try, rm_try, logical_nbrs, max_rounds=3)
            cost = compute_total_cost(m_try, weights)
            if cost < best_cost:
                best_cost = cost
                best_m = m_try
                best_rm = rm_try

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 6: Bidirectional routing simulation
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, dag_succ, dag_pred, dep_count, max_layers=20):
        """Run routing simulation in a given DAG direction."""
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)

        if not gates_2q:
            return swap_counts

        pred_remaining = {g: len(dag_pred[g]) for g in gates_2q}
        front = set()
        for g in gates_2q:
            if pred_remaining[g] == 0:
                front.add(g)

        layers_done = 0
        sim_decay = [1.0] * num_q

        while front and layers_done < max_layers:
            executable = []
            for g in front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = sim_m[gq1], sim_m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executable.append(g)

            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag_succ[g]:
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
                for nb in self.backend.get(pq, []):
                    candidates.add((min(pq, nb), max(pq, nb)))

            best_swap = None
            best_score = float('inf')

            for (s1, s2) in candidates:
                l1 = sim_rm[s1]
                l2 = sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])

                score = 0.0
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

                    deps = dep_count.get(g, 0) + 1
                    score += deps * self.distance_matrix[p1][p2]

                score *= max_d
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

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts

    # ---------------------------------------------------------------
    # Step 7: Build mapping with a given effective weight dictionary
    # ---------------------------------------------------------------
    def build_best_mapping(eff_weights):
        eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)

        candidates = []
        if seed_lqs and seed_pqs:
            for s_lq in seed_lqs:
                for s_pq in seed_pqs:
                    m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs, eff_deg)
                    fill_unmapped(m, rm)
                    cost = compute_total_cost(m, eff_weights)
                    candidates.append((cost, m, rm))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            num_to_refine = min(4, len(candidates))

            best_cost = float('inf')
            best_m = None
            best_rm = None

            for idx in range(num_to_refine):
                _, m, rm = candidates[idx]
                m_c = list(m)
                rm_c = list(rm)

                run_swap_refinement(m_c, rm_c, eff_nbrs, max_rounds=4)
                m_c, rm_c = perturb_and_refine(m_c, rm_c, eff_weights, eff_nbrs, num_perturbations=5)

                cost = compute_total_cost(m_c, eff_weights)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m_c
                    best_rm = rm_c

            return best_m, best_rm
        else:
            return list(range(num_q)), list(range(num_q))

    # ---------------------------------------------------------------
    # Step 8: RSDIWR with Forward-Backward Fusion
    #   T=3 iterations, α schedule for static/routing blend,
    #   λ schedule for forward/backward fusion (0.8 → 0.5)
    # ---------------------------------------------------------------
    T = 3
    alpha_schedule = [1.0, 0.65, 0.3]
    # λ decays from 0.8 (trust forward more) to 0.5 (equal)
    lambda_schedule = [0.8, 0.65, 0.5]
    swap_counts_fused = defaultdict(float)

    best_overall_m = None
    best_overall_rm = None
    best_overall_cost = float('inf')

    for t in range(T):
        alpha_blend = alpha_schedule[t]

        # Compute effective weights: blend static + fused swap counts
        eff_weights = defaultdict(float)

        if t == 0 or not swap_counts_fused:
            for key, w in static_weight.items():
                eff_weights[key] = w
        else:
            max_static = max(static_weight.values()) if static_weight else 1.0
            max_swap = max(swap_counts_fused.values()) if swap_counts_fused else 1.0
            scale = max_static / max(max_swap, 1e-10)

            all_keys = set(static_weight.keys()) | set(swap_counts_fused.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = swap_counts_fused.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        # Build mapping with effective weights
        cur_m, cur_rm = build_best_mapping(eff_weights)

        # Evaluate with static weights for fair comparison
        cost_static = compute_total_cost(cur_m, static_weight)
        if cost_static < best_overall_cost:
            best_overall_cost = cost_static
            best_overall_m = list(cur_m)
            best_overall_rm = list(cur_rm)

        # Bidirectional routing simulation (skip on last iteration)
        if t < T - 1:
            lam = lambda_schedule[t]

            # Forward routing simulation
            swap_fwd = simulate_routing(cur_m, cur_rm,
                                        dag2q_fwd_succ, dag2q_fwd_pred,
                                        dep_count_fwd, max_layers=20)

            # Backward routing simulation (reverse DAG direction)
            swap_bwd = simulate_routing(cur_m, cur_rm,
                                        dag2q_bwd_succ, dag2q_bwd_pred,
                                        dep_count_bwd, max_layers=20)

            # Fuse: swap_fused = λ·swap_fwd + (1-λ)·swap_bwd
            swap_counts_fused = defaultdict(float)
            all_pair_keys = set(swap_fwd.keys()) | set(swap_bwd.keys())
            for key in all_pair_keys:
                swap_counts_fused[key] = lam * swap_fwd.get(key, 0.0) + (1.0 - lam) * swap_bwd.get(key, 0.0)

    # ---------------------------------------------------------------
    # Step 9: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)