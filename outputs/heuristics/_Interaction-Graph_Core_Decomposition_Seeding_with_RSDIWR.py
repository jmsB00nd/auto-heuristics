def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build DAG and compute critical-path weights
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

    # Kahn's topological sort + layer assignment
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
    # Step 2: Build critical-path-weighted interaction graph
    # ---------------------------------------------------------------
    alpha_decay = 2.0
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

    if len(logical_qubits) <= 1:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Build adjacency for interaction graph
    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in static_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # ---------------------------------------------------------------
    # Step 3: K-core decomposition of interaction graph (weighted)
    # ---------------------------------------------------------------
    # Iterative peeling: remove vertex with minimum weighted degree,
    # record core number k(v) for each vertex
    remaining_verts = set(interacting_logical)
    current_degree = {q: logical_degree_static.get(q, 0.0) for q in interacting_logical}
    logical_core_number = {}

    while remaining_verts:
        # Find vertex with minimum current weighted degree
        min_v = min(remaining_verts, key=lambda v: current_degree[v])
        core_val = current_degree[min_v]
        logical_core_number[min_v] = core_val

        # Remove min_v and update neighbors' degrees
        remaining_verts.discard(min_v)
        for nb, w in logical_neighbors.get(min_v, {}).items():
            if nb in remaining_verts:
                current_degree[nb] -= w

    # Assign non-interacting qubits core number 0
    for q in logical_qubits:
        if q not in logical_core_number:
            logical_core_number[q] = 0.0

    # ---------------------------------------------------------------
    # Step 4: K-core decomposition of hardware graph (unweighted)
    # ---------------------------------------------------------------
    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    hw_remaining = set(physical_qubits)
    hw_current_deg = {pq: len(hw_adj[pq]) for pq in physical_qubits}
    hw_core_number = {}

    while hw_remaining:
        min_pq = min(hw_remaining, key=lambda v: hw_current_deg[v])
        core_val = hw_current_deg[min_pq]
        hw_core_number[min_pq] = core_val

        hw_remaining.discard(min_pq)
        for nb in hw_adj[min_pq]:
            if nb in hw_remaining:
                hw_current_deg[nb] -= 1

    # ---------------------------------------------------------------
    # Step 5: Shell-by-shell placement (high core to low core)
    # ---------------------------------------------------------------
    # Discretize logical core numbers into shells
    # Sort interacting qubits by core number descending, then by weighted degree descending
    sorted_logical_by_core = sorted(
        interacting_logical,
        key=lambda q: (logical_core_number.get(q, 0), logical_degree_static.get(q, 0)),
        reverse=True
    )

    # Sort physical qubits by core number descending, then by degree descending
    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}
    sorted_phys_by_core = sorted(
        physical_qubits,
        key=lambda pq: (hw_core_number.get(pq, 0), phys_degree.get(pq, 0)),
        reverse=True
    )

    # Precompute centrality for fallback
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    dist = self.distance_matrix

    # Place interacting qubits shell-by-shell with greedy BFS adjacency bonus
    m = [-1] * num_q
    rm = [-1] * num_q
    used_phys = set()
    placed = set()

    # Place first (highest core) logical qubit on highest core physical qubit
    if sorted_logical_by_core and sorted_phys_by_core:
        first_lq = sorted_logical_by_core[0]
        first_pq = sorted_phys_by_core[0]
        m[first_lq] = first_pq
        rm[first_pq] = first_lq
        used_phys.add(first_pq)
        placed.add(first_lq)

    # Place remaining interacting qubits in core-order with adjacency-aware greedy
    for lq in sorted_logical_by_core[1:]:
        # Score each free physical qubit
        nbrs_placed = {plq: logical_neighbors[lq].get(plq, 0.0)
                       for plq in placed if plq in logical_neighbors.get(lq, {})}

        if nbrs_placed:
            best_pq = None
            best_score = float('inf')
            for pq in sorted_phys_by_core:
                if pq in used_phys:
                    continue
                score = 0.0
                for plq, iw in nbrs_placed.items():
                    d = dist[pq][m[plq]]
                    cost = iw * d
                    # Adjacency bonus: reduce cost if physically adjacent
                    if m[plq] in hw_adj[pq]:
                        cost *= 0.85
                    score += cost
                # Slight preference for higher-core physical qubits (already sorted)
                if score < best_score:
                    best_score = score
                    best_pq = pq
        else:
            # No placed neighbors; pick by centrality among high-core physical qubits
            best_pq = None
            best_score = float('inf')
            for pq in sorted_phys_by_core:
                if pq not in used_phys:
                    score = phys_centrality[pq]
                    if score < best_score:
                        best_score = score
                        best_pq = pq

        if best_pq is not None:
            m[lq] = best_pq
            rm[best_pq] = lq
            used_phys.add(best_pq)
            placed.add(lq)

    # Place non-interacting logical qubits
    non_interacting = [q for q in logical_qubits if q not in placed]
    for lq in non_interacting:
        for pq in sorted_phys_by_core:
            if pq not in used_phys:
                m[lq] = pq
                rm[pq] = lq
                used_phys.add(pq)
                placed.add(lq)
                break

    # Fill any remaining unmapped (logical qubits not in logical_qubits_set)
    unmapped_lq = [q for q in range(num_q) if m[q] == -1]
    free_pq = [pq for pq in range(num_q) if rm[pq] == -1]
    for lq, pq in zip(unmapped_lq, free_pq):
        m[lq] = pq
        rm[pq] = lq

    # ---------------------------------------------------------------
    # Step 6: Swap refinement helper
    # ---------------------------------------------------------------
    def compute_total_cost(mc, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if mc[q1] >= 0 and mc[q2] >= 0:
                cost += w * dist[mc[q1]][mc[q2]]
        return cost

    def run_swap_refinement(mc, rmc, nbrs, max_rounds):
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
                    pq_a = mc[lq_a]
                    pq_b = mc[lq_b]
                    delta = 0.0
                    affected = set()
                    affected.update(nbrs.get(lq_a, {}).keys())
                    affected.update(nbrs.get(lq_b, {}).keys())
                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = mc[q]
                        w_a = nbrs.get(lq_a, {}).get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (dist[pq_b][pq_q] - dist[pq_a][pq_q])
                        w_b = nbrs.get(lq_b, {}).get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (dist[pq_a][pq_q] - dist[pq_b][pq_q])
                    if delta < -1e-12:
                        mc[lq_a] = pq_b
                        mc[lq_b] = pq_a
                        rmc[pq_a] = lq_b
                        rmc[pq_b] = lq_a
                        improved = True

    def perturb_and_refine(mc, rmc, weights, nbrs, num_perturbations=5):
        if len(interacting_logical) < 4:
            return mc, rmc
        best_m = list(mc)
        best_rm = list(rmc)
        best_cost = compute_total_cost(mc, weights)
        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        for p_idx in range(num_perturbations):
            m_try = list(best_m)
            rm_try = list(best_rm)
            mode = p_idx % 3

            if mode == 0:
                # Edge-targeted perturbation
                edge_costs = []
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * dist[m_try[eq1]][m_try[eq2]]
                        edge_costs.append((c, eq1, eq2))
                edge_costs.sort(reverse=True)
                for _, eq1, eq2 in edge_costs[:3]:
                    pq1, pq2 = m_try[eq1], m_try[eq2]
                    current_d = dist[pq1][pq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_d = dist[adj_pq][pq2]
                        dd = new_d - current_d
                        if dd < best_delta:
                            best_delta = dd
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    if best_swap_pair:
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

            elif mode == 1:
                # Neighborhood-aware swap
                qcost = defaultdict(float)
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * dist[m_try[eq1]][m_try[eq2]]
                        qcost[eq1] += c
                        qcost[eq2] += c
                cost_pairs = sorted([(qcost.get(q, 0.0), q) for q in interacting_logical], reverse=True)
                swapped = False
                for _, lq_a in cost_pairs[:3]:
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
                    indices = rng.choice(len(interacting_logical), size=min(4, len(interacting_logical)), replace=False)
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
                indices = rng.choice(len(interacting_logical), size=min(2 * n_swaps, len(interacting_logical)), replace=False)
                for s in range(0, len(indices) - 1, 2):
                    lq_a = interacting_logical[indices[s]]
                    lq_b = interacting_logical[indices[s + 1]]
                    pq_a, pq_b = m_try[lq_a], m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            run_swap_refinement(m_try, rm_try, nbrs, max_rounds=3)
            cost = compute_total_cost(m_try, weights)
            if cost < best_cost:
                best_cost = cost
                best_m = m_try
                best_rm = rm_try

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 7: Build 2q-only DAG for routing simulation
    # ---------------------------------------------------------------
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

    # Dependency counts via topological order
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

    # ---------------------------------------------------------------
    # Step 8: Routing simulation with progressive depth
    # ---------------------------------------------------------------
    def simulate_routing(mc, rmc, max_layers=20):
        sim_m = list(mc)
        sim_rm = list(rmc)
        swap_counts = defaultdict(float)
        if not gates_2q:
            return swap_counts

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
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
                    deps = dep_count_2q.get(g, 0) + 1
                    score += deps * dist[p1][p2]
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
    # Step 9: Greedy placement builder parameterized by weights
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

    def run_greedy_placement(start_lq, start_pq, logical_nbrs):
        used_p = {start_pq}
        mc = [-1] * num_q
        rmc = [-1] * num_q
        mc[start_lq] = start_pq
        rmc[start_pq] = start_lq
        pl = {start_lq}
        remaining = set(logical_qubits) - pl

        while remaining:
            best_lq = None
            best_w = -1.0
            for lq in remaining:
                w = sum(logical_nbrs[lq].get(plq, 0.0) for plq in pl)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_pl = {plq: logical_nbrs[best_lq].get(plq, 0.0)
                        for plq in pl if plq in logical_nbrs[best_lq]}

            if nbrs_pl:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_p:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_pl.items():
                        d = dist[pq][mc[plq]]
                        cost = iw * d
                        if mc[plq] in hw_adj[pq]:
                            cost *= 0.85
                        score += cost
                    if score < best_score:
                        best_score = score
                        best_pq = pq
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_p:
                        score = phys_centrality[pq]
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            mc[best_lq] = best_pq
            rmc[best_pq] = best_lq
            used_p.add(best_pq)
            pl.add(best_lq)
            remaining.discard(best_lq)

        return mc, rmc

    def fill_unmapped(mc, rmc):
        unmapped = [q for q in range(num_q) if mc[q] == -1]
        free = [pq for pq in range(num_q) if rmc[pq] == -1]
        for lq, pq in zip(unmapped, free):
            mc[lq] = pq
            rmc[pq] = lq

    def build_best_mapping(eff_weights):
        eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)

        # Use top-3 seeds from core decomposition
        seed_lqs = sorted_logical_by_core[:min(3, len(sorted_logical_by_core))]
        seed_pqs = sorted_phys_by_core[:min(3, len(sorted_phys_by_core))]

        candidates = []
        if seed_lqs and seed_pqs:
            for s_lq in seed_lqs:
                for s_pq in seed_pqs:
                    mc, rmc = run_greedy_placement(s_lq, s_pq, eff_nbrs)
                    fill_unmapped(mc, rmc)
                    cost = compute_total_cost(mc, eff_weights)
                    candidates.append((cost, mc, rmc))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            num_to_refine = min(4, len(candidates))
            best_cost = float('inf')
            best_mc = None
            best_rmc = None

            for idx in range(num_to_refine):
                _, mc, rmc = candidates[idx]
                mc_c = list(mc)
                rmc_c = list(rmc)
                run_swap_refinement(mc_c, rmc_c, eff_nbrs, max_rounds=4)
                mc_c, rmc_c = perturb_and_refine(mc_c, rmc_c, eff_weights, eff_nbrs, num_perturbations=5)
                cost = compute_total_cost(mc_c, eff_weights)
                if cost < best_cost:
                    best_cost = cost
                    best_mc = mc_c
                    best_rmc = rmc_c

            return best_mc, best_rmc
        else:
            return list(range(num_q)), list(range(num_q))

    # ---------------------------------------------------------------
    # Step 10: RSDIWR outer loop with progressive simulation depth
    # ---------------------------------------------------------------
    T = 3
    alpha_schedule = [1.0, 0.6, 0.3]
    depth_schedule = [15, 25, 35]
    swap_counts = defaultdict(float)

    # Use core-decomposition seeded initial placement as baseline
    run_swap_refinement(m, rm, logical_neighbors, max_rounds=4)
    m, rm = perturb_and_refine(m, rm, static_weight, logical_neighbors, num_perturbations=5)

    best_overall_m = list(m)
    best_overall_rm = list(rm)
    best_overall_cost = compute_total_cost(m, static_weight)

    for t in range(T):
        alpha_blend = alpha_schedule[t]

        eff_weights = defaultdict(float)
        if t == 0 or not swap_counts:
            for key, w in static_weight.items():
                eff_weights[key] = w
        else:
            max_static = max(static_weight.values()) if static_weight else 1.0
            max_swap = max(swap_counts.values()) if swap_counts else 1.0
            scale = max_static / max(max_swap, 1e-10)
            all_keys = set(static_weight.keys()) | set(swap_counts.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        cur_m, cur_rm = build_best_mapping(eff_weights)

        cost_static = compute_total_cost(cur_m, static_weight)
        if cost_static < best_overall_cost:
            best_overall_cost = cost_static
            best_overall_m = list(cur_m)
            best_overall_rm = list(cur_rm)

        if t < T - 1:
            swap_counts = simulate_routing(cur_m, cur_rm, max_layers=depth_schedule[t])

    # ---------------------------------------------------------------
    # Step 11: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)