def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque
    from time import time as _time

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # ---------------------------------------------------------------
    # Step 1: Build DAG and compute topological rank
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
    # Step 2: Build 2q DAG and interaction weights
    # ---------------------------------------------------------------
    gates_2q = {}
    logical_qubits_set = set()
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)
    last_2q_on_qubit = {}

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)

    # Topological order for 2q DAG
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

    # Critical-path: forward dependency count
    dep_count = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count[g] += dep_count[s] + 1

    # Temporal-decay + critical-path base weights
    alpha_decay = 2.5
    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, (gq1, gq2) in gates_2q.items():
        key = (min(gq1, gq2), max(gq1, gq2))
        r = topo_rank.get(gate, 0)
        temporal_w = math.exp(-alpha_decay * r / total_gates)
        cp_w = dep_count.get(gate, 0) + 1
        w = temporal_w * cp_w
        static_weight[key] += w
        logical_degree[gq1] += w
        logical_degree[gq2] += w

    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    if not gates_2q:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ---------------------------------------------------------------
    # Step 3: Physical graph properties
    # ---------------------------------------------------------------
    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(dist[pq][pq2] for pq2 in physical_qubits)

    # ---------------------------------------------------------------
    # Step 4: Helper functions
    # ---------------------------------------------------------------
    def build_neighbors(weights):
        nbrs = defaultdict(dict)
        deg = defaultdict(float)
        for (q1, q2), w in weights.items():
            nbrs[q1][q2] = w
            nbrs[q2][q1] = w
            deg[q1] += w
            deg[q2] += w
        return nbrs, deg

    def compute_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * dist[m[q1]][m[q2]]
        return cost

    def delta_swap_cost(m, rm, pq_a, pq_b, nbrs):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        delta = 0.0
        affected = set()
        if lq_a in nbrs:
            affected.update(nbrs[lq_a].keys())
        if lq_b in nbrs:
            affected.update(nbrs[lq_b].keys())
        for q in affected:
            if q == lq_a or q == lq_b:
                continue
            pq_q = m[q]
            w_a = nbrs.get(lq_a, {}).get(q, 0.0)
            if w_a > 0:
                delta += w_a * (dist[pq_b][pq_q] - dist[pq_a][pq_q])
            w_b = nbrs.get(lq_b, {}).get(q, 0.0)
            if w_b > 0:
                delta += w_b * (dist[pq_a][pq_q] - dist[pq_b][pq_q])
        return delta

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        m[lq_a], m[lq_b] = pq_b, pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    # ---------------------------------------------------------------
    # Step 5: Greedy initial placement (multi-seed)
    # ---------------------------------------------------------------
    sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)
    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    max_iw = max(static_weight.values()) if static_weight else 1.0

    def run_greedy_placement(start_lq, start_pq, nbrs, deg):
        used_phys = {start_pq}
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq, best_w = None, -1.0
            for lq in remaining:
                w = sum(nbrs.get(lq, {}).get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: nbrs.get(best_lq, {}).get(plq, 0.0)
                           for plq in placed if plq in nbrs.get(best_lq, {})}

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        d = dist[pq][m[plq]]
                        cost = iw * d
                        if m[plq] in hw_adj[pq]:
                            cost *= 0.90 - 0.10 * (iw / max_iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    candidates_list = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates_list) > 1 and best_lq in logical_degree_rank:
                        lq_rn = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(candidates_list, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rn), x[0]))[1]
                    else:
                        best_pq = min(candidates_list, key=lambda x: x[0])[1]
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

    # ---------------------------------------------------------------
    # Step 6: Local search (steepest descent)
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost(m, weights)
        for _ in range(max_rounds):
            improved = False
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost(m, rm, pq1, pq2, nbrs)
                    if d < best_d:
                        best_d = d
                        best_pair = (pq1, pq2)
            n_random = min(150, len(interacting_logical) * 3)
            for _ in range(n_random):
                i, j = random.sample(range(len(interacting_logical)), 2)
                pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                d = delta_swap_cost(m, rm, pq1, pq2, nbrs)
                if d < best_d:
                    best_d = d
                    best_pair = (pq1, pq2)
            if best_pair and best_d < -1e-12:
                do_swap(m, rm, *best_pair)
                improved = True
            if not improved:
                for i in range(len(interacting_logical)):
                    for j in range(i + 1, len(interacting_logical)):
                        pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                        d = delta_swap_cost(m, rm, pq1, pq2, nbrs)
                        if d < best_d:
                            best_d = d
                            best_pair = (pq1, pq2)
                if best_pair and best_d < -1e-12:
                    do_swap(m, rm, *best_pair)
                else:
                    break
        return compute_cost(m, weights)

    # ---------------------------------------------------------------
    # Step 7: Routing simulation with progressive depth
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        edge_congestion = defaultdict(float)

        if not gates_2q:
            return swap_counts, edge_congestion, {}

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_remaining[g] == 0)
        layers_done = 0
        sim_decay = [1.0] * num_q
        gate_swap_cost = defaultdict(float)

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

            candidates_sw = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    candidates_sw.add((min(pq, nb), max(pq, nb)))

            best_swap, best_score = None, float('inf')
            for (s1, s2) in candidates_sw:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = sim_m[gq1], sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = dep_count.get(g, 0) + 1
                    score += deps * dist[p1][p2]
                score *= max_d
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1
            sim_decay[s1] += 0.001
            sim_decay[s2] += 0.001

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0
                gate_swap_cost[g] += 1.0

            edge_congestion[(min(s1, s2), max(s1, s2))] += 1.0

        return swap_counts, edge_congestion, gate_swap_cost

    # ---------------------------------------------------------------
    # Step 8: 5 Perturbation strategies (bandit arms)
    # ---------------------------------------------------------------
    # Arm 0: Edge-targeted — swap qubits on highest-cost interaction edge
    def perturb_edge_targeted(m, rm, weights=None, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        edge_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * dist[m[q1]][m[q2]]
                edge_costs.append((c, q1, q2))
        if not edge_costs:
            return
        edge_costs.sort(reverse=True)
        top_n = min(5, len(edge_costs))
        _, eq1, eq2 = edge_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[eq1], m[eq2]
        # Try swapping one endpoint with a hardware neighbor of the other
        neighbors_of_pq2 = list(hw_adj.get(pq2, set()))
        if neighbors_of_pq2:
            target = random.choice(neighbors_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    # Arm 1: Neighborhood-aware — swap highest-cost qubit with low-cost adjacent qubit
    def perturb_neighborhood(m, rm, nbrs=None, weights=None, **kw):
        if nbrs is None or not interacting_logical:
            perturb_random(m, rm)
            return
        qcost = {}
        for lq in interacting_logical:
            c = 0.0
            for partner, w in nbrs.get(lq, {}).items():
                if m[partner] >= 0 and m[lq] >= 0:
                    c += w * dist[m[lq]][m[partner]]
            qcost[lq] = c
        sorted_qs = sorted(qcost, key=lambda q: qcost[q], reverse=True)
        top_lq = sorted_qs[0]
        pq_top = m[top_lq]
        adj_phys = list(hw_adj.get(pq_top, set()))
        if adj_phys:
            # Pick the adjacent physical qubit whose current logical has lowest cost
            best_adj, best_c = None, float('inf')
            for apq in adj_phys:
                alq = rm[apq]
                c = qcost.get(alq, 0.0)
                if c < best_c:
                    best_c = c
                    best_adj = apq
            if best_adj is not None:
                do_swap(m, rm, pq_top, best_adj)
            else:
                do_swap(m, rm, pq_top, random.choice(adj_phys))
        elif len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    # Arm 2: 3-opt chain — 3-cyclic permutation of active logical qubits
    def perturb_3opt(m, rm, **kw):
        if len(interacting_logical) >= 3:
            lqs = random.sample(interacting_logical, 3)
            ps = [m[lq] for lq in lqs]
            m[lqs[0]], m[lqs[1]], m[lqs[2]] = ps[1], ps[2], ps[0]
            rm[ps[0]], rm[ps[1]], rm[ps[2]] = lqs[2], lqs[0], lqs[1]
        elif len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    # Arm 3: Random — random swap of two active logical qubits
    def perturb_random(m, rm, **kw):
        if len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    # Arm 4: Focused perturbation — target top-3 highest-cost pairs,
    # pick one, 2-opt swap of one endpoint with hardware-adjacent qubit of other
    def perturb_focused(m, rm, weights=None, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        pair_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * dist[m[q1]][m[q2]]
                pair_costs.append((c, q1, q2))
        if not pair_costs:
            return
        pair_costs.sort(reverse=True)
        top_3 = pair_costs[:min(3, len(pair_costs))]
        _, tq1, tq2 = random.choice(top_3)
        pq1, pq2 = m[tq1], m[tq2]
        # Targeted 2-opt: swap one endpoint with a hardware-adjacent qubit of the other
        adj_of_pq2 = list(hw_adj.get(pq2, set()))
        adj_of_pq1 = list(hw_adj.get(pq1, set()))
        candidates = []
        for apq in adj_of_pq2:
            if apq != pq1:
                candidates.append((pq1, apq))
        for apq in adj_of_pq1:
            if apq != pq2:
                candidates.append((pq2, apq))
        if candidates:
            swap_pair = random.choice(candidates)
            do_swap(m, rm, swap_pair[0], swap_pair[1])
        else:
            do_swap(m, rm, pq1, pq2)

    perturbations = [perturb_edge_targeted, perturb_neighborhood,
                     perturb_3opt, perturb_random, perturb_focused]
    K = len(perturbations)

    # ---------------------------------------------------------------
    # Step 9: UCB1 Bandit with c=0.5
    # ---------------------------------------------------------------
    arm_total_reward = [0.0] * K
    arm_count = [0] * K
    total_pulls = [0]

    def ucb1_select():
        for k in range(K):
            if arm_count[k] == 0:
                return k
        c = 0.5
        best_k, best_val = 0, -float('inf')
        ln_n = math.log(total_pulls[0])
        for k in range(K):
            mu = arm_total_reward[k] / arm_count[k]
            val = mu + c * math.sqrt(ln_n / arm_count[k])
            if val > best_val:
                best_val = val
                best_k = k
        return best_k

    def update_bandit(arm, reward):
        arm_total_reward[arm] += reward
        arm_count[arm] += 1
        total_pulls[0] += 1

    def reset_bandit():
        for k in range(K):
            arm_total_reward[k] = 0.0
            arm_count[k] = 0
        total_pulls[0] = 0

    # ---------------------------------------------------------------
    # Step 10: Build initial mapping candidates (multi-seed greedy)
    # ---------------------------------------------------------------
    static_nbrs, static_deg = build_neighbors(static_weight)

    candidates = []
    for s_lq in seed_lqs:
        for s_pq in seed_pqs:
            m, rm = run_greedy_placement(s_lq, s_pq, static_nbrs, static_deg)
            fill_unmapped(m, rm)
            cost = compute_cost(m, static_weight)
            candidates.append((cost, m, rm))

    if not candidates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    candidates.sort(key=lambda x: x[0])
    _, best_m, best_rm = candidates[0]
    best_m, best_rm = list(best_m), list(best_rm)
    best_cost = local_search(best_m, best_rm, static_nbrs, static_weight, max_rounds=5)

    for idx in range(1, min(4, len(candidates))):
        _, m_c, rm_c = candidates[idx]
        m_c, rm_c = list(m_c), list(rm_c)
        c = local_search(m_c, rm_c, static_nbrs, static_weight, max_rounds=4)
        if c < best_cost:
            best_cost = c
            best_m = list(m_c)
            best_rm = list(rm_c)

    # ---------------------------------------------------------------
    # Step 11: RSDIWR outer loop with Bandit-Adaptive ILS
    #          + progressive simulation depth
    # ---------------------------------------------------------------
    time_budget = 25.0
    t_start = _time()
    n_rsdiwr = 4
    n_ils = max(40, num_q // 2)

    cur_m = list(best_m)
    cur_rm = list(best_rm)
    gate_swap_cost = {}
    edge_cong = {}
    swap_counts = defaultdict(float)

    for rsdiwr_iter in range(n_rsdiwr):
        if _time() - t_start > time_budget:
            break

        # Progressive simulation depth: increases with each RSDIWR iteration
        sim_depth = 10 + rsdiwr_iter * 5

        # Build effective weights: blend static + routing feedback
        if rsdiwr_iter == 0 or not swap_counts:
            eff_weights = dict(static_weight)
        else:
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max(static_weight.values()) / max(max_sw, 1e-10)
            alpha_blend = max(0.3, 1.0 - 0.3 * rsdiwr_iter)
            eff_weights = defaultdict(float)
            all_keys = set(static_weight.keys()) | set(swap_counts.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        eff_nbrs, eff_deg = build_neighbors(eff_weights)
        cur_cost = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, max_rounds=4)

        # Update best if improved under static weights
        static_cost = compute_cost(cur_m, static_weight)
        if static_cost < best_cost:
            best_cost = static_cost
            best_m = list(cur_m)
            best_rm = list(cur_rm)

        # Reset bandit for new weight landscape
        reset_bandit()

        # SA temperature schedule
        T = max(cur_cost * 0.05, 1.0)
        alpha_sa = 0.95

        for ils_iter in range(n_ils):
            if _time() - t_start > time_budget:
                break

            saved_m = list(cur_m)
            saved_rm = list(cur_rm)
            saved_cost = cur_cost

            # UCB1 selects perturbation arm
            arm = ucb1_select()

            # Apply perturbation
            perturbations[arm](cur_m, cur_rm,
                               nbrs=eff_nbrs, weights=eff_weights,
                               gate_swap_cost=gate_swap_cost, edge_cong=edge_cong)

            # Local search after perturbation
            new_cost = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, max_rounds=3)

            # Bandit reward: r = max(0, cost_before - cost_after) / cost_before
            if saved_cost > 1e-12:
                reward = max(0.0, (saved_cost - new_cost) / saved_cost)
            else:
                reward = 0.0

            # SA acceptance criterion
            improvement = saved_cost - new_cost
            if improvement > 0 or random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                cur_cost = new_cost
                sc = compute_cost(cur_m, static_weight)
                if sc < best_cost:
                    best_cost = sc
                    best_m = list(cur_m)
                    best_rm = list(cur_rm)
            else:
                cur_m[:] = saved_m
                cur_rm[:] = saved_rm
                cur_cost = saved_cost

            # Update bandit with normalized reward
            update_bandit(arm, reward)

            T *= alpha_sa

        # Routing simulation with progressive depth for RSDIWR weight update
        if rsdiwr_iter < n_rsdiwr - 1:
            result = simulate_routing(best_m, best_rm, max_layers=sim_depth)
            swap_counts, edge_cong, gate_swap_cost = result

        # Reset to best for next RSDIWR iteration
        cur_m = list(best_m)
        cur_rm = list(best_rm)

    # ---------------------------------------------------------------
    # Step 12: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)