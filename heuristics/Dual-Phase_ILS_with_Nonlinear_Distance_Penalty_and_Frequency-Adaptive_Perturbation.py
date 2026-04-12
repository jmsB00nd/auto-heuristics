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
            dep_count[g] = max(dep_count[g], dep_count[s] + 1)

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

    # INNOVATION 1: Nonlinear distance penalty cost function
    def compute_cost_nonlinear(m, weights, alpha_exp):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                cost += w * (d ** alpha_exp)
        return cost

    def compute_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * dist[m[q1]][m[q2]]
        return cost

    def delta_swap_cost_nonlinear(m, rm, pq_a, pq_b, nbrs, alpha_exp):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        delta = 0.0
        affected = set()
        if lq_a in nbrs:
            affected.update(nbrs[lq_a].keys())
        if lq_b in nbrs:
            affected.update(nbrs[lq_b].keys())
        # Handle the direct edge between lq_a and lq_b
        w_ab = nbrs.get(lq_a, {}).get(lq_b, 0.0)
        if w_ab > 0:
            old_d = dist[pq_a][pq_b]
            new_d = dist[pq_b][pq_a]  # same, but after swap positions are swapped
            # After swap: lq_a->pq_b, lq_b->pq_a, distance is the same
            # so no change for this pair
        for q in affected:
            if q == lq_a or q == lq_b:
                continue
            pq_q = m[q]
            w_a = nbrs.get(lq_a, {}).get(q, 0.0)
            if w_a > 0:
                old_d = dist[pq_a][pq_q] ** alpha_exp
                new_d = dist[pq_b][pq_q] ** alpha_exp
                delta += w_a * (new_d - old_d)
            w_b = nbrs.get(lq_b, {}).get(q, 0.0)
            if w_b > 0:
                old_d = dist[pq_b][pq_q] ** alpha_exp
                new_d = dist[pq_a][pq_q] ** alpha_exp
                delta += w_b * (new_d - old_d)
        return delta

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        m[lq_a], m[lq_b] = pq_b, pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    # ---------------------------------------------------------------
    # Step 5: Greedy initial placement (multi-seed, connectivity-matching)
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
    # Step 6: Local search with nonlinear distance penalty
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, alpha_exp, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost_nonlinear(m, weights, alpha_exp)
        for _ in range(max_rounds):
            improved = False
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d = d
                        best_pair = (pq1, pq2)
            n_random = min(150, len(interacting_logical) * 3)
            for _ in range(n_random):
                i, j = random.sample(range(len(interacting_logical)), 2)
                pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
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
                        d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                        if d < best_d:
                            best_d = d
                            best_pair = (pq1, pq2)
                if best_pair and best_d < -1e-12:
                    do_swap(m, rm, *best_pair)
                else:
                    break
        return compute_cost_nonlinear(m, weights, alpha_exp)

    # ---------------------------------------------------------------
    # Step 7: Routing simulation
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)

        if not gates_2q:
            return swap_counts

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_remaining[g] == 0)
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

        return swap_counts

    # ---------------------------------------------------------------
    # Step 8: Five perturbation modes for frequency-adaptive selection
    # ---------------------------------------------------------------
    # Mode 0: Random swap of two interacting logical qubits
    def perturb_random(m, rm, **kw):
        if len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    # Mode 1: Segment shuffle — pick 3-5 interacting qubits, shuffle their positions
    def perturb_segment_shuffle(m, rm, **kw):
        k = min(random.randint(3, 5), len(interacting_logical))
        if k < 2:
            perturb_random(m, rm)
            return
        lqs = random.sample(interacting_logical, k)
        phys_positions = [m[lq] for lq in lqs]
        random.shuffle(phys_positions)
        for lq, pq in zip(lqs, phys_positions):
            rm[m[lq]] = -1
        for lq, pq in zip(lqs, phys_positions):
            m[lq] = pq
            rm[pq] = lq

    # Mode 2: Worst-pair swap — swap one endpoint of highest-cost pair toward the other
    def perturb_worst_pair(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        pair_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                c = w * (d ** alpha_exp)
                pair_costs.append((c, q1, q2))
        if not pair_costs:
            return
        pair_costs.sort(reverse=True)
        top_n = min(3, len(pair_costs))
        _, tq1, tq2 = pair_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[tq1], m[tq2]
        adj_of_pq2 = list(hw_adj.get(pq2, set()))
        if adj_of_pq2:
            target = random.choice(adj_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    # Mode 3: Edge-targeted — swap along a high-congestion hardware edge
    def perturb_edge_targeted(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        edge_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * (dist[m[q1]][m[q2]] ** alpha_exp)
                edge_costs.append((c, q1, q2))
        if not edge_costs:
            return
        edge_costs.sort(reverse=True)
        top_n = min(5, len(edge_costs))
        _, eq1, eq2 = edge_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[eq1], m[eq2]
        neighbors_of_pq2 = list(hw_adj.get(pq2, set()))
        if neighbors_of_pq2:
            target = random.choice(neighbors_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    # Mode 4: LNS (Large Neighborhood Search) — unplace subset and re-greedily place
    def perturb_lns(m, rm, nbrs=None, alpha_exp=1.5, **kw):
        if nbrs is None or len(interacting_logical) < 3:
            perturb_random(m, rm)
            return
        # Pick a subset of 3-8 interacting qubits with highest cost contribution
        k = min(random.randint(3, 8), len(interacting_logical))
        qcost = {}
        for lq in interacting_logical:
            c = 0.0
            for partner, w in nbrs.get(lq, {}).items():
                if m[partner] >= 0 and m[lq] >= 0:
                    c += w * (dist[m[lq]][m[partner]] ** alpha_exp)
            qcost[lq] = c
        sorted_qs = sorted(qcost, key=lambda q: qcost[q], reverse=True)
        # Bias toward high-cost qubits: pick top half + some random
        top_half = sorted_qs[:max(k, len(sorted_qs) // 2)]
        subset = random.sample(top_half, min(k, len(top_half)))

        # Free their physical positions
        freed_phys = []
        for lq in subset:
            freed_phys.append(m[lq])
            rm[m[lq]] = -1
            m[lq] = -1

        # Re-place greedily to minimize cost
        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)
        for lq in sorted(subset, key=lambda q: qcost[q], reverse=True):
            best_pq, best_sc = None, float('inf')
            for pq in freed_phys:
                if rm[pq] != -1:
                    continue
                sc = 0.0
                for partner, w in nbrs.get(lq, {}).items():
                    if partner in placed_set and m[partner] >= 0:
                        sc += w * (dist[pq][m[partner]] ** alpha_exp)
                if sc < best_sc:
                    best_sc = sc
                    best_pq = pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                placed_set.add(lq)
            else:
                # Fallback: assign any free position
                for pq in freed_phys:
                    if rm[pq] == -1:
                        m[lq] = pq
                        rm[pq] = lq
                        placed_set.add(lq)
                        break

    perturbation_modes = [perturb_random, perturb_segment_shuffle,
                          perturb_worst_pair, perturb_edge_targeted, perturb_lns]
    K = len(perturbation_modes)

    # ---------------------------------------------------------------
    # INNOVATION 2: Frequency-adaptive perturbation with sliding window
    # ---------------------------------------------------------------
    WINDOW_SIZE = 100
    EPSILON = 0.1
    # Sliding window: list of (mode_index, success_bool)
    window = []
    mode_successes = [0] * K
    mode_attempts = [0] * K

    def select_perturbation_mode():
        """Multi-armed bandit with epsilon-greedy over sliding window success rates."""
        if random.random() < EPSILON or sum(mode_attempts) < K * 2:
            return random.randint(0, K - 1)
        # Compute success rates from sliding window
        rates = []
        for k in range(K):
            if mode_attempts[k] > 0:
                rates.append(mode_successes[k] / mode_attempts[k])
            else:
                rates.append(1.0)  # Optimistic for unexplored
        total = sum(rates)
        if total < 1e-12:
            return random.randint(0, K - 1)
        # Proportional selection
        r = random.random() * total
        cumul = 0.0
        for k in range(K):
            cumul += rates[k]
            if r <= cumul:
                return k
        return K - 1

    def update_window(mode_idx, success):
        window.append((mode_idx, success))
        mode_attempts[mode_idx] += 1
        if success:
            mode_successes[mode_idx] += 1
        # Evict oldest if window full
        if len(window) > WINDOW_SIZE:
            old_mode, old_success = window.pop(0)
            mode_attempts[old_mode] -= 1
            if old_success:
                mode_successes[old_mode] -= 1

    # ---------------------------------------------------------------
    # Step 9: Build initial mapping candidates (multi-seed greedy)
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
    # Initial local search with aggressive alpha
    best_cost_nl = local_search(best_m, best_rm, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=5)
    best_cost = compute_cost(best_m, static_weight)

    for idx in range(1, min(4, len(candidates))):
        _, m_c, rm_c = candidates[idx]
        m_c, rm_c = list(m_c), list(rm_c)
        local_search(m_c, rm_c, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=4)
        c = compute_cost(m_c, static_weight)
        if c < best_cost:
            best_cost = c
            best_m = list(m_c)
            best_rm = list(rm_c)

    # ---------------------------------------------------------------
    # Step 10: Dual-Phase ILS with RSDIWR outer loop
    # ---------------------------------------------------------------
    time_budget = 25.0
    t_start = _time()
    n_rsdiwr = 4
    n_ils = max(50, num_q // 2)

    cur_m = list(best_m)
    cur_rm = list(best_rm)
    swap_counts = defaultdict(float)

    for rsdiwr_iter in range(n_rsdiwr):
        if _time() - t_start > time_budget:
            break

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

        # INNOVATION 1: Anneal alpha from 2.0 (early) to 1.0 (late)
        alpha_start = 2.0
        alpha_end = 1.0

        # Initial local search for this phase
        cur_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                   alpha_exp=alpha_start, max_rounds=4)

        # Track best under linear (static) cost
        static_cost = compute_cost(cur_m, static_weight)
        if static_cost < best_cost:
            best_cost = static_cost
            best_m = list(cur_m)
            best_rm = list(cur_rm)

        # Reset sliding window for new weight landscape
        window.clear()
        for k in range(K):
            mode_successes[k] = 0
            mode_attempts[k] = 0

        # SA temperature schedule with reheat
        T = max(cur_cost_nl * 0.05, 1.0)
        T_init = T
        alpha_sa = 0.95
        reheat_interval = max(n_ils // 3, 10)

        for ils_iter in range(n_ils):
            if _time() - t_start > time_budget:
                break

            # INNOVATION 1: Anneal alpha exponent over iterations
            progress = ils_iter / max(n_ils - 1, 1)
            alpha_exp = alpha_start + (alpha_end - alpha_start) * progress

            saved_m = list(cur_m)
            saved_rm = list(cur_rm)
            saved_cost = cur_cost_nl

            # INNOVATION 2: Select perturbation via frequency-adaptive bandit
            mode = select_perturbation_mode()

            # Apply perturbation
            perturbation_modes[mode](cur_m, cur_rm,
                                     nbrs=eff_nbrs, weights=eff_weights,
                                     alpha_exp=alpha_exp)

            # Local search after perturbation with current alpha
            new_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                       alpha_exp=alpha_exp, max_rounds=3)

            # Determine success for bandit
            success = new_cost_nl < saved_cost - 1e-12

            # SA acceptance with cost-biased cooling
            improvement = saved_cost - new_cost_nl
            if improvement > 0:
                cur_cost_nl = new_cost_nl
                sc = compute_cost(cur_m, static_weight)
                if sc < best_cost:
                    best_cost = sc
                    best_m = list(cur_m)
                    best_rm = list(cur_rm)
            elif random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                cur_cost_nl = new_cost_nl
            else:
                cur_m[:] = saved_m
                cur_rm[:] = saved_rm
                cur_cost_nl = saved_cost

            # Update frequency-adaptive window
            update_window(mode, success)

            # Cool temperature
            T *= alpha_sa

            # Reheat periodically
            if (ils_iter + 1) % reheat_interval == 0:
                T = max(T, T_init * 0.3)

        # Routing simulation for RSDIWR weight update
        if rsdiwr_iter < n_rsdiwr - 1:
            sim_depth = 10 + rsdiwr_iter * 5
            swap_counts = simulate_routing(best_m, best_rm, max_layers=sim_depth)

        # Reset to best for next RSDIWR iteration
        cur_m = list(best_m)
        cur_rm = list(best_rm)
        cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

    # ---------------------------------------------------------------
    # Step 11: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)