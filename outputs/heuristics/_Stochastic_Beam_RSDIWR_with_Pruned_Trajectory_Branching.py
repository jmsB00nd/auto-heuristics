def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    B = 3  # Beam width

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
    # Step 2: Build static weighted interaction graph with critical-path weights
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
    # Step 4: Build 2q-only DAG for routing simulation
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
    # Step 5: Helper functions
    # ---------------------------------------------------------------
    rng = np.random.RandomState(42)

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

    def compute_edge_costs(mc, weights):
        edge_costs = []
        for (eq1, eq2), w in weights.items():
            if mc[eq1] >= 0 and mc[eq2] >= 0:
                c = w * self.distance_matrix[mc[eq1]][mc[eq2]]
                edge_costs.append((c, eq1, eq2))
        edge_costs.sort(reverse=True)
        return edge_costs

    def ils_burst(m, rm, weights, logical_nbrs, num_iters=30, temperature=0.5):
        """Short ILS burst with SA acceptance."""
        if len(interacting_logical) < 4:
            return list(m), list(rm)
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m, weights)

        cur_m = list(m)
        cur_rm = list(rm)
        cur_cost = best_cost

        for p_idx in range(num_iters):
            m_try = list(cur_m)
            rm_try = list(cur_rm)
            mode = p_idx % 4

            if mode == 0:
                # Edge-targeted perturbation
                edge_costs = compute_edge_costs(m_try, weights)
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

            elif mode == 1:
                # Neighborhood-aware swap
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

            elif mode == 2:
                # 3-opt cyclic rotation
                indices = rng.choice(len(interacting_logical), size=3, replace=False)
                lq_a = interacting_logical[indices[0]]
                lq_b = interacting_logical[indices[1]]
                lq_c = interacting_logical[indices[2]]
                pq_a, pq_b, pq_c = m_try[lq_a], m_try[lq_b], m_try[lq_c]
                m_try[lq_a] = pq_c
                m_try[lq_b] = pq_a
                m_try[lq_c] = pq_b
                rm_try[pq_a] = lq_b
                rm_try[pq_b] = lq_c
                rm_try[pq_c] = lq_a

            else:
                # Random perturbation
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

            run_swap_refinement(m_try, rm_try, logical_nbrs, max_rounds=2)
            cost = compute_total_cost(m_try, weights)

            # SA acceptance
            temp = temperature * (1.0 - p_idx / num_iters)
            if cost < cur_cost or (temp > 0 and rng.random() < math.exp(-(cost - cur_cost) / max(temp * max(cur_cost, 1.0), 1e-10))):
                cur_m = m_try
                cur_rm = rm_try
                cur_cost = cost

            if cost < best_cost:
                best_cost = cost
                best_m = list(m_try)
                best_rm = list(rm_try)

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 6: Routing simulation — count per-pair SWAPs
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        total_swaps = 0

        if not gates_2q:
            return swap_counts, total_swaps

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
            total_swaps += 1

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts, total_swaps

    # ---------------------------------------------------------------
    # Step 7: Build mapping with effective weights (single candidate)
    # ---------------------------------------------------------------
    def build_single_mapping(eff_weights, s_lq, s_pq):
        eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)
        m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs, eff_deg)
        fill_unmapped(m, rm)
        run_swap_refinement(m, rm, eff_nbrs, max_rounds=3)
        return m, rm

    # ---------------------------------------------------------------
    # Step 8: Diverse seed initialization for beam slots
    # ---------------------------------------------------------------
    def create_spectral_seed():
        """Spectral seed: use Fiedler vector of interaction graph."""
        if len(interacting_logical) < 2 or not static_weight:
            return None, None
        n_int = len(interacting_logical)
        lq_idx = {q: i for i, q in enumerate(interacting_logical)}
        lap = np.zeros((n_int, n_int))
        for (q1, q2), w in static_weight.items():
            if q1 in lq_idx and q2 in lq_idx:
                i, j = lq_idx[q1], lq_idx[q2]
                lap[i][i] += w
                lap[j][j] += w
                lap[i][j] -= w
                lap[j][i] -= w
        try:
            eigvals, eigvecs = np.linalg.eigh(lap)
            fiedler_idx = 1 if n_int > 1 else 0
            fiedler = eigvecs[:, fiedler_idx]
            sorted_lq = [interacting_logical[i] for i in np.argsort(fiedler)]
            mid_lq = sorted_lq[len(sorted_lq) // 2]
        except Exception:
            mid_lq = interacting_logical[0]

        # Pick physical qubit with best centrality
        best_pq = min(physical_qubits, key=lambda pq: phys_centrality[pq])
        return mid_lq, best_pq

    def create_mis_seed():
        """MIS-based seed: pick qubit from maximal independent set of interaction graph."""
        if not interacting_logical:
            return None, None
        # Greedy MIS on interaction graph
        adj = defaultdict(set)
        for (q1, q2) in static_weight:
            adj[q1].add(q2)
            adj[q2].add(q1)
        mis = []
        remaining = set(interacting_logical)
        sorted_lqs = sorted(remaining, key=lambda q: len(adj.get(q, set()) & remaining))
        for q in sorted_lqs:
            if q in remaining:
                mis.append(q)
                remaining -= adj.get(q, set())
                remaining.discard(q)
        # Seed from highest-degree qubit in MIS
        if mis:
            seed_lq = max(mis, key=lambda q: logical_degree_static.get(q, 0))
        else:
            seed_lq = interacting_logical[0]
        # Use second-best centrality physical qubit for diversity
        sorted_phys = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
        seed_pq = sorted_phys[min(1, len(sorted_phys) - 1)]
        return seed_lq, seed_pq

    # ---------------------------------------------------------------
    # Step 9: Perturbation operators for beam branching (no crossover)
    # ---------------------------------------------------------------
    def perturb_lns(m, rm, destruction_size):
        """Large Neighborhood Search: destroy and reconstruct a region."""
        m_new = list(m)
        rm_new = list(rm)
        if len(interacting_logical) < destruction_size + 2:
            return m_new, rm_new

        # Find worst-cost qubits
        qcost = defaultdict(float)
        for (eq1, eq2), w in static_weight.items():
            if m_new[eq1] >= 0 and m_new[eq2] >= 0:
                c = w * self.distance_matrix[m_new[eq1]][m_new[eq2]]
                qcost[eq1] += c
                qcost[eq2] += c
        cost_sorted = sorted(interacting_logical, key=lambda q: qcost.get(q, 0), reverse=True)
        to_destroy = cost_sorted[:destruction_size]

        freed_phys = [m_new[lq] for lq in to_destroy]
        for lq in to_destroy:
            pq = m_new[lq]
            rm_new[pq] = -1
            m_new[lq] = -1

        rng.shuffle(freed_phys)
        # Re-assign greedily
        eff_nbrs, _ = build_neighbors_from_weights(static_weight)
        for lq in to_destroy:
            best_pq = None
            best_score = float('inf')
            for pq in freed_phys:
                if rm_new[pq] != -1:
                    continue
                score = 0.0
                for nbr, w in eff_nbrs[lq].items():
                    if m_new[nbr] >= 0:
                        score += w * self.distance_matrix[pq][m_new[nbr]]
                if score < best_score:
                    best_score = score
                    best_pq = pq
            if best_pq is not None:
                m_new[lq] = best_pq
                rm_new[best_pq] = lq

        # Safety: fill any remaining
        fill_unmapped(m_new, rm_new)
        return m_new, rm_new

    def perturb_segment_shuffle(m, rm, position):
        """Shuffle a segment of interacting qubits' assignments."""
        m_new = list(m)
        rm_new = list(rm)
        if len(interacting_logical) < 4:
            return m_new, rm_new
        seg_size = max(3, len(interacting_logical) // 4)
        start = position % max(1, len(interacting_logical) - seg_size)
        segment = interacting_logical[start:start + seg_size]
        phys_vals = [m_new[lq] for lq in segment]
        rng.shuffle(phys_vals)
        for lq, pq in zip(segment, phys_vals):
            old_pq = m_new[lq]
            rm_new[old_pq] = -1
        for lq, pq in zip(segment, phys_vals):
            m_new[lq] = pq
            rm_new[pq] = lq
        return m_new, rm_new

    def perturb_worst_cluster(m, rm):
        """Re-assign the worst-cost cluster of qubits."""
        m_new = list(m)
        rm_new = list(rm)
        if len(interacting_logical) < 6:
            # Just do random swaps
            if len(interacting_logical) >= 4:
                indices = rng.choice(len(interacting_logical), size=4, replace=False)
                for s in range(0, len(indices) - 1, 2):
                    lq_a = interacting_logical[indices[s]]
                    lq_b = interacting_logical[indices[s + 1]]
                    pq_a, pq_b = m_new[lq_a], m_new[lq_b]
                    m_new[lq_a] = pq_b
                    m_new[lq_b] = pq_a
                    rm_new[pq_a] = lq_b
                    rm_new[pq_b] = lq_a
            return m_new, rm_new

        # Find qubit with highest cost, take its neighborhood
        qcost = defaultdict(float)
        adj = defaultdict(set)
        for (eq1, eq2), w in static_weight.items():
            adj[eq1].add(eq2)
            adj[eq2].add(eq1)
            if m_new[eq1] >= 0 and m_new[eq2] >= 0:
                c = w * self.distance_matrix[m_new[eq1]][m_new[eq2]]
                qcost[eq1] += c
                qcost[eq2] += c

        worst_q = max(interacting_logical, key=lambda q: qcost.get(q, 0))
        cluster = [worst_q] + [q for q in adj.get(worst_q, set()) if q in set(interacting_logical)]
        cluster = cluster[:max(3, len(interacting_logical) // 5)]

        phys_vals = [m_new[lq] for lq in cluster]
        rng.shuffle(phys_vals)
        for lq in cluster:
            rm_new[m_new[lq]] = -1
        for lq, pq in zip(cluster, phys_vals):
            m_new[lq] = pq
            rm_new[pq] = lq
        return m_new, rm_new

    # ---------------------------------------------------------------
    # Step 10: Stochastic Beam RSDIWR Main Loop
    # ---------------------------------------------------------------
    T = 4  # RSDIWR iterations
    alpha_schedule = [1.0, 0.7, 0.45, 0.3]

    # Initialize beam with B=3 diverse seeds
    beam = []  # list of (m, rm, swap_counts_dict)

    # Seed 1: Greedy (highest degree logical → best centrality physical)
    if seed_lqs and seed_pqs:
        m1, rm1 = build_single_mapping(static_weight, seed_lqs[0], seed_pqs[0])
        beam.append((m1, rm1, defaultdict(float)))

    # Seed 2: Spectral
    spec_lq, spec_pq = create_spectral_seed()
    if spec_lq is not None:
        m2, rm2 = build_single_mapping(static_weight, spec_lq, spec_pq)
        beam.append((m2, rm2, defaultdict(float)))
    elif len(seed_lqs) > 1 and len(seed_pqs) > 1:
        m2, rm2 = build_single_mapping(static_weight, seed_lqs[min(1, len(seed_lqs)-1)], seed_pqs[min(1, len(seed_pqs)-1)])
        beam.append((m2, rm2, defaultdict(float)))

    # Seed 3: MIS
    mis_lq, mis_pq = create_mis_seed()
    if mis_lq is not None:
        m3, rm3 = build_single_mapping(static_weight, mis_lq, mis_pq)
        beam.append((m3, rm3, defaultdict(float)))
    elif len(seed_lqs) > 2 and len(seed_pqs) > 2:
        m3, rm3 = build_single_mapping(static_weight, seed_lqs[2], seed_pqs[2])
        beam.append((m3, rm3, defaultdict(float)))

    # Ensure we have at least one candidate
    if not beam:
        m_triv = list(range(num_q))
        rm_triv = list(range(num_q))
        beam.append((m_triv, rm_triv, defaultdict(float)))

    # Pad beam to B if needed
    while len(beam) < B and seed_lqs and seed_pqs:
        idx = len(beam) % (len(seed_lqs) * len(seed_pqs))
        s_lq = seed_lqs[idx % len(seed_lqs)]
        s_pq = seed_pqs[idx % len(seed_pqs)]
        m_pad, rm_pad = build_single_mapping(static_weight, s_lq, s_pq)
        beam.append((m_pad, rm_pad, defaultdict(float)))

    best_overall_m = None
    best_overall_rm = None
    best_overall_swaps = float('inf')

    for t in range(T):
        alpha_blend = alpha_schedule[min(t, len(alpha_schedule) - 1)]

        # (1) For each beam candidate: compute effective weights, rebuild mapping, run routing
        new_beam = []
        for bi in range(len(beam)):
            m_b, rm_b, prev_swap_counts = beam[bi]

            # Compute effective weights for this candidate
            eff_weights = defaultdict(float)
            if t == 0 or not prev_swap_counts:
                for key, w in static_weight.items():
                    eff_weights[key] = w
            else:
                max_static = max(static_weight.values()) if static_weight else 1.0
                max_swap = max(prev_swap_counts.values()) if prev_swap_counts else 1.0
                scale = max_static / max(max_swap, 1e-10)
                all_keys = set(static_weight.keys()) | set(prev_swap_counts.keys())
                for key in all_keys:
                    w_s = static_weight.get(key, 0.0)
                    w_r = prev_swap_counts.get(key, 0.0) * scale
                    eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

            # Rebuild mapping with updated weights (if not first iteration)
            if t > 0 and prev_swap_counts:
                # Pick seed based on beam index for diversity
                si_lq = seed_lqs[bi % len(seed_lqs)] if seed_lqs else 0
                si_pq = seed_pqs[bi % len(seed_pqs)] if seed_pqs else 0
                cur_m, cur_rm = build_single_mapping(eff_weights, si_lq, si_pq)
            else:
                cur_m, cur_rm = list(m_b), list(rm_b)

            # Run routing simulation
            swap_counts, total_swaps = simulate_routing(cur_m, cur_rm, max_layers=25)

            # Track best by actual swap count
            if total_swaps < best_overall_swaps:
                best_overall_swaps = total_swaps
                best_overall_m = list(cur_m)
                best_overall_rm = list(cur_rm)

            new_beam.append((cur_m, cur_rm, swap_counts, total_swaps))

        # (2) For each candidate, run short ILS burst producing refined version
        eff_nbrs_static, _ = build_neighbors_from_weights(static_weight)
        all_candidates = []  # (total_swaps, m, rm, swap_counts)

        for bi in range(len(new_beam)):
            cur_m, cur_rm, swap_counts, total_swaps = new_beam[bi]

            # Original candidate
            all_candidates.append((total_swaps, list(cur_m), list(cur_rm), swap_counts))

            # ILS-refined candidate
            # Build effective weights for ILS using this candidate's swap feedback
            eff_weights_ils = defaultdict(float)
            if swap_counts:
                max_static = max(static_weight.values()) if static_weight else 1.0
                max_swap = max(swap_counts.values()) if swap_counts else 1.0
                scale = max_static / max(max_swap, 1e-10)
                all_keys = set(static_weight.keys()) | set(swap_counts.keys())
                for key in all_keys:
                    w_s = static_weight.get(key, 0.0)
                    w_r = swap_counts.get(key, 0.0) * scale
                    eff_weights_ils[key] = 0.5 * w_s + 0.5 * w_r
            else:
                eff_weights_ils = dict(static_weight)

            eff_nbrs_ils, _ = build_neighbors_from_weights(eff_weights_ils)
            ref_m, ref_rm = ils_burst(cur_m, cur_rm, eff_weights_ils, eff_nbrs_ils, num_iters=30)

            # Evaluate refined candidate via routing
            ref_swap_counts, ref_total_swaps = simulate_routing(ref_m, ref_rm, max_layers=25)

            if ref_total_swaps < best_overall_swaps:
                best_overall_swaps = ref_total_swaps
                best_overall_m = list(ref_m)
                best_overall_rm = list(ref_rm)

            all_candidates.append((ref_total_swaps, ref_m, ref_rm, ref_swap_counts))

        # (3) Prune: select top-B candidates by routing swap count
        all_candidates.sort(key=lambda x: x[0])
        beam = [(c[1], c[2], c[3]) for c in all_candidates[:B]]

        # (4) Every 2nd RSDIWR round: branch for diversity
        if t % 2 == 1 and t < T - 1 and len(interacting_logical) >= 4:
            # Take the single best candidate
            best_m_branch, best_rm_branch, best_sc = beam[0]

            # Generate B-1 diverse variants via different perturbations
            branched = [(list(best_m_branch), list(best_rm_branch), best_sc)]

            # Variant 1: LNS with small destruction
            lns_m, lns_rm = perturb_lns(best_m_branch, best_rm_branch,
                                        destruction_size=max(3, len(interacting_logical) // 5))
            run_swap_refinement(lns_m, lns_rm, eff_nbrs_static, max_rounds=3)
            branched.append((lns_m, lns_rm, defaultdict(float)))

            # Variant 2: Segment shuffle or worst-cluster depending on beam size
            if len(beam) > 1:
                seg_m, seg_rm = perturb_segment_shuffle(best_m_branch, best_rm_branch, position=0)
                run_swap_refinement(seg_m, seg_rm, eff_nbrs_static, max_rounds=3)
                branched.append((seg_m, seg_rm, defaultdict(float)))
            else:
                wc_m, wc_rm = perturb_worst_cluster(best_m_branch, best_rm_branch)
                run_swap_refinement(wc_m, wc_rm, eff_nbrs_static, max_rounds=3)
                branched.append((wc_m, wc_rm, defaultdict(float)))

            beam = branched[:B]

    # ---------------------------------------------------------------
    # Step 11: Final selection — evaluate all beam candidates with routing
    # ---------------------------------------------------------------
    for m_b, rm_b, _ in beam:
        _, total_swaps = simulate_routing(m_b, rm_b, max_layers=30)
        if total_swaps < best_overall_swaps:
            best_overall_swaps = total_swaps
            best_overall_m = list(m_b)
            best_overall_rm = list(rm_b)

    # Fallback
    if best_overall_m is None:
        best_overall_m = list(range(num_q))
        best_overall_rm = list(range(num_q))

    # ---------------------------------------------------------------
    # Step 12: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)