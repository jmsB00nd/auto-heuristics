def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # ---------------------------------------------------------------
    # Step 1: Build DAG and compute topological rank + critical path
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

    while queue:
        g = queue.popleft()
        topo_order.append(g)
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ---------------------------------------------------------------
    # Step 2: Build interaction weights with critical-path + temporal decay
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha_decay = 2.5

    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        r = topo_rank.get(g, 0)
        w_cp = cp * (max_layer - layer + 1)
        w_td = math.exp(-alpha_decay * r / total_gates)
        w = math.sqrt(w_cp * w_td)
        interaction_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

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

    # ---------------------------------------------------------------
    # Step 4: Build 2q DAG for routing simulation
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
    # Step 5: Spectral seed via Fiedler vector
    # ---------------------------------------------------------------
    def compute_spectral_seed():
        if len(interacting_logical) < 4:
            return None
        idx_map = {q: i for i, q in enumerate(interacting_logical)}
        n = len(interacting_logical)
        L = np.zeros((n, n))
        for (q1, q2), w in interaction_weight.items():
            if q1 in idx_map and q2 in idx_map:
                i, j = idx_map[q1], idx_map[q2]
                L[i][i] += w
                L[j][j] += w
                L[i][j] -= w
                L[j][i] -= w
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler_idx = 1 if n > 1 else 0
            fiedler = eigvecs[:, fiedler_idx]
            sorted_logical = [interacting_logical[i] for i in np.argsort(fiedler)]
            return sorted_logical
        except:
            return None

    spectral_order = compute_spectral_seed()

    # ---------------------------------------------------------------
    # Step 6: Multi-seed greedy placement
    # ---------------------------------------------------------------
    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    adjacency_bonus = 0.85

    def run_greedy_placement(start_lq, start_pq, weights_nbrs):
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
                w = sum(weights_nbrs[lq].get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: weights_nbrs[best_lq].get(plq, 0.0)
                           for plq in placed if plq in weights_nbrs[best_lq]}

            if nbrs_placed:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        dist = self.distance_matrix[pq][m[plq]]
                        cost = iw * dist
                        if m[plq] in hw_adj[pq]:
                            cost *= adjacency_bonus
                        score += cost
                    if score < best_score:
                        best_score = score
                        best_pq = pq
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

    def run_spectral_greedy(spectral_sorted, start_pq, weights_nbrs):
        """Place qubits in spectral order via BFS on hardware from start_pq."""
        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        # BFS order on physical graph from start_pq
        phys_order = []
        visited = set()
        bfs_q = deque([start_pq])
        visited.add(start_pq)
        while bfs_q:
            pq = bfs_q.popleft()
            phys_order.append(pq)
            for nb in sorted(self.backend.get(pq, [])):
                if nb not in visited:
                    visited.add(nb)
                    bfs_q.append(nb)

        # Assign spectral-ordered logical qubits to BFS-ordered physical qubits
        p_idx = 0
        for lq in spectral_sorted:
            if p_idx < len(phys_order):
                pq = phys_order[p_idx]
                m[lq] = pq
                rm[pq] = lq
                used_phys.add(pq)
                p_idx += 1

        # Place remaining logical qubits
        remaining_lq = [q for q in logical_qubits if q not in set(spectral_sorted)]
        for lq in remaining_lq:
            while p_idx < len(phys_order) and phys_order[p_idx] in used_phys:
                p_idx += 1
            if p_idx < len(phys_order):
                pq = phys_order[p_idx]
                m[lq] = pq
                rm[pq] = lq
                used_phys.add(pq)
                p_idx += 1

        return m, rm

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    # ---------------------------------------------------------------
    # Step 7: Dual objective computation
    # ---------------------------------------------------------------
    def compute_O1(m, weights):
        """Sum of w * dist^alpha_exp."""
        cost = 0.0
        alpha_exp = 1.3
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = self.distance_matrix[m[q1]][m[q2]]
                cost += w * (d ** alpha_exp)
        return cost

    def compute_O2(m, weights):
        """Worst-pair cost: max over all pairs of w * dist."""
        worst = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = self.distance_matrix[m[q1]][m[q2]]
                c = w * d
                if c > worst:
                    worst = c
        return worst

    # ---------------------------------------------------------------
    # Step 8: Pareto front management
    # ---------------------------------------------------------------
    MAX_FRONT = 8

    def dominates(a, b):
        """a dominates b if a is <= in all objectives and < in at least one."""
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def add_to_pareto_front(front, candidate):
        """Try to add candidate (o1, o2, m, rm) to front. Returns updated front."""
        o1_c, o2_c = candidate[0], candidate[1]
        # Check if candidate is dominated
        for entry in front:
            if dominates((entry[0], entry[1]), (o1_c, o2_c)):
                return front  # candidate dominated

        # Remove entries dominated by candidate
        new_front = [e for e in front if not dominates((o1_c, o2_c), (e[0], e[1]))]
        new_front.append(candidate)

        # If front exceeds max size, remove the one with smallest hypervolume contribution
        if len(new_front) > MAX_FRONT:
            hv_contribs = compute_hypervolume_contributions(new_front)
            min_idx = min(range(len(new_front)), key=lambda i: hv_contribs[i])
            new_front.pop(min_idx)

        return new_front

    def compute_hypervolume_contributions(front):
        """Compute dominated hypervolume contribution of each member.
        Reference point: (max_o1 * 1.1, max_o2 * 1.1)."""
        if not front:
            return []
        max_o1 = max(e[0] for e in front) * 1.1 + 1e-10
        max_o2 = max(e[1] for e in front) * 1.1 + 1e-10

        # Sort by o1
        sorted_front = sorted(range(len(front)), key=lambda i: front[i][0])
        contributions = [0.0] * len(front)

        for k, idx in enumerate(sorted_front):
            o1_k = front[idx][0]
            o2_k = front[idx][1]
            # Width
            if k == 0:
                left_o1 = 0.0
            else:
                left_o1 = front[sorted_front[k - 1]][0]
            if k == len(sorted_front) - 1:
                right_o1 = max_o1
            else:
                right_o1 = front[sorted_front[k + 1]][0]
            # Height: from o2_k to the minimum o2 of neighbors or ref point
            if k == len(sorted_front) - 1:
                upper_o2 = max_o2
            else:
                upper_o2 = max_o2
                for j in range(k + 1, len(sorted_front)):
                    upper_o2 = min(upper_o2, front[sorted_front[j]][1])

            # Simplified contribution: rectangle from this point to ref minus overlap
            width = right_o1 - o1_k
            height = max_o2 - o2_k
            contributions[idx] = width * height

        return contributions

    # ---------------------------------------------------------------
    # Step 9: Swap refinement
    # ---------------------------------------------------------------
    def run_swap_refinement(m, rm, weights_nbrs, max_rounds):
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
                    affected.update(weights_nbrs[lq_a].keys())
                    affected.update(weights_nbrs[lq_b].keys())

                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = m[q]
                        w_a = weights_nbrs[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                        w_b = weights_nbrs[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    # ---------------------------------------------------------------
    # Step 10: Bandit-adaptive perturbation with SA + reheat
    # ---------------------------------------------------------------
    def perturb_and_refine_pareto(m, rm, eff_weights, eff_nbrs, pareto_front, rng, num_perturbations=12):
        if len(interacting_logical) < 4:
            return m, rm, pareto_front

        best_m = list(m)
        best_rm = list(rm)
        best_o1 = compute_O1(m, eff_weights)
        current_m = list(m)
        current_rm = list(rm)
        current_o1 = best_o1

        T_init = max(best_o1 * 0.03, 1.0)
        T_min = T_init * 0.005

        # Bandit: 5 perturbation modes with UCB
        n_modes = 5
        mode_rewards = [0.0] * n_modes
        mode_counts = [1] * n_modes  # start at 1 to avoid div by 0

        def compute_edge_costs(mc, weights):
            edge_costs = []
            for (eq1, eq2), w in weights.items():
                if mc[eq1] >= 0 and mc[eq2] >= 0:
                    c = w * self.distance_matrix[mc[eq1]][mc[eq2]]
                    edge_costs.append((c, eq1, eq2))
            edge_costs.sort(reverse=True)
            return edge_costs

        for p_idx in range(num_perturbations):
            # Nonlinear annealing with reheat
            progress = p_idx / max(num_perturbations - 1, 1)
            if progress < 0.5:
                T = T_init * (T_min / T_init) ** (2.0 * progress)
            else:
                # Reheat phase
                reheat_progress = (progress - 0.5) * 2.0
                T_reheat = T_init * 0.3
                T = T_reheat * (T_min / T_reheat) ** reheat_progress

            # UCB mode selection
            total_tries = sum(mode_counts)
            ucb_scores = []
            for md in range(n_modes):
                exploit = mode_rewards[md] / mode_counts[md]
                explore = math.sqrt(2.0 * math.log(total_tries) / mode_counts[md])
                ucb_scores.append(exploit + explore)
            mode = max(range(n_modes), key=lambda i: ucb_scores[i])

            m_try = list(current_m)
            rm_try = list(current_rm)

            if mode == 0:
                # Random swap
                n_swaps = min(2, len(interacting_logical) // 2)
                indices = rng.choice(len(interacting_logical),
                                     size=min(2 * n_swaps, len(interacting_logical)),
                                     replace=False)
                for s in range(0, len(indices) - 1, 2):
                    lq_a = interacting_logical[indices[s]]
                    lq_b = interacting_logical[indices[s + 1]]
                    pq_a, pq_b = m_try[lq_a], m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            elif mode == 1:
                # Segment shuffle: shuffle a contiguous block of interacting qubits
                seg_size = min(max(3, len(interacting_logical) // 4), len(interacting_logical))
                start_idx = rng.randint(0, max(len(interacting_logical) - seg_size, 0) + 1)
                segment = interacting_logical[start_idx:start_idx + seg_size]
                phys_positions = [m_try[lq] for lq in segment]
                rng.shuffle(phys_positions)
                for lq, pq in zip(segment, phys_positions):
                    old_pq = m_try[lq]
                    m_try[lq] = pq
                for lq in segment:
                    rm_try[m_try[lq]] = lq

            elif mode == 2:
                # Worst-pair targeted: swap the worst-cost pair with neighbors
                edge_costs = compute_edge_costs(m_try, eff_weights)
                if edge_costs:
                    _, eq1, eq2 = edge_costs[0]
                    pq1, pq2 = m_try[eq1], m_try[eq2]
                    # Try swapping eq1 with a neighbor of eq2's physical position
                    adj_candidates = list(hw_adj[pq2])
                    if adj_candidates:
                        target_pq = adj_candidates[rng.randint(0, len(adj_candidates))]
                        occ = rm_try[target_pq]
                        if occ >= 0:
                            m_try[eq1] = target_pq
                            m_try[occ] = pq1
                            rm_try[pq1] = occ
                            rm_try[target_pq] = eq1

            elif mode == 3:
                # Edge-targeted perturbation
                edge_costs = compute_edge_costs(m_try, eff_weights)
                for _, eq1, eq2 in edge_costs[:min(3, len(edge_costs))]:
                    pq1 = m_try[eq1]
                    pq2 = m_try[eq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_dist = self.distance_matrix[adj_pq][pq2]
                        old_dist = self.distance_matrix[pq1][pq2]
                        delta = new_dist - old_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    if best_swap_pair is not None:
                        lq_a, lq_b, pq_a, pq_b = best_swap_pair
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        break

            elif mode == 4:
                # LNS: destroy and reconstruct a neighborhood
                if len(interacting_logical) >= 6:
                    qcost = defaultdict(float)
                    for (eq1, eq2), w in eff_weights.items():
                        if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                            c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                            qcost[eq1] += c
                            qcost[eq2] += c
                    cost_sorted = sorted(interacting_logical, key=lambda q: qcost.get(q, 0.0), reverse=True)
                    destroy_size = min(max(3, len(interacting_logical) // 5), 6)
                    to_destroy = cost_sorted[:destroy_size]
                    freed_phys = [m_try[lq] for lq in to_destroy]
                    rng.shuffle(freed_phys)
                    for lq, pq in zip(to_destroy, freed_phys):
                        old_pq = m_try[lq]
                        m_try[lq] = pq
                        rm_try[pq] = lq

            run_swap_refinement(m_try, rm_try, eff_nbrs, max_rounds=3)

            o1_new = compute_O1(m_try, eff_weights)
            o2_new = compute_O2(m_try, eff_weights)

            # Add to Pareto front
            pareto_front = add_to_pareto_front(pareto_front,
                                                (o1_new, o2_new, list(m_try), list(rm_try)))

            # SA acceptance on O1
            delta = o1_new - current_o1
            reward = 0.0
            if delta < 0:
                current_m = list(m_try)
                current_rm = list(rm_try)
                current_o1 = o1_new
                reward = 1.0
                if o1_new < best_o1:
                    best_o1 = o1_new
                    best_m = list(m_try)
                    best_rm = list(rm_try)
                    reward = 2.0
            elif T > 1e-12 and rng.random() < math.exp(-delta / max(T, 1e-12)):
                current_m = list(m_try)
                current_rm = list(rm_try)
                current_o1 = o1_new
                reward = 0.3

            mode_counts[mode] += 1
            mode_rewards[mode] += reward

        return best_m, best_rm, pareto_front

    # ---------------------------------------------------------------
    # Step 11: Routing simulation (for RSDIWR and final selection)
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=25):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        total_swaps = 0

        if not gates_2q:
            return swap_counts, 0

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

    def simulate_routing_full(m, rm):
        """Full routing simulation to count total swaps — used for Pareto front selection."""
        sim_m = list(m)
        sim_rm = list(rm)
        total_swaps = 0

        if not gates_2q:
            return 0

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set()
        for g in gates_2q:
            if pred_remaining[g] == 0:
                front.add(g)

        sim_decay = [1.0] * num_q
        max_iters = len(gates_2q) * 20  # safety bound

        iters = 0
        while front and iters < max_iters:
            iters += 1
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

        return total_swaps

    # ---------------------------------------------------------------
    # Step 12: Build mapping helper
    # ---------------------------------------------------------------
    def build_neighbors_from_weights(weights):
        neighbors = defaultdict(dict)
        for (q1, q2), w in weights.items():
            neighbors[q1][q2] = w
            neighbors[q2][q1] = w
        return neighbors

    def build_best_mapping(eff_weights):
        eff_nbrs = build_neighbors_from_weights(eff_weights)

        candidates = []
        if seed_lqs and seed_pqs:
            for s_lq in seed_lqs:
                for s_pq in seed_pqs:
                    m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs)
                    fill_unmapped(m, rm)
                    cost = compute_O1(m, eff_weights)
                    candidates.append((cost, m, rm))

        # Add spectral seeds
        if spectral_order and seed_pqs:
            for s_pq in seed_pqs[:2]:
                m, rm = run_spectral_greedy(spectral_order, s_pq, eff_nbrs)
                fill_unmapped(m, rm)
                cost = compute_O1(m, eff_weights)
                candidates.append((cost, m, rm))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            num_to_refine = min(5, len(candidates))

            best_o1 = float('inf')
            best_m = None
            best_rm = None

            for idx in range(num_to_refine):
                _, m, rm = candidates[idx]
                m_c = list(m)
                rm_c = list(rm)

                run_swap_refinement(m_c, rm_c, eff_nbrs, max_rounds=4)

                o1 = compute_O1(m_c, eff_weights)
                if o1 < best_o1:
                    best_o1 = o1
                    best_m = m_c
                    best_rm = rm_c

            return best_m, best_rm, best_o1
        else:
            return list(range(num_q)), list(range(num_q)), float('inf')

    # ---------------------------------------------------------------
    # Step 13: RSDIWR with Pareto front
    # ---------------------------------------------------------------
    T_iters = 4
    alpha_schedule = [1.0, 0.65, 0.35, 0.15]
    swap_counts = defaultdict(float)
    pareto_front = []  # list of (o1, o2, m, rm)

    rng = np.random.RandomState(42)

    for t in range(T_iters):
        alpha_blend = alpha_schedule[t]

        # Build effective weights
        eff_weights = defaultdict(float)
        if t == 0 or not swap_counts:
            for key, w in interaction_weight.items():
                eff_weights[key] = w
        else:
            max_s = max(interaction_weight.values()) if interaction_weight else 1.0
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max_s / max(max_sw, 1e-10)

            a_keys = set(interaction_weight.keys()) | set(swap_counts.keys())
            for key in a_keys:
                w_s = interaction_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        eff_nbrs = build_neighbors_from_weights(eff_weights)

        # Build initial mapping for this round
        cur_m, cur_rm, cur_o1 = build_best_mapping(eff_weights)

        # Compute objectives and add to Pareto front
        o1 = compute_O1(cur_m, eff_weights)
        o2 = compute_O2(cur_m, eff_weights)
        pareto_front = add_to_pareto_front(pareto_front, (o1, o2, list(cur_m), list(cur_rm)))

        # Perturbation + SA with Pareto tracking
        cur_m, cur_rm, pareto_front = perturb_and_refine_pareto(
            cur_m, cur_rm, eff_weights, eff_nbrs, pareto_front, rng,
            num_perturbations=12
        )

        # Routing simulation for weight refinement
        if t < T_iters - 1:
            swap_counts, _ = simulate_routing(cur_m, cur_rm, max_layers=25)

    # ---------------------------------------------------------------
    # Step 14: Router-validated selection from Pareto front
    # ---------------------------------------------------------------
    if len(pareto_front) > 0:
        if len(pareto_front) == 1:
            best_m = pareto_front[0][2]
            best_rm = pareto_front[0][3]
        else:
            best_swaps = float('inf')
            best_m = None
            best_rm = None
            for entry in pareto_front:
                m_cand = entry[2]
                rm_cand = entry[3]
                swaps = simulate_routing_full(m_cand, rm_cand)
                if swaps < best_swaps:
                    best_swaps = swaps
                    best_m = m_cand
                    best_rm = rm_cand
    else:
        best_m = list(range(num_q))
        best_rm = list(range(num_q))

    # ---------------------------------------------------------------
    # Step 15: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)