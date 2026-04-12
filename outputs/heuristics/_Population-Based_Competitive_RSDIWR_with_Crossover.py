def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import math
    import random

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build DAG + topological sort + critical path
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
    # Step 2: Build TWO static weight schemes (A: critical-path, B: geometric)
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha_decay = 2.5

    logical_qubits_set = set()
    static_weight_A = defaultdict(float)
    logical_degree_A = defaultdict(float)
    static_weight_B = defaultdict(float)
    logical_degree_B = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))

        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w_cp = cp * (max_layer - layer + 1)
        static_weight_A[key] += w_cp
        logical_degree_A[q1] += w_cp
        logical_degree_A[q2] += w_cp

        r = topo_rank.get(g, 0)
        w_td = math.exp(-alpha_decay * r / total_gates)
        w_geo = math.sqrt(w_cp * w_td)
        static_weight_B[key] += w_geo
        logical_degree_B[q1] += w_geo
        logical_degree_B[q2] += w_geo

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)

    logical_degree_combined = defaultdict(float)
    for q in logical_qubits:
        logical_degree_combined[q] = logical_degree_A.get(q, 0) + logical_degree_B.get(q, 0)

    interacting_logical = [q for q in logical_qubits if logical_degree_combined.get(q, 0) > 0]

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

    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree_combined[q], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree_combined[q], reverse=True)
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

    # Unified evaluation weights
    max_A = max(static_weight_A.values()) if static_weight_A else 1.0
    max_B = max(static_weight_B.values()) if static_weight_B else 1.0
    static_weight_eval = defaultdict(float)
    all_static_keys = set(static_weight_A.keys()) | set(static_weight_B.keys())
    for key in all_static_keys:
        w_a = static_weight_A.get(key, 0.0) / max(max_A, 1e-10)
        w_b = static_weight_B.get(key, 0.0) / max(max_B, 1e-10)
        static_weight_eval[key] = (w_a + w_b) * 0.5

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

    def perturb_and_refine_sa(m, rm, weights, logical_nbrs, num_perturbations=3):
        if len(interacting_logical) < 4:
            return m, rm
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m, weights)
        current_m = list(m)
        current_rm = list(rm)
        current_cost = best_cost

        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        T_init = max(best_cost * 0.02, 1.0)
        T_min = T_init * 0.01

        def compute_edge_costs(mc):
            edge_costs = []
            for (eq1, eq2), w in weights.items():
                if mc[eq1] >= 0 and mc[eq2] >= 0:
                    c = w * self.distance_matrix[mc[eq1]][mc[eq2]]
                    edge_costs.append((c, eq1, eq2))
            edge_costs.sort(reverse=True)
            return edge_costs

        for p_idx in range(num_perturbations):
            m_try = list(current_m)
            rm_try = list(current_rm)
            mode = p_idx % 4

            T = T_init * (T_min / T_init) ** (p_idx / max(num_perturbations - 1, 1))

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
                        dd = new_dist - current_dist
                        if dd < best_delta:
                            best_delta = dd
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    for adj_pq in hw_adj[pq2]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq1:
                            continue
                        new_dist = self.distance_matrix[pq1][adj_pq]
                        dd = new_dist - current_dist
                        if dd < best_delta:
                            best_delta = dd
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

            elif mode == 2 and len(interacting_logical) >= 6:
                qcost = defaultdict(float)
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
                        qcost[eq1] += c
                        qcost[eq2] += c
                cost_pairs = [(qcost.get(q, 0.0), q) for q in interacting_logical]
                cost_pairs.sort(reverse=True)
                top3 = [q for _, q in cost_pairs[:min(5, len(cost_pairs))]]
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

            delta = cost - current_cost
            if delta < 0 or (T > 1e-12 and rng.random() < math.exp(-delta / T)):
                current_m = list(m_try)
                current_rm = list(rm_try)
                current_cost = cost

            if cost < best_cost:
                best_cost = cost
                best_m = list(m_try)
                best_rm = list(rm_try)

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 6: Routing simulation for individual population members
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
    # Step 7: PMX Crossover — produces valid permutation offspring
    # ---------------------------------------------------------------
    def pmx_crossover(parent1, parent2, rng):
        """Partially Mapped Crossover on mapping arrays (logical->physical).
        Only crosses over logical_qubits positions; non-logical positions stay identity."""
        n = len(parent1)
        offspring = [-1] * n

        # Choose crossover segment within logical qubits indices
        if len(logical_qubits) < 2:
            return list(parent1), list(parent2)

        # Use logical qubit positions for crossover
        lqs = list(logical_qubits)
        size = len(lqs)
        cx1 = rng.randint(0, size - 1)
        cx2 = rng.randint(cx1 + 1, size)  # cx2 > cx1

        segment_lqs = lqs[cx1:cx2]

        # Build offspring1: copy segment from parent1, fill from parent2
        o1 = [-1] * n
        o1_reverse = [-1] * n

        # Copy segment from parent1
        used_phys = set()
        for lq in segment_lqs:
            o1[lq] = parent1[lq]
            used_phys.add(parent1[lq])

        # PMX repair: for remaining logical qubits, try to copy from parent2
        # If conflict, follow the mapping chain in parent1 segment
        p1_to_p2 = {}  # mapping from parent1[lq] -> parent2[lq] for segment
        for lq in segment_lqs:
            p1_to_p2[parent1[lq]] = parent2[lq]

        remaining_lqs = [lq for lq in lqs if lq not in segment_lqs]
        for lq in remaining_lqs:
            val = parent2[lq]
            # Follow chain until we find unused physical qubit
            visited = set()
            while val in used_phys:
                if val in visited:
                    break
                visited.add(val)
                # val is taken by some lq_s in segment where parent1[lq_s] = val
                # Find lq_s such that parent1[lq_s] = val -> use parent2[lq_s]
                if val in p1_to_p2:
                    val = p1_to_p2[val]
                else:
                    break
            if val not in used_phys:
                o1[lq] = val
                used_phys.add(val)

        # Any remaining logical qubits that still have -1: assign from free physicals
        free_phys = [pq for pq in range(n) if pq not in used_phys]
        free_idx = 0
        for lq in lqs:
            if o1[lq] == -1:
                o1[lq] = free_phys[free_idx]
                used_phys.add(free_phys[free_idx])
                free_idx += 1

        # Fill non-logical qubits
        non_logical = [q for q in range(n) if q not in logical_qubits_set]
        free_phys_remaining = sorted(set(range(n)) - used_phys)
        for i, lq in enumerate(non_logical):
            if i < len(free_phys_remaining):
                o1[lq] = free_phys_remaining[i]
            else:
                o1[lq] = lq  # fallback

        # Build reverse mapping
        for i in range(n):
            if o1[i] >= 0:
                o1_reverse[o1[i]] = i

        # Build offspring2: segment from parent2, fill from parent1 (symmetric)
        o2 = [-1] * n
        o2_reverse = [-1] * n
        used_phys2 = set()

        for lq in segment_lqs:
            o2[lq] = parent2[lq]
            used_phys2.add(parent2[lq])

        p2_to_p1 = {}
        for lq in segment_lqs:
            p2_to_p1[parent2[lq]] = parent1[lq]

        for lq in remaining_lqs:
            val = parent1[lq]
            visited = set()
            while val in used_phys2:
                if val in visited:
                    break
                visited.add(val)
                if val in p2_to_p1:
                    val = p2_to_p1[val]
                else:
                    break
            if val not in used_phys2:
                o2[lq] = val
                used_phys2.add(val)

        free_phys2 = [pq for pq in range(n) if pq not in used_phys2]
        free_idx2 = 0
        for lq in lqs:
            if o2[lq] == -1:
                o2[lq] = free_phys2[free_idx2]
                used_phys2.add(free_phys2[free_idx2])
                free_idx2 += 1

        free_phys2_remaining = sorted(set(range(n)) - used_phys2)
        for i, lq in enumerate(non_logical):
            if i < len(free_phys2_remaining):
                o2[lq] = free_phys2_remaining[i]
            else:
                o2[lq] = lq

        for i in range(n):
            if o2[i] >= 0:
                o2_reverse[o2[i]] = i

        return o1, o2

    def build_reverse(m):
        rm = [-1] * len(m)
        for i, v in enumerate(m):
            if v >= 0:
                rm[v] = i
        return rm

    # ---------------------------------------------------------------
    # Step 8: Population-Based Competitive RSDIWR with Crossover
    # ---------------------------------------------------------------
    P = 6  # Population size
    T_outer = 4  # Outer iterations
    alpha_schedule = [1.0, 0.65, 0.35, 0.15]

    rng = np.random.RandomState(42)

    # Phase 1: Generate P=6 diverse initial mappings
    # 3 from weight scheme A (different seed pairs), 3 from weight scheme B
    population = []  # Each entry: (m, rm, own_swap_counts)

    for scheme_idx, static_w in enumerate([static_weight_A, static_weight_B]):
        eff_nbrs, eff_deg = build_neighbors_from_weights(static_w)

        # Generate 3 mappings from different seed pairs
        generated = 0
        for s_lq in seed_lqs:
            if generated >= 3:
                break
            for s_pq in seed_pqs:
                if generated >= 3:
                    break
                m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs, eff_deg)
                fill_unmapped(m, rm)
                run_swap_refinement(m, rm, eff_nbrs, max_rounds=3)
                population.append({
                    'm': m,
                    'rm': rm,
                    'swap_counts': defaultdict(float),
                    'scheme': scheme_idx,
                    'static_w': static_w,
                })
                generated += 1

    # Ensure we have exactly P members (pad with trivial if needed)
    while len(population) < P:
        m = list(range(num_q))
        rm = list(range(num_q))
        population.append({
            'm': m,
            'rm': rm,
            'swap_counts': defaultdict(float),
            'scheme': 0,
            'static_w': static_weight_A,
        })

    # Trim to P
    population = population[:P]

    # Phase 2-5: RSDIWR outer loop with population
    for t in range(T_outer):
        alpha_blend = alpha_schedule[t]

        # Phase 2: Run routing simulation on all P mappings, get per-pair SWAP counts
        sim_results = []  # (total_swaps, index)
        for idx, member in enumerate(population):
            sc, total_sw = simulate_routing(member['m'], member['rm'], max_layers=25)
            member['swap_counts'] = sc
            sim_results.append((total_sw, idx))

        # Phase 3: Crossover — select top-2, produce 2 offspring via PMX
        sim_results.sort(key=lambda x: x[0])
        top2_indices = [sim_results[0][1], sim_results[1][1]]
        bottom2_indices = [sim_results[-1][1], sim_results[-2][1]]

        parent1_m = population[top2_indices[0]]['m']
        parent2_m = population[top2_indices[1]]['m']

        o1_m, o2_m = pmx_crossover(parent1_m, parent2_m, rng)
        o1_rm = build_reverse(o1_m)
        o2_rm = build_reverse(o2_m)

        # Phase 4: Replace bottom-2 with offspring
        # Inherit scheme from the closer parent
        population[bottom2_indices[0]] = {
            'm': o1_m,
            'rm': o1_rm,
            'swap_counts': defaultdict(float),
            'scheme': population[top2_indices[0]]['scheme'],
            'static_w': population[top2_indices[0]]['static_w'],
        }
        population[bottom2_indices[1]] = {
            'm': o2_m,
            'rm': o2_rm,
            'swap_counts': defaultdict(float),
            'scheme': population[top2_indices[1]]['scheme'],
            'static_w': population[top2_indices[1]]['static_w'],
        }

        # Phase 5: Update each mapping's weight matrix using its own feedback,
        # then apply ILS+SA refinement (reduced budget: ~1.5 perturbations per mapping)
        for idx, member in enumerate(population):
            static_w = member['static_w']
            sc = member['swap_counts']

            # Blend static and empirical weights
            eff_weights = defaultdict(float)
            if t == 0 or not sc:
                for key, w in static_w.items():
                    eff_weights[key] = w
            else:
                max_s = max(static_w.values()) if static_w else 1.0
                max_sw = max(sc.values()) if sc else 1.0
                scale = max_s / max(max_sw, 1e-10)
                a_keys = set(static_w.keys()) | set(sc.keys())
                for key in a_keys:
                    w_s = static_w.get(key, 0.0)
                    w_r = sc.get(key, 0.0) * scale
                    eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

            eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)

            # Swap refinement
            run_swap_refinement(member['m'], member['rm'], eff_nbrs, max_rounds=3)

            # Reduced ILS+SA (budget = ~1.5 perturbations per mapping to maintain total compute)
            member['m'], member['rm'] = perturb_and_refine_sa(
                member['m'], member['rm'], eff_weights, eff_nbrs, num_perturbations=2)

    # ---------------------------------------------------------------
    # Step 9: Select best mapping across entire population
    # ---------------------------------------------------------------
    best_overall_m = None
    best_overall_rm = None
    best_overall_cost = float('inf')

    for member in population:
        # Evaluate with unified eval weights
        cost = compute_total_cost(member['m'], static_weight_eval)
        # Also run a final routing simulation to get swap count as tiebreaker
        _, total_sw = simulate_routing(member['m'], member['rm'], max_layers=30)

        # Primary: eval cost, secondary: simulated swaps
        score = (cost, total_sw)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_m = list(member['m'])
            best_overall_rm = list(member['rm'])

    # Final refinement on the overall best
    if best_overall_m is not None:
        eval_nbrs, _ = build_neighbors_from_weights(static_weight_eval)
        run_swap_refinement(best_overall_m, best_overall_rm, eval_nbrs, max_rounds=4)
        best_overall_m, best_overall_rm = perturb_and_refine_sa(
            best_overall_m, best_overall_rm, static_weight_eval, eval_nbrs, num_perturbations=5)

    if best_overall_m is None:
        best_overall_m = list(range(num_q))
        best_overall_rm = list(range(num_q))

    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)