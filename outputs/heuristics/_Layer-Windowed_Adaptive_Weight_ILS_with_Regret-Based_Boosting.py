def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

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
    topo_order = []
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        topo_order.append(g)
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    # ---------------------------------------------------------------
    # Step 2: Build base interaction weights with temporal decay
    # ---------------------------------------------------------------
    alpha = 2.5
    logical_qubits_set = set()
    base_interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    two_qubit_gates = []  # (gate_id, q1, q2) in topo order

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            base_interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
            two_qubit_gates.append((gate, q1, q2))
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    # Sort 2q gates by topological rank for layer windowing
    two_qubit_gates.sort(key=lambda x: topo_rank.get(x[0], 0))

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in base_interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Partition 2q gates into layer windows of K=10
    # ---------------------------------------------------------------
    K_WINDOW = 10
    # Assign each 2q gate to a layer based on topo_rank
    # Group into windows of K layers
    if two_qubit_gates:
        max_rank = max(topo_rank.get(g, 0) for g, _, _ in two_qubit_gates)
    else:
        max_rank = 0

    # Build windows: each window contains pairs from K consecutive layers
    windows = []
    if two_qubit_gates:
        num_layers = max_rank + 1
        num_windows = max((num_layers + K_WINDOW - 1) // K_WINDOW, 1)
        for wi in range(num_windows):
            start_layer = wi * K_WINDOW
            end_layer = start_layer + K_WINDOW
            window_pairs = []
            for gate, q1, q2 in two_qubit_gates:
                r = topo_rank.get(gate, 0)
                if start_layer <= r < end_layer:
                    window_pairs.append((q1, q2))
            if window_pairs:
                windows.append(window_pairs)

    # ---------------------------------------------------------------
    # Step 4: Precompute physical graph properties
    # ---------------------------------------------------------------
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}

    max_iw = max(base_interaction_weight.values()) if base_interaction_weight else 1.0

    # ---------------------------------------------------------------
    # Step 5: Multi-seed greedy construction with connectivity matching
    # ---------------------------------------------------------------
    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
        num_seed_lq = min(3, len(sorted_by_degree))
        seed_lqs = sorted_by_degree[:num_seed_lq]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    num_seed_pq = min(3, len(phys_by_centrality))
    seed_pqs = phys_by_centrality[:num_seed_pq]

    # Degree rank for connectivity matching tie-break
    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    def run_greedy_placement(start_lq, start_pq):
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
                w = sum(logical_neighbors[lq].get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            neighbors_placed = {plq: logical_neighbors[best_lq].get(plq, 0.0)
                                for plq in placed if plq in logical_neighbors[best_lq]}

            if neighbors_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in neighbors_placed.items():
                        dist = self.distance_matrix[pq][m[plq]]
                        cost = iw * dist
                        if m[plq] in hw_adj[pq]:
                            ratio = iw / max_iw
                            cost *= (0.90 - 0.10 * ratio)
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
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rank_norm), x[0]))[1]
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

    # ---------------------------------------------------------------
    # Step 6: Adaptive-weight cost function with regret boosting
    # ---------------------------------------------------------------
    def compute_regret_weights(m, current_weights):
        """Compute regret for each window and boost/dampen weights."""
        new_weights = dict(current_weights)

        for window_pairs in windows:
            regrets = {}
            max_regret = 0
            for q1, q2 in window_pairs:
                key = (min(q1, q2), max(q1, q2))
                if m[q1] >= 0 and m[q2] >= 0:
                    d = self.distance_matrix[m[q1]][m[q2]]
                    regret = max(d - 1, 0)
                    if key not in regrets:
                        regrets[key] = 0
                    regrets[key] += regret
                    max_regret = max(max_regret, regrets[key])

            if max_regret > 0:
                for key, regret in regrets.items():
                    if regret > 0:
                        # Boost pairs with high regret
                        boost = 1.0 + regret / max_regret
                        new_weights[key] = new_weights.get(key, 0.0) * boost
                    else:
                        # Dampen pairs already adjacent
                        new_weights[key] = new_weights.get(key, 0.0) * 0.5

        return new_weights

    def compute_cost_with_weights(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    # ---------------------------------------------------------------
    # Step 7: ILS with adaptive weight updates
    # ---------------------------------------------------------------
    def run_adaptive_ils(m, rm, max_iterations=200):
        """ILS/SA refinement with adaptive regret-based weight boosting."""
        if len(interacting_logical) <= 1:
            return m, rm

        current_weights = dict(base_interaction_weight)
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_cost_with_weights(m, current_weights)

        m_cur = list(m)
        rm_cur = list(rm)
        cur_cost = best_cost

        n_interact = len(interacting_logical)
        rng_seed = int(abs(best_cost * 1000)) % (2**31)
        rng = np.random.RandomState(rng_seed)

        # SA temperature schedule
        T = best_cost * 0.1 if best_cost > 0 else 1.0
        T_min = T * 0.001
        cooling = (T_min / T) ** (1.0 / max(max_iterations, 1))

        weight_update_interval = 50

        for iteration in range(max_iterations):
            # Adaptive weight update every 50 iterations
            if iteration > 0 and iteration % weight_update_interval == 0:
                current_weights = compute_regret_weights(m_cur, base_interaction_weight)
                cur_cost = compute_cost_with_weights(m_cur, current_weights)
                best_cost = compute_cost_with_weights(best_m, current_weights)

            # Choose move type
            move_type = rng.randint(0, 3)

            if move_type == 0:
                # Pairwise swap of two interacting logical qubits
                if n_interact < 2:
                    continue
                idx = rng.choice(n_interact, size=2, replace=False)
                lq_a = interacting_logical[idx[0]]
                lq_b = interacting_logical[idx[1]]
                pq_a = m_cur[lq_a]
                pq_b = m_cur[lq_b]

                # Delta cost computation
                delta = 0.0
                affected = set()
                affected.update(logical_neighbors[lq_a].keys())
                affected.update(logical_neighbors[lq_b].keys())

                for q in affected:
                    if q == lq_a or q == lq_b:
                        continue
                    pq_q = m_cur[q]
                    key_a = (min(lq_a, q), max(lq_a, q))
                    key_b = (min(lq_b, q), max(lq_b, q))
                    w_a = current_weights.get(key_a, 0.0)
                    w_b = current_weights.get(key_b, 0.0)
                    if w_a > 0:
                        delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                    if w_b > 0:
                        delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                # Also account for the direct interaction between lq_a and lq_b
                key_ab = (min(lq_a, lq_b), max(lq_a, lq_b))
                # Distance doesn't change for a<->b swap, so no delta from direct pair

            elif move_type == 1:
                # Swap interacting qubit with a neighbor on the physical graph
                if n_interact < 1:
                    continue
                lq_a = interacting_logical[rng.randint(0, n_interact)]
                pq_a = m_cur[lq_a]
                neighbors = list(hw_adj[pq_a])
                if not neighbors:
                    continue
                pq_b = neighbors[rng.randint(0, len(neighbors))]
                lq_b = rm_cur[pq_b]

                delta = 0.0
                affected = set()
                affected.update(logical_neighbors.get(lq_a, {}).keys())
                affected.update(logical_neighbors.get(lq_b, {}).keys())

                for q in affected:
                    if q == lq_a or q == lq_b:
                        continue
                    pq_q = m_cur[q]
                    key_a = (min(lq_a, q), max(lq_a, q))
                    key_b = (min(lq_b, q), max(lq_b, q))
                    w_a = current_weights.get(key_a, 0.0)
                    w_b = current_weights.get(key_b, 0.0)
                    if w_a > 0:
                        delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                    if w_b > 0:
                        delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

            else:
                # 3-opt cyclic permutation: a->b->c->a
                if n_interact < 3:
                    continue
                idx = rng.choice(n_interact, size=3, replace=False)
                lq_a = interacting_logical[idx[0]]
                lq_b = interacting_logical[idx[1]]
                lq_c = interacting_logical[idx[2]]
                pq_a = m_cur[lq_a]
                pq_b = m_cur[lq_b]
                pq_c = m_cur[lq_c]

                # Compute cost before and after for the three qubits
                # After cyclic: a->pq_c, b->pq_a, c->pq_b
                triplet = {lq_a, lq_b, lq_c}
                affected = set()
                for lq in triplet:
                    affected.update(logical_neighbors.get(lq, {}).keys())

                delta = 0.0
                new_pos = {lq_a: pq_c, lq_b: pq_a, lq_c: pq_b}

                for q in affected:
                    pq_q = m_cur[q]
                    for lq in triplet:
                        if q == lq:
                            continue
                        if q in triplet:
                            continue
                        key = (min(lq, q), max(lq, q))
                        w = current_weights.get(key, 0.0)
                        if w > 0:
                            old_d = self.distance_matrix[m_cur[lq]][pq_q]
                            new_d = self.distance_matrix[new_pos[lq]][pq_q]
                            delta += w * (new_d - old_d)

                # Internal edges among the triplet
                for i_t, lq_i in enumerate([lq_a, lq_b, lq_c]):
                    for lq_j in [lq_a, lq_b, lq_c][i_t+1:]:
                        key = (min(lq_i, lq_j), max(lq_i, lq_j))
                        w = current_weights.get(key, 0.0)
                        if w > 0:
                            old_d = self.distance_matrix[m_cur[lq_i]][m_cur[lq_j]]
                            new_d = self.distance_matrix[new_pos[lq_i]][new_pos[lq_j]]
                            delta += w * (new_d - old_d)

                # Apply cyclic move if accepted
                if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                    m_cur[lq_a] = pq_c
                    m_cur[lq_b] = pq_a
                    m_cur[lq_c] = pq_b
                    rm_cur[pq_a] = lq_b
                    rm_cur[pq_b] = lq_c
                    rm_cur[pq_c] = lq_a
                    cur_cost += delta

                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_m = list(m_cur)
                        best_rm = list(rm_cur)

                T *= cooling
                continue

            # Accept/reject for move types 0 and 1
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                # Apply swap
                if move_type == 0:
                    lq_swap_a, lq_swap_b = lq_a, lq_b
                else:
                    lq_swap_a, lq_swap_b = lq_a, lq_b
                pq_sa = m_cur[lq_swap_a]
                pq_sb = m_cur[lq_swap_b]
                m_cur[lq_swap_a] = pq_sb
                m_cur[lq_swap_b] = pq_sa
                rm_cur[pq_sa] = lq_swap_b
                rm_cur[pq_sb] = lq_swap_a
                cur_cost += delta

                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_m = list(m_cur)
                    best_rm = list(rm_cur)

            T *= cooling

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 8: Multi-start execution
    # ---------------------------------------------------------------
    candidates = []

    if seed_lqs and seed_pqs:
        for s_lq in seed_lqs:
            for s_pq in seed_pqs:
                m, rm = run_greedy_placement(s_lq, s_pq)
                fill_unmapped(m, rm)
                cost = compute_cost_with_weights(m, base_interaction_weight)
                candidates.append((cost, m, rm))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        num_to_refine = min(4, len(candidates))

        best_cost = float('inf')
        best_mapping = None
        best_reverse = None

        # Scale ILS iterations based on problem size
        n_interact = len(interacting_logical)
        if n_interact <= 10:
            ils_iters = 150
        elif n_interact <= 30:
            ils_iters = 200
        elif n_interact <= 60:
            ils_iters = 250
        else:
            ils_iters = 300

        for idx in range(num_to_refine):
            _, m, rm = candidates[idx]
            m_copy = list(m)
            rm_copy = list(rm)

            # Run adaptive ILS with regret-based weight boosting
            m_copy, rm_copy = run_adaptive_ils(m_copy, rm_copy, max_iterations=ils_iters)

            cost = compute_cost_with_weights(m_copy, base_interaction_weight)
            if cost < best_cost:
                best_cost = cost
                best_mapping = m_copy
                best_reverse = rm_copy

        mapping_dict = best_mapping
        reverse_mapping_dict = best_reverse
    else:
        mapping_dict = list(range(num_q))
        reverse_mapping_dict = list(range(num_q))

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)