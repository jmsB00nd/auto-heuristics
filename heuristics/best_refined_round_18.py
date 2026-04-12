def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

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
    # Step 2: Build temporal-decay weighted interaction graph (alpha=2.5)
    # ---------------------------------------------------------------
    alpha = 2.5
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

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
    max_iw = max(interaction_weight.values()) if interaction_weight else 1.0

    # ---------------------------------------------------------------
    # Step 4: Multi-seed greedy placement with connectivity-matching
    #   tie-breaker and dynamic adjacency bonus (from both parents)
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

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    def dynamic_adjacency_bonus(iw):
        ratio = iw / max_iw
        return 0.90 - 0.10 * ratio

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
                            cost *= dynamic_adjacency_bonus(iw)
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

    def compute_total_cost(m):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    # ---------------------------------------------------------------
    # Step 5: Pairwise swap refinement with delta-cost computation
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
                    affected.update(logical_neighbors[lq_a].keys())
                    affected.update(logical_neighbors[lq_b].keys())

                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = m[q]
                        w_a = logical_neighbors[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                        w_b = logical_neighbors[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    # ---------------------------------------------------------------
    # Step 6: CROSSOVER — Three-mode perturbation combining:
    #   Mode 0 (NEW — edge-targeted): Find the highest-cost
    #     interaction EDGE (pair), then for each endpoint, check
    #     hardware-adjacent positions for a swap that reduces that
    #     edge's contribution. This directly attacks the dominant
    #     cost contributor at the granularity of edges, not qubits.
    #   Mode 1 (from Parent 2 — neighborhood-aware): Find the
    #     highest-cost qubit, swap with the lowest-cost occupant
    #     of an adjacent physical position. Broader than mode 0.
    #   Mode 2 (from Parent 1 — random): Pure random swap pairs
    #     for exploration diversity to escape structured local minima.
    # ---------------------------------------------------------------
    def compute_qubit_cost(m):
        qcost = defaultdict(float)
        for (q1, q2), w in interaction_weight.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * self.distance_matrix[m[q1]][m[q2]]
                qcost[q1] += c
                qcost[q2] += c
        return qcost

    def compute_edge_costs(m):
        """Compute cost per interaction edge for edge-targeted perturbation."""
        edge_costs = []
        for (q1, q2), w in interaction_weight.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * self.distance_matrix[m[q1]][m[q2]]
                edge_costs.append((c, q1, q2))
        edge_costs.sort(reverse=True)
        return edge_costs

    def perturb_and_refine(m, rm, num_perturbations=5):
        if len(interacting_logical) < 4:
            return m, rm
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m)

        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        interacting_set = set(interacting_logical)

        for p_idx in range(num_perturbations):
            m_try = list(best_m)
            rm_try = list(best_rm)
            mode = p_idx % 3

            if mode == 0 and len(interacting_logical) >= 4:
                # Edge-targeted perturbation (CROSSOVER NOVELTY)
                # Find the highest-cost edge, then try swapping one of
                # its endpoints with an adjacent physical occupant to
                # directly reduce that edge's distance.
                edge_costs = compute_edge_costs(m_try)
                swapped = False
                for _, eq1, eq2 in edge_costs[:min(3, len(edge_costs))]:
                    pq1 = m_try[eq1]
                    pq2 = m_try[eq2]
                    current_dist = self.distance_matrix[pq1][pq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    # Try swapping eq1 with each of its physical neighbors' occupants
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        # After swap: eq1->adj_pq, occ->pq1
                        new_dist = self.distance_matrix[adj_pq][pq2]
                        # Approximate delta: only consider the target edge
                        delta = new_dist - current_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    # Also try swapping eq2 with its physical neighbors' occupants
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
                        swapped = True
                        break
                # Add a random swap for exploration even when edge-targeted succeeds
                if len(interacting_logical) >= 4:
                    indices = rng.choice(len(interacting_logical), size=2, replace=False)
                    lq_c = interacting_logical[indices[0]]
                    lq_d = interacting_logical[indices[1]]
                    pq_c, pq_d = m_try[lq_c], m_try[lq_d]
                    m_try[lq_c] = pq_d
                    m_try[lq_d] = pq_c
                    rm_try[pq_c] = lq_d
                    rm_try[pq_d] = lq_c

            elif mode == 1 and len(interacting_logical) >= 4:
                # Neighborhood-aware swap (from Parent 2)
                # Find highest-cost qubit, swap with lowest-cost
                # occupant of an adjacent physical position
                qcost = compute_qubit_cost(m_try)
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
                    # Fallback to random
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
                # Pure random perturbation (from Parent 1)
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

            run_swap_refinement(m_try, rm_try, max_rounds=3)
            cost = compute_total_cost(m_try)
            if cost < best_cost:
                best_cost = cost
                best_m = m_try
                best_rm = rm_try

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 7: Multi-start execution — refine top-4 candidates with
    #   4 rounds swap refinement + 5 three-mode perturbations
    # ---------------------------------------------------------------
    candidates = []

    if seed_lqs and seed_pqs:
        for s_lq in seed_lqs:
            for s_pq in seed_pqs:
                m, rm = run_greedy_placement(s_lq, s_pq)
                fill_unmapped(m, rm)
                cost = compute_total_cost(m)
                candidates.append((cost, m, rm))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        num_to_refine = min(4, len(candidates))

        best_cost = float('inf')
        best_mapping = None
        best_reverse = None

        for idx in range(num_to_refine):
            _, m, rm = candidates[idx]
            m_copy = list(m)
            rm_copy = list(rm)

            # Phase 1: Pairwise swap refinement (4 rounds)
            run_swap_refinement(m_copy, rm_copy, max_rounds=4)

            # Phase 2: Three-mode perturbation restarts
            m_copy, rm_copy = perturb_and_refine(m_copy, rm_copy, num_perturbations=5)

            cost = compute_total_cost(m_copy)
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