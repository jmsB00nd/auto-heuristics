def init_mapping(self):
    import random
    import math
    from collections import defaultdict, deque

    num_q = self.num_qubits
    backend = self.backend
    dist = self.distance_matrix

    # ── 1. Classify physical qubits by hardware degree ──
    phys_degree = {}
    for pq in backend:
        phys_degree[pq] = len(backend[pq])

    degree3_pqs = sorted([p for p, d in phys_degree.items() if d >= 3],
                         key=lambda p: sum(dist[p].values()) if isinstance(dist[p], dict) else sum(dist[p]))
    degree2_pqs = [p for p, d in phys_degree.items() if d == 2]
    degree1_pqs = [p for p, d in phys_degree.items() if d <= 1]

    # If no degree-1 nodes exist, split degree-2 into two groups
    if not degree1_pqs and not degree3_pqs:
        degree3_pqs = degree2_pqs[:len(degree2_pqs)//3]
        degree1_pqs = degree2_pqs[2*len(degree2_pqs)//3:]
        degree2_pqs = degree2_pqs[len(degree2_pqs)//3:2*len(degree2_pqs)//3]
    elif not degree3_pqs:
        degree3_pqs = degree2_pqs[:len(degree2_pqs)//2]
        degree2_pqs = degree2_pqs[len(degree2_pqs)//2:]

    # Build degree lookup for each physical qubit
    pq_stratum = {}
    for p in degree3_pqs:
        pq_stratum[p] = 3
    for p in degree2_pqs:
        pq_stratum[p] = 2
    for p in degree1_pqs:
        pq_stratum[p] = 1

    # ── 2. Identify logical qubits and compute interaction weights ──
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    pair_weight = defaultdict(float)

    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_qubits_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits
            pair_weight[(q1, q2)] += 1.0
            pair_weight[(q2, q1)] += 1.0
            interaction_weight[q1] += 1.0
            interaction_weight[q2] += 1.0

    logical_qubits = sorted(logical_qubits_set)
    num_logical = len(logical_qubits)

    # Rank logical qubits by weighted interaction degree (descending)
    sorted_logical = sorted(logical_qubits, key=lambda lq: interaction_weight.get(lq, 0), reverse=True)

    # ── 3. Stratified assignment of logical qubits to strata ──
    n3 = len(degree3_pqs)
    n2 = len(degree2_pqs)
    n1 = len(degree1_pqs)

    # Assign top-k to degree-3, next batch to degree-2, rest to degree-1
    stratum3_lqs = sorted_logical[:min(n3, num_logical)]
    stratum2_lqs = sorted_logical[min(n3, num_logical):min(n3 + n2, num_logical)]
    stratum1_lqs = sorted_logical[min(n3 + n2, num_logical):]

    # Ensure we have enough physical qubits in each stratum
    # Overflow from degree-1 goes to degree-2, then degree-3
    if len(stratum1_lqs) > n1:
        overflow = stratum1_lqs[n1:]
        stratum1_lqs = stratum1_lqs[:n1]
        stratum2_lqs = stratum2_lqs + overflow
    if len(stratum2_lqs) > n2:
        overflow = stratum2_lqs[n2:]
        stratum2_lqs = stratum2_lqs[:n2]
        stratum3_lqs = stratum3_lqs + overflow

    # ── 4. BFS greedy placement within each stratum ──
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    used_phys = set()

    def centrality_score(pq, pq_list):
        """Lower total distance to other nodes in list = more central."""
        s = 0
        for other in pq_list:
            if other != pq:
                d = dist[pq][other] if isinstance(dist[pq], dict) else dist[pq][other]
                s += d
        return s

    def bfs_place(lq_list, pq_pool):
        """Place logical qubits via BFS expansion on hardware graph."""
        if not lq_list or not pq_pool:
            return
        pq_set = set(pq_pool) - used_phys
        if not pq_set:
            return

        # Sort logical qubits by interaction degree descending
        lqs = sorted(lq_list, key=lambda lq: interaction_weight.get(lq, 0), reverse=True)

        # Seed: most central physical qubit in pool for highest-interaction logical qubit
        available = list(pq_set)
        seed_pq = min(available, key=lambda p: centrality_score(p, available))

        placed_lqs = set()
        # Place first logical qubit at seed
        mapping_dict[lqs[0]] = seed_pq
        reverse_mapping_dict[seed_pq] = lqs[0]
        used_phys.add(seed_pq)
        pq_set.discard(seed_pq)
        placed_lqs.add(lqs[0])

        # BFS frontier from placed physical qubits
        bfs_queue = deque([seed_pq])
        bfs_visited = {seed_pq}

        lq_idx = 1
        while lq_idx < len(lqs) and pq_set:
            # Expand BFS to find next available physical qubit
            next_pq = None
            while bfs_queue and next_pq is None:
                current = bfs_queue.popleft()
                neighbors = backend.get(current, [])
                if isinstance(neighbors, set):
                    neighbors = list(neighbors)
                for nbr in neighbors:
                    if nbr not in bfs_visited:
                        bfs_visited.add(nbr)
                        bfs_queue.append(nbr)
                        if nbr in pq_set:
                            next_pq = nbr
                            break

            if next_pq is None:
                # Fallback: pick closest available to any placed qubit
                if pq_set:
                    next_pq = min(pq_set, key=lambda p: min(
                        dist[p][mapping_dict[lq]] if isinstance(dist[p], dict) else dist[p][mapping_dict[lq]]
                        for lq in placed_lqs
                    ))
                else:
                    break

            # Choose best logical qubit for this physical position
            # Among unplaced, pick one with highest adjacency bonus to already-placed neighbors
            best_lq = None
            best_score = -1
            for li in range(lq_idx, len(lqs)):
                lq_cand = lqs[li]
                score = 0
                for placed_lq in placed_lqs:
                    pw = pair_weight.get((lq_cand, placed_lq), 0)
                    if pw > 0:
                        placed_phys = mapping_dict[placed_lq]
                        d = dist[next_pq][placed_phys] if isinstance(dist[next_pq], dict) else dist[next_pq][placed_phys]
                        score += pw / max(d, 1)
                if score > best_score:
                    best_score = score
                    best_lq = li

            if best_lq is None:
                best_lq = lq_idx

            # Swap to front
            lqs[lq_idx], lqs[best_lq] = lqs[best_lq], lqs[lq_idx]
            lq = lqs[lq_idx]

            mapping_dict[lq] = next_pq
            reverse_mapping_dict[next_pq] = lq
            used_phys.add(next_pq)
            pq_set.discard(next_pq)
            placed_lqs.add(lq)
            lq_idx += 1

    # Place each stratum
    bfs_place(stratum3_lqs, degree3_pqs)
    bfs_place(stratum2_lqs, degree2_pqs)
    bfs_place(stratum1_lqs, degree1_pqs)

    # ── 5. Fill any remaining unmapped logical qubits ──
    unmapped_logical = [lq for lq in range(num_q) if mapping_dict[lq] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    random.shuffle(free_physical)
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # ── 6. ILS + SA refinement ──
    def compute_cost(md):
        cost = 0.0
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                q1, q2 = qubits
                p1, p2 = md[q1], md[q2]
                d = dist[p1][p2] if isinstance(dist[p1], dict) else dist[p1][p2]
                cost += d
        return cost

    current_cost = compute_cost(mapping_dict)
    best_mapping = mapping_dict[:]
    best_reverse = reverse_mapping_dict[:]
    best_cost = current_cost

    # Build stratum membership for logical qubits based on assigned physical qubit
    def get_lq_stratum(lq, md):
        pq = md[lq]
        return pq_stratum.get(pq, 0)

    num_2q_gates = sum(1 for g in self.access.values() if len(g) == 2)
    max_iters = min(max(num_2q_gates * 8, 500), 15000)
    T0 = max(current_cost * 0.15, 1.0)
    T_min = 0.01

    stratum_unlock_iter = int(max_iters * 0.3)

    for it in range(max_iters):
        T = T0 * (T_min / T0) ** (it / max(max_iters - 1, 1))

        # Pick two logical qubits to swap
        if it < stratum_unlock_iter:
            # Same-stratum swaps only
            strata_groups = defaultdict(list)
            for lq in logical_qubits:
                s = pq_stratum.get(mapping_dict[lq], 0)
                strata_groups[s].append(lq)
            # Pick a stratum with at least 2 members
            valid_strata = [s for s, members in strata_groups.items() if len(members) >= 2]
            if not valid_strata:
                continue
            chosen_stratum = random.choice(valid_strata)
            members = strata_groups[chosen_stratum]
            lq1, lq2 = random.sample(members, 2)
        else:
            # Any two logical qubits
            if num_logical >= 2:
                lq1, lq2 = random.sample(logical_qubits, 2)
            else:
                continue

        p1, p2 = mapping_dict[lq1], mapping_dict[lq2]

        # Compute delta cost
        delta = 0.0
        affected_lqs = {lq1, lq2}
        for gate_id, qubits in self.access.items():
            if len(qubits) != 2:
                continue
            qa, qb = qubits
            if qa not in affected_lqs and qb not in affected_lqs:
                continue
            old_pa, old_pb = mapping_dict[qa], mapping_dict[qb]
            new_pa = p2 if qa == lq1 else (p1 if qa == lq2 else old_pa)
            new_pb = p2 if qb == lq1 else (p1 if qb == lq2 else old_pb)
            old_d = dist[old_pa][old_pb] if isinstance(dist[old_pa], dict) else dist[old_pa][old_pb]
            new_d = dist[new_pa][new_pb] if isinstance(dist[new_pa], dict) else dist[new_pa][new_pb]
            delta += (new_d - old_d)

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            # Accept swap
            mapping_dict[lq1], mapping_dict[lq2] = p2, p1
            reverse_mapping_dict[p1] = lq2
            reverse_mapping_dict[p2] = lq1
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_mapping = mapping_dict[:]
                best_reverse = reverse_mapping_dict[:]

        # ILS perturbation every 200 iterations
        if it > 0 and it % 200 == 0 and it > stratum_unlock_iter:
            # Random perturbation: 3 random swaps
            for _ in range(3):
                rl1, rl2 = random.sample(logical_qubits, 2) if num_logical >= 2 else (0, 1)
                rp1, rp2 = mapping_dict[rl1], mapping_dict[rl2]
                mapping_dict[rl1], mapping_dict[rl2] = rp2, rp1
                reverse_mapping_dict[rp1] = rl2
                reverse_mapping_dict[rp2] = rl1
            current_cost = compute_cost(mapping_dict)

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)