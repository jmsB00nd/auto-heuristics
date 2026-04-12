def init_mapping(self):
    import random
    import math
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # -----------------------------------------------------------------
    # Step 1: Build DAG for topological ordering
    # -----------------------------------------------------------------
    schedule = sorted(self.access.keys())

    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
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

    # -----------------------------------------------------------------
    # Step 2: Identify 2-qubit gates and build interaction graph
    # -----------------------------------------------------------------
    two_qubit_gates = {}
    interaction_weight = defaultdict(float)
    logical_qubits_set = set()
    logical_neighbors = defaultdict(dict)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            two_qubit_gates[gate] = (qubits[0], qubits[1])
            logical_qubits_set.add(qubits[0])
            logical_qubits_set.add(qubits[1])
            key = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
            interaction_weight[key] += 1.0
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    all_gates = set(self.access.keys())

    # -----------------------------------------------------------------
    # Step 3: Compute gate degree (number of 2q neighbors in interaction graph)
    # -----------------------------------------------------------------
    gate_logical_degree = {}
    for g in two_qubit_gates:
        l1, l2 = two_qubit_gates[g]
        gate_logical_degree[g] = len(logical_neighbors.get(l1, {})) + len(logical_neighbors.get(l2, {}))

    # Dependency count per gate (number of successors transitively)
    dep_count = defaultdict(int)
    # Compute via reverse topological BFS
    in_degree_copy = {g: len(successors.get(g, set())) for g in all_gates}
    rev_topo = []
    q_rev = deque(g for g in all_gates if in_degree_copy[g] == 0)
    while q_rev:
        g = q_rev.popleft()
        rev_topo.append(g)
        for p in predecessors.get(g, set()):
            in_degree_copy[p] -= 1
            if in_degree_copy[p] == 0:
                q_rev.append(p)

    # dep_count[g] = number of 2q gates reachable from g
    for g in rev_topo:
        dep_count[g] = 0
        for s in successors.get(g, set()):
            dep_count[g] += dep_count[s] + (1 if s in two_qubit_gates else 0)

    # -----------------------------------------------------------------
    # Step 4: Generate 5 topological orderings with different tie-breaks
    # -----------------------------------------------------------------
    def topological_sort_with_tiebreak(tiebreak_fn):
        in_deg = {g: len(predecessors.get(g, set())) for g in all_gates}
        ready = [g for g in all_gates if in_deg[g] == 0]
        ready.sort(key=tiebreak_fn)
        order = []
        while ready:
            g = ready.pop(0)
            order.append(g)
            for s in successors.get(g, set()):
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    ready.append(s)
                    ready.sort(key=tiebreak_fn)
        return order

    # Centrality of logical qubits involved in a gate
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    def gate_centrality(g):
        if g in two_qubit_gates:
            l1, l2 = two_qubit_gates[g]
            return -(len(logical_neighbors.get(l1, {})) + len(logical_neighbors.get(l2, {})))
        return 0

    tiebreaks = [
        lambda g: g,                                                    # default (gate order)
        lambda g: -gate_logical_degree.get(g, 0),                       # highest-degree-first
        lambda g: -dep_count.get(g, 0),                                 # most-dependencies-first
        lambda g: -g,                                                   # reverse order
        lambda g: gate_centrality(g),                                   # centrality-based
    ]

    topo_orders = [topological_sort_with_tiebreak(tb) for tb in tiebreaks]

    # -----------------------------------------------------------------
    # Step 5: Progressive freezing placement
    # -----------------------------------------------------------------
    def progressive_placement(topo_order):
        placed_logical = set()
        used_physical = set()
        mapping = [-1] * num_q
        reverse_mapping = [-1] * num_q

        def place(lq, pq):
            mapping[lq] = pq
            reverse_mapping[pq] = lq
            placed_logical.add(lq)
            used_physical.add(pq)

        def get_free_physical():
            return [pq for pq in physical_qubits if pq not in used_physical]

        def best_adjacent_pair():
            """Find best adjacent physical pair for two unplaced qubits."""
            best_pair = None
            best_score = float('inf')
            for pq1 in physical_qubits:
                if pq1 in used_physical:
                    continue
                for pq2 in self.backend.get(pq1, set()):
                    if pq2 in used_physical or pq2 <= pq1:
                        continue
                    # Score: sum of distances to other used physical qubits
                    # (lower = more central among already placed)
                    score = 0
                    if used_physical:
                        for up in used_physical:
                            score += self.distance_matrix[pq1][up] + self.distance_matrix[pq2][up]
                    else:
                        # If nothing placed yet, prefer central pair
                        score = phys_centrality.get(pq1, 0) + phys_centrality.get(pq2, 0)
                    if score < best_score:
                        best_score = score
                        best_pair = (pq1, pq2)
            return best_pair

        def best_free_for(lq, anchor_pq):
            """Find best free physical qubit near anchor, with connectivity tie-break."""
            free = get_free_physical()
            if not free:
                return None
            # Score: distance to anchor + future cost estimate
            unplaced_neighbors = [oq for oq in logical_neighbors.get(lq, {}) if oq not in placed_logical]
            best_pq = None
            best_score = float('inf')
            for pq in free:
                score = self.distance_matrix[pq][anchor_pq]
                # Tie-break: prefer physical qubits with more free neighbors
                # (for future placement flexibility)
                free_adj = sum(1 for n in self.backend.get(pq, set()) if n not in used_physical)
                score -= 0.01 * free_adj
                # Future cost: for unplaced logical neighbors, estimate distance
                for oq in unplaced_neighbors:
                    w = logical_neighbors[lq].get(oq, 1.0)
                    # Estimate: nearest free neighbor of pq
                    min_d = min((self.distance_matrix[pq][fpq] for fpq in free if fpq != pq), default=1)
                    score += 0.1 * w * min_d
                if score < best_score:
                    best_score = score
                    best_pq = pq
            return best_pq

        def best_pair_for_two(l1, l2):
            """Find best adjacent physical pair specifically for (l1, l2)."""
            best_pair = None
            best_score = float('inf')
            for pq1 in physical_qubits:
                if pq1 in used_physical:
                    continue
                for pq2 in self.backend.get(pq1, set()):
                    if pq2 in used_physical:
                        continue
                    # Base: already adjacent (distance 1), so gate cost = 0
                    score = 0.0
                    # Future cost for l1 placed at pq1
                    for oq, w in logical_neighbors.get(l1, {}).items():
                        if oq == l2:
                            continue
                        if oq in placed_logical:
                            score += w * self.distance_matrix[pq1][mapping[oq]]
                        else:
                            free_temp = [fp for fp in physical_qubits if fp not in used_physical and fp != pq1 and fp != pq2]
                            if free_temp:
                                score += w * min(self.distance_matrix[pq1][fp] for fp in free_temp)
                    # Future cost for l2 placed at pq2
                    for oq, w in logical_neighbors.get(l2, {}).items():
                        if oq == l1:
                            continue
                        if oq in placed_logical:
                            score += w * self.distance_matrix[pq2][mapping[oq]]
                        else:
                            free_temp = [fp for fp in physical_qubits if fp not in used_physical and fp != pq1 and fp != pq2]
                            if free_temp:
                                score += w * min(self.distance_matrix[pq2][fp] for fp in free_temp)
                    if score < best_score:
                        best_score = score
                        best_pair = (pq1, pq2)
            return best_pair

        # Process gates in topological order
        for gate in topo_order:
            if gate not in two_qubit_gates:
                # 1-qubit gate: ensure qubit is placed somewhere
                lq = self.access[gate][0]
                if lq not in placed_logical:
                    free = get_free_physical()
                    if free:
                        # Place at most central free qubit
                        best_pq = min(free, key=lambda pq: phys_centrality.get(pq, 0))
                        place(lq, best_pq)
                continue

            l1, l2 = two_qubit_gates[gate]
            l1_placed = l1 in placed_logical
            l2_placed = l2 in placed_logical

            if l1_placed and l2_placed:
                continue
            elif not l1_placed and not l2_placed:
                pair = best_pair_for_two(l1, l2)
                if pair:
                    place(l1, pair[0])
                    place(l2, pair[1])
                else:
                    # Fallback: place at best available
                    free = get_free_physical()
                    if len(free) >= 2:
                        place(l1, free[0])
                        place(l2, free[1])
                    elif len(free) == 1:
                        place(l1, free[0])
            elif l1_placed and not l2_placed:
                anchor = mapping[l1]
                pq = best_free_for(l2, anchor)
                if pq is not None:
                    place(l2, pq)
            elif l2_placed and not l1_placed:
                anchor = mapping[l2]
                pq = best_free_for(l1, anchor)
                if pq is not None:
                    place(l1, pq)

        # Place any remaining unplaced logical qubits
        unmapped = [q for q in range(num_q) if mapping[q] == -1]
        free = [q for q in range(num_q) if reverse_mapping[q] == -1]
        for lq, pq in zip(unmapped, free):
            mapping[lq] = pq
            reverse_mapping[pq] = lq

        return mapping, reverse_mapping

    # -----------------------------------------------------------------
    # Step 6: Generate 5 candidate solutions
    # -----------------------------------------------------------------
    candidates = []
    for topo_order in topo_orders:
        m, rm = progressive_placement(topo_order)
        candidates.append((m, rm))

    # -----------------------------------------------------------------
    # Step 7: Evaluate candidates using critical-path-weighted cost
    # -----------------------------------------------------------------
    def evaluate_cost(mapping):
        cost = 0.0
        for gate, (l1, l2) in two_qubit_gates.items():
            p1, p2 = mapping[l1], mapping[l2]
            weight = dep_count.get(gate, 0) + 1
            cost += weight * self.distance_matrix[p1][p2]
        return cost

    best_mapping = None
    best_reverse = None
    best_cost = float('inf')

    for m, rm in candidates:
        c = evaluate_cost(m)
        if c < best_cost:
            best_cost = c
            best_mapping = m[:]
            best_reverse = rm[:]

    # -----------------------------------------------------------------
    # Step 8: ILS + SA refinement (unfreeze all qubits)
    # -----------------------------------------------------------------
    interacting_logical = sorted(logical_qubits_set)

    if len(interacting_logical) > 1:
        current_mapping = best_mapping[:]
        current_reverse = best_reverse[:]
        current_cost = best_cost

        T = current_cost / max(len(two_qubit_gates), 1) * 0.5
        T_min = T * 0.01
        cooling = 0.97
        max_iters = min(len(interacting_logical) * len(interacting_logical), 5000)

        for iteration in range(max_iters):
            # Pick two random logical qubits to swap
            i_idx = random.randint(0, len(interacting_logical) - 1)
            j_idx = random.randint(0, len(interacting_logical) - 2)
            if j_idx >= i_idx:
                j_idx += 1

            lq_a = interacting_logical[i_idx]
            lq_b = interacting_logical[j_idx]
            pq_a = current_mapping[lq_a]
            pq_b = current_mapping[lq_b]

            if pq_a == pq_b:
                continue

            # Compute delta cost
            delta = 0.0
            affected = set()
            affected.update(logical_neighbors.get(lq_a, {}).keys())
            affected.update(logical_neighbors.get(lq_b, {}).keys())

            for q in affected:
                if q == lq_a or q == lq_b:
                    continue
                pq_q = current_mapping[q]

                w_a = logical_neighbors.get(lq_a, {}).get(q, 0.0)
                if w_a > 0:
                    # Weighted by dep_count of gates between lq_a and q
                    delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])

                w_b = logical_neighbors.get(lq_b, {}).get(q, 0.0)
                if w_b > 0:
                    delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

            # Direct interaction between lq_a and lq_b doesn't change with swap

            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / max(T, 1e-10))):
                current_mapping[lq_a] = pq_b
                current_mapping[lq_b] = pq_a
                current_reverse[pq_a] = lq_b
                current_reverse[pq_b] = lq_a
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mapping = current_mapping[:]
                    best_reverse = current_reverse[:]

            T *= cooling

        # ILS: Perturbation + restart (3 restarts)
        for restart in range(3):
            # Perturb best solution: randomly swap 3-5 pairs
            perturbed_mapping = best_mapping[:]
            perturbed_reverse = best_reverse[:]
            n_perturb = random.randint(3, min(5, len(interacting_logical) // 2 + 1))

            for _ in range(n_perturb):
                i_idx = random.randint(0, len(interacting_logical) - 1)
                j_idx = random.randint(0, len(interacting_logical) - 2)
                if j_idx >= i_idx:
                    j_idx += 1
                la = interacting_logical[i_idx]
                lb = interacting_logical[j_idx]
                pa, pb = perturbed_mapping[la], perturbed_mapping[lb]
                perturbed_mapping[la] = pb
                perturbed_mapping[lb] = pa
                perturbed_reverse[pa] = lb
                perturbed_reverse[pb] = la

            current_mapping = perturbed_mapping
            current_reverse = perturbed_reverse
            current_cost = evaluate_cost(current_mapping)

            T = current_cost / max(len(two_qubit_gates), 1) * 0.3
            cooling = 0.97

            for iteration in range(max_iters // 2):
                i_idx = random.randint(0, len(interacting_logical) - 1)
                j_idx = random.randint(0, len(interacting_logical) - 2)
                if j_idx >= i_idx:
                    j_idx += 1

                lq_a = interacting_logical[i_idx]
                lq_b = interacting_logical[j_idx]
                pq_a = current_mapping[lq_a]
                pq_b = current_mapping[lq_b]

                if pq_a == pq_b:
                    continue

                delta = 0.0
                affected = set()
                affected.update(logical_neighbors.get(lq_a, {}).keys())
                affected.update(logical_neighbors.get(lq_b, {}).keys())

                for q in affected:
                    if q == lq_a or q == lq_b:
                        continue
                    pq_q = current_mapping[q]

                    w_a = logical_neighbors.get(lq_a, {}).get(q, 0.0)
                    if w_a > 0:
                        delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])

                    w_b = logical_neighbors.get(lq_b, {}).get(q, 0.0)
                    if w_b > 0:
                        delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                if delta < 0 or (T > T_min and random.random() < math.exp(-delta / max(T, 1e-10))):
                    current_mapping[lq_a] = pq_b
                    current_mapping[lq_b] = pq_a
                    current_reverse[pq_a] = lq_b
                    current_reverse[pq_b] = lq_a
                    current_cost += delta

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_mapping = current_mapping[:]
                        best_reverse = current_reverse[:]

                T *= cooling

    # -----------------------------------------------------------------
    # Final assignment
    # -----------------------------------------------------------------
    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)