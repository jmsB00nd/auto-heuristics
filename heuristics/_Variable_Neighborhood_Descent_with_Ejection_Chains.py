def init_mapping(self):
    import random
    from collections import defaultdict

    num_q = self.num_qubits
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    # --- Step 1: Build interaction graph from self.access ---
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0
            logical_degree[q1] += 1.0
            logical_degree[q2] += 1.0
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)
    n_logical = len(logical_qubits)
    physical_qubits = sorted(self.backend.keys())

    if n_logical == 0:
        for i in range(num_q):
            mapping_dict[i] = i
            reverse_mapping_dict[i] = i
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            from src.utils.python_to_isl import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Cost function ---
    interaction_pairs = list(interaction_weight.keys())

    def compute_cost(m_dict):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            p1, p2 = m_dict[q1], m_dict[q2]
            cost += w * self.distance_matrix[p1][p2]
        return cost

    # --- Step 3: Generate seed mapping (greedy centroid-based) ---
    # Sort logical qubits by interaction degree (descending)
    sorted_logical = sorted(logical_qubits, key=lambda q: logical_degree.get(q, 0), reverse=True)

    # Find physical qubit with best centrality (min sum of distances to all others)
    phys_centrality = {}
    for p in physical_qubits:
        phys_centrality[p] = sum(self.distance_matrix[p][p2] for p2 in physical_qubits)

    best_seeds = sorted(physical_qubits, key=lambda p: phys_centrality[p])

    # Multi-start greedy: try a few seeds, keep best
    best_cost = float('inf')
    best_mapping = None
    best_reverse = None

    num_seeds = min(5, len(best_seeds))
    for seed_idx in range(num_seeds):
        m_dict = [-1] * num_q
        r_dict = [-1] * num_q
        assigned_physical = set()

        # Place the highest-degree logical qubit at the seed physical qubit
        first_logical = sorted_logical[0]
        seed_phys = best_seeds[seed_idx]
        m_dict[first_logical] = seed_phys
        r_dict[seed_phys] = first_logical
        assigned_physical.add(seed_phys)

        # Greedily place remaining logical qubits
        placed = {first_logical}
        for lq in sorted_logical[1:]:
            # Compute gravity center: weighted average distance to already-placed neighbors
            best_p = None
            best_score = float('inf')
            for p in physical_qubits:
                if p in assigned_physical:
                    continue
                score = 0.0
                for (q1, q2), w in interaction_weight.items():
                    partner = None
                    if q1 == lq and q2 in placed:
                        partner = q2
                    elif q2 == lq and q1 in placed:
                        partner = q1
                    if partner is not None:
                        score += w * self.distance_matrix[p][m_dict[partner]]
                if score < best_score:
                    best_score = score
                    best_p = p
            m_dict[lq] = best_p
            r_dict[best_p] = lq
            assigned_physical.add(best_p)
            placed.add(lq)

        # Assign unmapped logical qubits to remaining physical qubits
        unmapped_logical = [i for i in range(num_q) if m_dict[i] == -1]
        remaining_physical = [p for p in physical_qubits if p not in assigned_physical]
        # Also include physical qubits not in backend keys
        all_physical = set(range(num_q))
        remaining_physical = sorted(all_physical - assigned_physical)

        for lq, pq in zip(unmapped_logical, remaining_physical):
            m_dict[lq] = pq
            r_dict[pq] = lq

        cost = compute_cost(m_dict)
        if cost < best_cost:
            best_cost = cost
            best_mapping = m_dict[:]
            best_reverse = r_dict[:]

    mapping_dict = best_mapping
    reverse_mapping_dict = best_reverse

    # --- Step 4: Variable Neighborhood Descent (VND) ---
    current_cost = best_cost
    max_chain_len = min(8, max(2, n_logical // 5))

    # Precompute logical qubit neighbors on hardware for efficient candidate generation
    def get_hw_neighbors(pq):
        return self.backend.get(pq, set())

    # N1: Pairwise swaps (2-opt)
    def neighborhood_swap():
        nonlocal current_cost
        improved = True
        while improved:
            improved = False
            for i in range(len(logical_qubits)):
                for j in range(i + 1, len(logical_qubits)):
                    lq1 = logical_qubits[i]
                    lq2 = logical_qubits[j]
                    p1 = mapping_dict[lq1]
                    p2 = mapping_dict[lq2]

                    # Compute delta cost
                    delta = 0.0
                    for (qa, qb), w in interaction_weight.items():
                        old_d = self.distance_matrix[mapping_dict[qa]][mapping_dict[qb]]
                        # Simulate swap
                        pa = mapping_dict[qa]
                        pb = mapping_dict[qb]
                        if qa == lq1:
                            pa = p2
                        elif qa == lq2:
                            pa = p1
                        if qb == lq1:
                            pb = p2
                        elif qb == lq2:
                            pb = p1
                        new_d = self.distance_matrix[pa][pb]
                        delta += w * (new_d - old_d)

                    if delta < -1e-9:
                        # Apply swap
                        mapping_dict[lq1], mapping_dict[lq2] = p2, p1
                        reverse_mapping_dict[p1] = lq2
                        reverse_mapping_dict[p2] = lq1
                        current_cost += delta
                        improved = True

    # N2: 3-way cyclic permutations
    def neighborhood_3cycle():
        nonlocal current_cost
        for i in range(len(logical_qubits)):
            lq1 = logical_qubits[i]
            p1 = mapping_dict[lq1]
            for p2 in get_hw_neighbors(p1):
                lq2 = reverse_mapping_dict[p2]
                if lq2 == -1 or lq2 not in logical_qubits_set:
                    continue
                for p3 in get_hw_neighbors(p2):
                    if p3 == p1:
                        continue
                    lq3 = reverse_mapping_dict[p3]
                    if lq3 == -1 or lq3 not in logical_qubits_set:
                        continue
                    # Try cyclic: lq1->p2, lq2->p3, lq3->p1
                    delta = 0.0
                    for (qa, qb), w in interaction_weight.items():
                        old_d = self.distance_matrix[mapping_dict[qa]][mapping_dict[qb]]
                        pa, pb = mapping_dict[qa], mapping_dict[qb]
                        if qa == lq1: pa = p2
                        elif qa == lq2: pa = p3
                        elif qa == lq3: pa = p1
                        if qb == lq1: pb = p2
                        elif qb == lq2: pb = p3
                        elif qb == lq3: pb = p1
                        new_d = self.distance_matrix[pa][pb]
                        delta += w * (new_d - old_d)

                    if delta < -1e-9:
                        mapping_dict[lq1] = p2
                        mapping_dict[lq2] = p3
                        mapping_dict[lq3] = p1
                        reverse_mapping_dict[p1] = lq3
                        reverse_mapping_dict[p2] = lq1
                        reverse_mapping_dict[p3] = lq2
                        current_cost += delta
                        return True
        return False

    # N3: 4-way cyclic permutations
    def neighborhood_4cycle():
        nonlocal current_cost
        for i in range(len(logical_qubits)):
            lq1 = logical_qubits[i]
            p1 = mapping_dict[lq1]
            for p2 in get_hw_neighbors(p1):
                lq2 = reverse_mapping_dict[p2]
                if lq2 == -1 or lq2 not in logical_qubits_set:
                    continue
                for p3 in get_hw_neighbors(p2):
                    if p3 == p1:
                        continue
                    lq3 = reverse_mapping_dict[p3]
                    if lq3 == -1 or lq3 not in logical_qubits_set:
                        continue
                    for p4 in get_hw_neighbors(p3):
                        if p4 == p1 or p4 == p2:
                            continue
                        lq4 = reverse_mapping_dict[p4]
                        if lq4 == -1 or lq4 not in logical_qubits_set:
                            continue
                        # Cyclic: lq1->p2, lq2->p3, lq3->p4, lq4->p1
                        delta = 0.0
                        for (qa, qb), w in interaction_weight.items():
                            old_d = self.distance_matrix[mapping_dict[qa]][mapping_dict[qb]]
                            pa, pb = mapping_dict[qa], mapping_dict[qb]
                            if qa == lq1: pa = p2
                            elif qa == lq2: pa = p3
                            elif qa == lq3: pa = p4
                            elif qa == lq4: pa = p1
                            if qb == lq1: pb = p2
                            elif qb == lq2: pb = p3
                            elif qb == lq3: pb = p4
                            elif qb == lq4: pb = p1
                            new_d = self.distance_matrix[pa][pb]
                            delta += w * (new_d - old_d)

                        if delta < -1e-9:
                            mapping_dict[lq1] = p2
                            mapping_dict[lq2] = p3
                            mapping_dict[lq3] = p4
                            mapping_dict[lq4] = p1
                            reverse_mapping_dict[p1] = lq4
                            reverse_mapping_dict[p2] = lq1
                            reverse_mapping_dict[p3] = lq2
                            reverse_mapping_dict[p4] = lq3
                            current_cost += delta
                            return True
        return False

    # N4: Ejection chains
    def neighborhood_ejection_chain():
        nonlocal current_cost

        # Compute per-qubit cost contribution
        qubit_cost = defaultdict(float)
        for (q1, q2), w in interaction_weight.items():
            d = self.distance_matrix[mapping_dict[q1]][mapping_dict[q2]]
            qubit_cost[q1] += w * d
            qubit_cost[q2] += w * d

        # Sort logical qubits by cost (worst first)
        candidates = sorted(logical_qubits, key=lambda q: qubit_cost.get(q, 0), reverse=True)

        for start_lq in candidates[:min(len(candidates), max(10, n_logical // 3))]:
            # Try to build an ejection chain starting from start_lq
            chain_logical = [start_lq]
            chain_physical = [mapping_dict[start_lq]]
            visited_logical = {start_lq}
            visited_physical = {mapping_dict[start_lq]}

            current_lq = start_lq
            for step in range(max_chain_len - 1):
                current_p = mapping_dict[current_lq]
                best_next_p = None
                best_delta = 0.0

                # Consider hardware-adjacent positions
                neighbor_positions = set()
                for p in chain_physical:
                    neighbor_positions.update(get_hw_neighbors(p))
                neighbor_positions -= visited_physical

                for next_p in neighbor_positions:
                    next_lq = reverse_mapping_dict[next_p]
                    if next_lq in visited_logical:
                        continue
                    if next_lq == -1 or next_lq not in logical_qubits_set:
                        continue

                    # Evaluate moving current_lq to next_p (ejecting next_lq)
                    delta = 0.0
                    for (qa, qb), w in interaction_weight.items():
                        if qa != current_lq and qb != current_lq:
                            continue
                        partner = qb if qa == current_lq else qa
                        p_partner = mapping_dict[partner]
                        # Check if partner is already in chain and relocated
                        for ci in range(len(chain_logical)):
                            if chain_logical[ci] == partner and ci + 1 < len(chain_physical):
                                p_partner = chain_physical[ci + 1]
                                break
                        old_d = self.distance_matrix[mapping_dict[current_lq]][p_partner]
                        new_d = self.distance_matrix[next_p][p_partner]
                        delta += w * (new_d - old_d)

                    if delta < best_delta:
                        best_delta = delta
                        best_next_p = next_p

                if best_next_p is None:
                    break

                next_lq = reverse_mapping_dict[best_next_p]
                chain_logical.append(next_lq)
                chain_physical.append(best_next_p)
                visited_logical.add(next_lq)
                visited_physical.add(best_next_p)
                current_lq = next_lq

            if len(chain_logical) < 2:
                continue

            # Now try to close the chain: place the last ejected qubit into
            # the position vacated by the first qubit in the chain
            # This is equivalent to a cyclic permutation of chain elements:
            # lq[0]->p[1], lq[1]->p[2], ..., lq[k-1]->p[0]

            # Evaluate the full cyclic permutation
            chain_len = len(chain_logical)
            # New positions: chain_logical[i] goes to chain_physical[(i+1) % chain_len]
            new_pos = {}
            for ci in range(chain_len):
                new_pos[chain_logical[ci]] = chain_physical[(ci + 1) % chain_len]

            delta = 0.0
            for (qa, qb), w in interaction_weight.items():
                old_pa = mapping_dict[qa]
                old_pb = mapping_dict[qb]
                new_pa = new_pos.get(qa, old_pa)
                new_pb = new_pos.get(qb, old_pb)
                delta += w * (self.distance_matrix[new_pa][new_pb] - self.distance_matrix[old_pa][old_pb])

            if delta < -1e-9:
                # Apply the cyclic permutation
                old_positions = [(chain_logical[ci], mapping_dict[chain_logical[ci]]) for ci in range(chain_len)]
                for ci in range(chain_len):
                    lq = chain_logical[ci]
                    new_p = chain_physical[(ci + 1) % chain_len]
                    mapping_dict[lq] = new_p
                    reverse_mapping_dict[new_p] = lq
                current_cost += delta
                return True

        return False

    # --- VND main loop ---
    max_vnd_iterations = 50
    for _ in range(max_vnd_iterations):
        # N1: pairwise swaps until no improvement
        neighborhood_swap()

        # N2: try 3-cycle
        if neighborhood_3cycle():
            continue  # Restart from N1

        # N3: try 4-cycle
        if neighborhood_4cycle():
            continue  # Restart from N1

        # N4: try ejection chain
        if neighborhood_ejection_chain():
            continue  # Restart from N1

        # No improvement found in any neighborhood
        break

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    if self.use_isl:
        from src.utils.python_to_isl import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)