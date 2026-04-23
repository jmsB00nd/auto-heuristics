def init_mapping(self):
    import heapq

    num_q = self.num_qubits
    qig = self.qubit_interaction_graph

    # Collect all logical qubits involved in 2-qubit gates
    logical_qubits = set()
    for q1 in qig:
        logical_qubits.add(q1)
        for q2 in qig[q1]:
            logical_qubits.add(q2)

    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    if not logical_qubits:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Step 1: Find the heaviest interaction edge as seed
    best_weight = -1
    seed_a, seed_b = None, None
    seen = set()
    for q1 in qig:
        for q2, w in qig[q1].items():
            if (q2, q1) not in seen:
                seen.add((q1, q2))
                if w > best_weight:
                    best_weight = w
                    seed_a, seed_b = q1, q2

    if seed_a is None:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Step 2: Pick the best physical edge for the seed pair
    dist = self.distance_matrix
    num_physical = len(dist)
    best_edge_score = float('inf')
    best_pa, best_pb = 0, 1
    for (pa, pb) in self.backend_connections:
        if pa >= num_physical or pb >= num_physical:
            continue
        score = sum(dist[pa]) + sum(dist[pb])
        if score < best_edge_score:
            best_edge_score = score
            best_pa, best_pb = pa, pb

    placed = set()
    used = set()

    self.mapping_dict[seed_a] = best_pa
    self.mapping_dict[seed_b] = best_pb
    self.reverse_mapping_dict[best_pa] = seed_a
    self.reverse_mapping_dict[best_pb] = seed_b
    placed.add(seed_a)
    placed.add(seed_b)
    used.add(best_pa)
    used.add(best_pb)

    # Step 3: Grow outward
    # Build candidate frontier: unplaced logical qubits interacting with placed ones
    def total_weight_to_placed(q):
        total = 0
        for p in placed:
            total += qig[q].get(p, 0)
        return total

    while len(placed) < len(logical_qubits):
        # Find the unplaced logical qubit with max total interaction weight to placed qubits
        best_q = None
        best_q_weight = -1
        for q in logical_qubits:
            if q in placed:
                continue
            tw = total_weight_to_placed(q)
            if tw > best_q_weight:
                best_q_weight = tw
                best_q = q

        if best_q is None:
            break

        placed.add(best_q)

        # Find heaviest placed partner of best_q
        heaviest_partner = None
        heaviest_w = -1
        for partner in placed:
            if partner == best_q:
                continue
            w = qig[best_q].get(partner, 0)
            if w > heaviest_w:
                heaviest_w = w
                heaviest_partner = partner

        # Physical location of heaviest partner
        partner_phys = self.mapping_dict[heaviest_partner]

        # Evaluate physical neighbors of partner_phys
        best_phys = None
        best_phys_score = float('inf')
        if partner_phys in self.backend:
            for candidate_phys in self.backend[partner_phys]:
                if candidate_phys in used or candidate_phys >= num_physical:
                    continue
                score = 0.0
                for placed_q in placed:
                    if placed_q == best_q:
                        continue
                    w = qig[best_q].get(placed_q, 0)
                    if w > 0:
                        placed_phys = self.mapping_dict[placed_q]
                        score += w * dist[candidate_phys][placed_phys]
                if score < best_phys_score:
                    best_phys_score = score
                    best_phys = candidate_phys

        # If no adjacent physical qubit is free, pick the closest unused one globally
        if best_phys is None:
            best_phys_score = float('inf')
            for p in range(num_physical):
                if p in used:
                    continue
                score = 0.0
                for placed_q in placed:
                    if placed_q == best_q:
                        continue
                    w = qig[best_q].get(placed_q, 0)
                    if w > 0:
                        placed_phys = self.mapping_dict[placed_q]
                        score += w * dist[p][placed_phys]
                if score < best_phys_score:
                    best_phys_score = score
                    best_phys = p

        self.mapping_dict[best_q] = best_phys
        self.reverse_mapping_dict[best_phys] = best_q
        used.add(best_phys)

    # Step 4: Fallback — assign remaining logical qubits to remaining physical qubits
    remaining_physical = [p for p in range(num_physical) if p not in used]
    remaining_logical = [q for q in range(num_q) if q not in placed]
    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)