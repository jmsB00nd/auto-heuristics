def init_mapping(self):
    import heapq

    N = self.num_qubits
    num_physical = len(self.distance_matrix)

    logical_qubits_in_access = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_qubits_in_access.add(q)
    for q in self.qubit_interaction_graph.keys():
        logical_qubits_in_access.add(q)

    logical_to_place = [q for q in logical_qubits_in_access if 0 <= q < N]

    placed = {}
    used = set()

    mapping_list = [-1] * N
    reverse_list = [-1] * N

    if not logical_to_place:
        for i in range(N):
            mapping_list[i] = i
            reverse_list[i] = i
        self.mapping_dict = mapping_list
        self.reverse_mapping_dict = reverse_list
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    def activity_of(q):
        return self.logical_activity.get(q, 0)

    anchor_logical = max(logical_to_place, key=lambda q: (activity_of(q), -q))

    physical_candidates = list(range(num_physical))
    if not physical_candidates:
        for i in range(N):
            mapping_list[i] = i
            reverse_list[i] = i
        self.mapping_dict = mapping_list
        self.reverse_mapping_dict = reverse_list
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    def centrality_of(p):
        return self.physical_centrality.get(p, 0.0)

    anchor_physical = max(physical_candidates, key=lambda p: (centrality_of(p), -p))

    placed[anchor_logical] = anchor_physical
    used.add(anchor_physical)

    def cost_on_physical(logical_q, phys_p):
        total = 0.0
        neighbors = self.qubit_interaction_graph.get(logical_q, {})
        for partner, w in neighbors.items():
            if partner in placed:
                partner_phys = placed[partner]
                if 0 <= partner_phys < len(self.distance_matrix) and 0 <= phys_p < len(self.distance_matrix):
                    total += w * self.distance_matrix[phys_p][partner_phys]
                else:
                    total += w * 1e9
        return total

    def placed_partner_weight(logical_q):
        total = 0.0
        neighbors = self.qubit_interaction_graph.get(logical_q, {})
        for partner, w in neighbors.items():
            if partner in placed:
                total += w
        return total

    pq = []
    counter = 0
    anchor_neighbors = self.qubit_interaction_graph.get(anchor_logical, {})
    for nb, w in anchor_neighbors.items():
        if nb in placed:
            continue
        if not (0 <= nb < N):
            continue
        heapq.heappush(pq, (-w, counter, nb))
        counter += 1

    while pq:
        neg_priority, _, logical_q = heapq.heappop(pq)
        if logical_q in placed:
            continue

        current_weight = placed_partner_weight(logical_q)
        if -neg_priority < current_weight:
            heapq.heappush(pq, (-current_weight, counter, logical_q))
            counter += 1
            continue

        free_phys = [p for p in range(num_physical) if p not in used]
        if not free_phys:
            break

        best_p = None
        best_cost = None
        for p in free_phys:
            c = cost_on_physical(logical_q, p)
            tiebreak_centrality = -centrality_of(p)
            key = (c, tiebreak_centrality, p)
            if best_cost is None or key < best_cost:
                best_cost = key
                best_p = p

        placed[logical_q] = best_p
        used.add(best_p)

        for nb, w in self.qubit_interaction_graph.get(logical_q, {}).items():
            if nb in placed:
                continue
            if not (0 <= nb < N):
                continue
            new_weight = placed_partner_weight(nb) + w
            heapq.heappush(pq, (-new_weight, counter, nb))
            counter += 1

    remaining_logical = [q for q in logical_to_place if q not in placed]
    remaining_physical = [p for p in range(num_physical) if p not in used]

    for lq in remaining_logical:
        if not remaining_physical:
            break
        if lq in placed:
            continue
        if lq < num_physical and lq not in used:
            chosen = lq
            remaining_physical.remove(lq)
        else:
            chosen = remaining_physical.pop(0)
        placed[lq] = chosen
        used.add(chosen)

    all_logical = list(range(N))
    unplaced_logical = [q for q in all_logical if q not in placed]
    remaining_physical = [p for p in range(N) if p not in used]

    for lq in unplaced_logical:
        if lq < N and lq not in used:
            chosen = lq
            if lq in remaining_physical:
                remaining_physical.remove(lq)
        elif remaining_physical:
            chosen = remaining_physical.pop(0)
        else:
            continue
        placed[lq] = chosen
        used.add(chosen)

    for lq, pq_phys in placed.items():
        if 0 <= lq < N and 0 <= pq_phys < N:
            mapping_list[lq] = pq_phys

    free_phys_final = [p for p in range(N) if p not in used]
    missing_logical = [i for i in range(N) if mapping_list[i] == -1]
    for lq in missing_logical:
        if free_phys_final:
            p = free_phys_final.pop(0)
            mapping_list[lq] = p
            used.add(p)
        else:
            mapping_list[lq] = lq

    seen = set()
    duplicates = []
    for i, p in enumerate(mapping_list):
        if p in seen or p < 0 or p >= N:
            duplicates.append(i)
        else:
            seen.add(p)
    free_phys_final = [p for p in range(N) if p not in seen]
    for i in duplicates:
        if free_phys_final:
            mapping_list[i] = free_phys_final.pop(0)
            seen.add(mapping_list[i])
        else:
            mapping_list[i] = i

    for i in range(N):
        p = mapping_list[i]
        if 0 <= p < N:
            reverse_list[p] = i

    self.mapping_dict = mapping_list
    self.reverse_mapping_dict = reverse_list

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)