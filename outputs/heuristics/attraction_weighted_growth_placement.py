def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    interaction_weight = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            if isinstance(q, int) and 0 <= q < N:
                logical_qubits.add(q)
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if not (0 <= a < N and 0 <= b < N):
                continue
            interaction_weight[a][b] += 1.0
            interaction_weight[b][a] += 1.0

    used_physical = [False] * N
    placed_logical = set()

    total_weight = {q: sum(interaction_weight[q].values()) for q in logical_qubits}

    def centrality(p):
        return sum(self.distance_matrix[p][q] for q in range(N))

    if logical_qubits:
        seed_logical = max(
            logical_qubits,
            key=lambda q: (total_weight.get(q, 0.0), -q),
        )
        best_phys = None
        best_score = None
        for p in range(N):
            s = centrality(p)
            if best_score is None or s < best_score or (s == best_score and p < best_phys):
                best_score = s
                best_phys = p
        self.mapping_dict[seed_logical] = best_phys
        self.reverse_mapping_dict[best_phys] = seed_logical
        used_physical[best_phys] = True
        placed_logical.add(seed_logical)

    remaining = set(logical_qubits) - placed_logical
    while remaining:
        best_cand = None
        best_cand_weight = -1.0
        for q in remaining:
            w = 0.0
            nbrs = interaction_weight.get(q, {})
            for nbr, wt in nbrs.items():
                if nbr in placed_logical:
                    w += wt
            if (w > best_cand_weight) or (w == best_cand_weight and (best_cand is None or q < best_cand)):
                best_cand_weight = w
                best_cand = q

        if best_cand_weight <= 0.0:
            best_cand = max(remaining, key=lambda q: (total_weight.get(q, 0.0), -q))

        neighbors_in_placed = [
            (nbr, wt) for nbr, wt in interaction_weight.get(best_cand, {}).items()
            if nbr in placed_logical
        ]

        best_phys = None
        best_dist = None
        for p in range(N):
            if used_physical[p]:
                continue
            if neighbors_in_placed:
                d = 0.0
                for nbr, wt in neighbors_in_placed:
                    d += wt * self.distance_matrix[p][self.mapping_dict[nbr]]
            else:
                d = centrality(p)
            if best_dist is None or d < best_dist or (d == best_dist and p < best_phys):
                best_dist = d
                best_phys = p

        if best_phys is None:
            break

        self.mapping_dict[best_cand] = best_phys
        self.reverse_mapping_dict[best_phys] = best_cand
        used_physical[best_phys] = True
        placed_logical.add(best_cand)
        remaining.discard(best_cand)

    for l in range(N):
        if self.mapping_dict[l] is None:
            if l < N and not used_physical[l]:
                self.mapping_dict[l] = l
                self.reverse_mapping_dict[l] = l
                used_physical[l] = True
            else:
                for p in range(N):
                    if not used_physical[p]:
                        self.mapping_dict[l] = p
                        self.reverse_mapping_dict[p] = l
                        used_physical[p] = True
                        break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)