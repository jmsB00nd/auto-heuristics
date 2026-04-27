def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    # Collect 2q interactions in temporal order (from access dict)
    try:
        sorted_gate_ids = sorted(self.access.keys())
    except Exception:
        sorted_gate_ids = list(self.access.keys())

    interactions = []
    depth_idx = 0
    logical_qubits = set()
    for gid in sorted_gate_ids:
        qubits = self.access[gid]
        if qubits is None:
            continue
        for q in qubits:
            if isinstance(q, int) and 0 <= q < N:
                logical_qubits.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 == q2:
                continue
            if not (0 <= q1 < N and 0 <= q2 < N):
                continue
            interactions.append((q1, q2, depth_idx))
            depth_idx += 1

    M = len(interactions)
    alpha = 3.0 / max(M, 1)

    # Temporally decayed weighted interaction graph
    interaction_weight = defaultdict(lambda: defaultdict(float))
    for q1, q2, d in interactions:
        w = math.exp(-alpha * d)
        interaction_weight[q1][q2] += w
        interaction_weight[q2][q1] += w

    total_weight = {q: sum(interaction_weight[q].values()) for q in logical_qubits}

    # Physical centrality: lower sum of distances => more central
    centrality_cache = [
        sum(self.distance_matrix[p][q] for q in range(N)) for p in range(N)
    ]
    max_centrality = max(centrality_cache) if centrality_cache else 1.0
    centrality_norm = max_centrality if max_centrality > 0 else 1.0

    used_physical = [False] * N
    placed_logical = set()

    LOOKAHEAD_DECAY = 0.35
    CENTRALITY_TIE = 1e-3

    # SINGLE-VERTEX SEED (hypothesis h4): reject edge-pair seeding to avoid
    # over-constraining the placement frontier. Pick the heaviest weighted-degree
    # logical qubit and place it on the minimum-eccentricity physical qubit.
    if logical_qubits:
        seed_logical = max(
            logical_qubits,
            key=lambda q: (total_weight.get(q, 0.0), -q),
        )
        best_phys = None
        best_score = None
        for p in range(N):
            s = centrality_cache[p]
            if best_score is None or s < best_score or (s == best_score and p < best_phys):
                best_score = s
                best_phys = p
        self.mapping_dict[seed_logical] = best_phys
        self.reverse_mapping_dict[best_phys] = seed_logical
        used_physical[best_phys] = True
        placed_logical.add(seed_logical)

    remaining = set(logical_qubits) - placed_logical

    while remaining:
        # Pick next logical qubit: strongest tie to already-placed set,
        # tiebreak by total weighted degree, then by id.
        best_cand = None
        best_cand_weight = -1.0
        best_cand_total = -1.0
        for q in remaining:
            w = 0.0
            nbrs = interaction_weight.get(q, {})
            for nbr, wt in nbrs.items():
                if nbr in placed_logical:
                    w += wt
            tw = total_weight.get(q, 0.0)
            if (w > best_cand_weight) \
               or (w == best_cand_weight and tw > best_cand_total) \
               or (w == best_cand_weight and tw == best_cand_total and (best_cand is None or q < best_cand)):
                best_cand_weight = w
                best_cand_total = tw
                best_cand = q

        if best_cand_weight <= 0.0:
            best_cand = max(remaining, key=lambda q: (total_weight.get(q, 0.0), -q))

        placed_neighbors = [
            (nbr, wt) for nbr, wt in interaction_weight.get(best_cand, {}).items()
            if nbr in placed_logical
        ]
        unplaced_neighbors = [
            (nbr, wt) for nbr, wt in interaction_weight.get(best_cand, {}).items()
            if nbr not in placed_logical and nbr != best_cand
        ]

        best_phys = None
        best_cost = None
        for p in range(N):
            if used_physical[p]:
                continue
            if placed_neighbors:
                d = 0.0
                for nbr, wt in placed_neighbors:
                    d += wt * self.distance_matrix[p][self.mapping_dict[nbr]]
                if unplaced_neighbors:
                    avg_dist_to_placed = 0.0
                    cnt = 0
                    for pl in placed_logical:
                        avg_dist_to_placed += self.distance_matrix[p][self.mapping_dict[pl]]
                        cnt += 1
                    if cnt > 0:
                        avg_dist_to_placed /= cnt
                    d += LOOKAHEAD_DECAY * sum(wt for _, wt in unplaced_neighbors) * avg_dist_to_placed
                cost = d + CENTRALITY_TIE * (centrality_cache[p] / centrality_norm)
            else:
                cost = centrality_cache[p]
            if best_cost is None or cost < best_cost or (cost == best_cost and p < best_phys):
                best_cost = cost
                best_phys = p

        if best_phys is None:
            break

        self.mapping_dict[best_cand] = best_phys
        self.reverse_mapping_dict[best_phys] = best_cand
        used_physical[best_phys] = True
        placed_logical.add(best_cand)
        remaining.discard(best_cand)

    # Identity-preferred fallback for any unmapped logical qubits
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