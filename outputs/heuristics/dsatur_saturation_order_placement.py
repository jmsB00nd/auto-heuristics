def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    weight = defaultdict(float)
    neighbors = defaultdict(set)
    wdeg = defaultdict(float)
    active = set()

    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            weight[(u, v)] += 1.0
            neighbors[u].add(v)
            neighbors[v].add(u)
            wdeg[u] += 1.0
            wdeg[v] += 1.0
            active.add(u)
            active.add(v)

    def edge_w(x, y):
        if x == y:
            return 0.0
        a, b = (x, y) if x < y else (y, x)
        return weight.get((a, b), 0.0)

    placed = {}
    used_phys = set()

    def choose_center_physical():
        best_p = 0
        best_score = None
        for p in range(N):
            row = self.distance_matrix[p]
            score = (max(row), sum(row))
            if best_score is None or score < best_score:
                best_score = score
                best_p = p
        return best_p

    def adjacent_to_used(p):
        for q in used_phys:
            if (p, q) in self.backend_connections or (q, p) in self.backend_connections:
                return True
        return False

    if active:
        seed_logical = min(active, key=lambda u: (-wdeg[u], u))
        seed_phys = choose_center_physical()
        placed[seed_logical] = seed_phys
        used_phys.add(seed_phys)

        saturation = defaultdict(int)
        for nb in neighbors[seed_logical]:
            if nb not in placed:
                saturation[nb] += 1

        unplaced = set(active) - {seed_logical}

        while unplaced:
            u = min(unplaced, key=lambda x: (-saturation[x], -wdeg[x], x))

            placed_nbrs = [v for v in neighbors[u] if v in placed]

            best_phys = None
            if placed_nbrs:
                best_score = None
                for p in range(N):
                    if p in used_phys:
                        continue
                    score = 0.0
                    for v in placed_nbrs:
                        score += edge_w(u, v) * self.distance_matrix[p][placed[v]]
                    if best_score is None or score < best_score or (score == best_score and (best_phys is None or p < best_phys)):
                        best_score = score
                        best_phys = p
            else:
                candidates = [p for p in range(N) if p not in used_phys and adjacent_to_used(p)]
                if not candidates:
                    candidates = [p for p in range(N) if p not in used_phys]
                best_phys = min(candidates) if candidates else None

            if best_phys is None:
                for p in range(N):
                    if p not in used_phys:
                        best_phys = p
                        break

            placed[u] = best_phys
            used_phys.add(best_phys)
            unplaced.discard(u)

            for nb in neighbors[u]:
                if nb not in placed:
                    saturation[nb] += 1

    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N

    assigned_logical = set()
    for L, P in placed.items():
        if L < N:
            self.mapping_dict[L] = P
            self.reverse_mapping_dict[P] = L
            assigned_logical.add(L)

    remaining_logical = [L for L in range(N) if L not in assigned_logical]
    remaining_phys = [P for P in range(N) if P not in used_phys]
    for L, P in zip(remaining_logical, remaining_phys):
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)