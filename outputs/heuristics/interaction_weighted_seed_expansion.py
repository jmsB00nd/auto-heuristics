def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    W = defaultdict(int)
    deg = defaultdict(int)
    L_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                L_set.add(a)
                continue
            key = (a, b) if a < b else (b, a)
            W[key] += 1
            deg[a] += 1
            deg[b] += 1
            L_set.add(a)
            L_set.add(b)
        elif len(qubits) == 1:
            L_set.add(qubits[0])

    def w_pair(x, y):
        if x == y:
            return 0
        k = (x, y) if x < y else (y, x)
        return W.get(k, 0)

    if not L_set:
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    dm = self.distance_matrix
    eccentricity = [max(dm[p]) if N > 1 else 0 for p in range(N)]
    closeness = [sum(dm[p]) for p in range(N)]

    best_edge = None
    best_score = None
    for (u, v) in self.backend_connections:
        if u >= v:
            continue
        ecc = max(eccentricity[u], eccentricity[v])
        clo = closeness[u] + closeness[v]
        score = (ecc, clo)
        if best_score is None or score < best_score:
            best_score = score
            best_edge = (u, v)

    if best_edge is None:
        center = min(range(N), key=lambda p: (eccentricity[p], closeness[p]))
        best_edge = (center, center)

    placed_logical = set()
    used_physical = set()

    if W:
        l1, l2 = max(W.keys(), key=lambda k: (W[k], deg[k[0]] + deg[k[1]]))
        p1, p2 = best_edge
        if p1 == p2:
            self.mapping_dict[l1] = p1
            self.reverse_mapping_dict[p1] = l1
            placed_logical.add(l1)
            used_physical.add(p1)
        else:
            self.mapping_dict[l1] = p1
            self.reverse_mapping_dict[p1] = l1
            self.mapping_dict[l2] = p2
            self.reverse_mapping_dict[p2] = l2
            placed_logical.update([l1, l2])
            used_physical.update([p1, p2])
    else:
        seed_l = next(iter(L_set))
        seed_p = min(range(N), key=lambda p: (eccentricity[p], closeness[p]))
        self.mapping_dict[seed_l] = seed_p
        self.reverse_mapping_dict[seed_p] = seed_l
        placed_logical.add(seed_l)
        used_physical.add(seed_p)

    remaining_logical = L_set - placed_logical

    while remaining_logical:
        best_l = None
        best_l_score = -1
        for l in remaining_logical:
            s = sum(w_pair(l, p) for p in placed_logical)
            if s > best_l_score or (s == best_l_score and best_l is not None and deg[l] > deg[best_l]):
                best_l_score = s
                best_l = l
        if best_l is None:
            break

        if best_l_score == 0:
            best_l = max(remaining_logical, key=lambda l: (deg[l], -l))

        frontier = set()
        for up in used_physical:
            for nb in self.backend[up]:
                if nb not in used_physical:
                    frontier.add(nb)
        free_candidates = frontier if frontier else set(range(N)) - used_physical
        if not free_candidates:
            break

        def phys_score(phys):
            total = 0.0
            for pl in placed_logical:
                w = w_pair(best_l, pl)
                if w == 0:
                    continue
                d = dm[phys][self.mapping_dict[pl]]
                if d <= 0:
                    d = 1
                total += w / d
            return total

        best_p = max(free_candidates, key=lambda p: (phys_score(p), -closeness[p]))

        self.mapping_dict[best_l] = best_p
        self.reverse_mapping_dict[best_p] = best_l
        placed_logical.add(best_l)
        used_physical.add(best_p)
        remaining_logical.remove(best_l)

    free_physical = [p for p in range(N) if p not in used_physical]
    fp_idx = 0
    for l in range(N):
        if self.mapping_dict[l] == -1:
            if l not in used_physical:
                self.mapping_dict[l] = l
                self.reverse_mapping_dict[l] = l
                used_physical.add(l)
                if l in free_physical:
                    free_physical.remove(l)
            else:
                while fp_idx < len(free_physical) and free_physical[fp_idx] in used_physical:
                    fp_idx += 1
                if fp_idx < len(free_physical):
                    p = free_physical[fp_idx]
                    fp_idx += 1
                    self.mapping_dict[l] = p
                    self.reverse_mapping_dict[p] = l
                    used_physical.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)