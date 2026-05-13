def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    edge_weight = defaultdict(int)
    L_used = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                L_used.add(a)
                continue
            u, v = (a, b) if a < b else (b, a)
            edge_weight[(u, v)] += 1
            L_used.add(a)
            L_used.add(b)
        elif len(qubits) == 1:
            L_used.add(qubits[0])

    sorted_logical_edges = sorted(edge_weight.items(), key=lambda kv: -kv[1])

    centrality = [0] * N
    for p in range(N):
        s = 0
        row = self.distance_matrix[p]
        for q in range(N):
            d = row[q]
            if d > 0:
                s += d
        centrality[p] = s

    phys_edges = set()
    for (a, b) in self.backend_connections:
        if a == b:
            continue
        e = (a, b) if a < b else (b, a)
        phys_edges.add(e)
    sorted_phys_edges = sorted(
        phys_edges,
        key=lambda e: (centrality[e[0]] + centrality[e[1]], e[0], e[1])
    )

    assigned = {}
    used_phys = set()

    def free_neighbor(p):
        best = None
        best_cent = None
        for nb in self.backend[p]:
            if nb in used_phys:
                continue
            c = centrality[nb]
            if best is None or c < best_cent:
                best = nb
                best_cent = c
        return best

    def closest_free(p):
        best = None
        best_d = None
        row = self.distance_matrix[p]
        for q in range(N):
            if q in used_phys or q == p:
                continue
            d = row[q]
            if d <= 0:
                continue
            if best is None or d < best_d or (d == best_d and centrality[q] < centrality[best]):
                best = q
                best_d = d
        if best is None:
            for q in range(N):
                if q not in used_phys:
                    return q
        return best

    def take_seed_edge():
        for e in sorted_phys_edges:
            if e[0] not in used_phys and e[1] not in used_phys:
                return e
        return None

    def assign(L, P):
        assigned[L] = P
        used_phys.add(P)

    for (u, v), _w in sorted_logical_edges:
        u_in = u in assigned
        v_in = v in assigned
        if u_in and v_in:
            continue
        if not u_in and not v_in:
            seed = take_seed_edge()
            if seed is None:
                p1 = closest_free(0 if 0 not in used_phys else next(iter(range(N))))
                if p1 is None:
                    break
                assign(u, p1)
                p2 = closest_free(p1)
                if p2 is None:
                    break
                assign(v, p2)
            else:
                p1, p2 = seed
                if centrality[p1] <= centrality[p2]:
                    assign(u, p1)
                    assign(v, p2)
                else:
                    assign(u, p2)
                    assign(v, p1)
        else:
            mapped_L, unmapped_L = (u, v) if u_in else (v, u)
            anchor = assigned[mapped_L]
            target = free_neighbor(anchor)
            if target is None:
                target = closest_free(anchor)
            if target is None:
                break
            assign(unmapped_L, target)

    for L in L_used:
        if L in assigned:
            continue
        if assigned:
            anchors = list(assigned.values())
            best = None
            best_d = None
            for q in range(N):
                if q in used_phys:
                    continue
                d_min = min(self.distance_matrix[a][q] for a in anchors)
                if best is None or d_min < best_d or (d_min == best_d and centrality[q] < centrality[best]):
                    best = q
                    best_d = d_min
            if best is None:
                for q in range(N):
                    if q not in used_phys:
                        best = q
                        break
            if best is None:
                break
            assign(L, best)
        else:
            for q in range(N):
                if q not in used_phys:
                    assign(L, q)
                    break

    mapping_dict = [-1] * N
    for L, P in assigned.items():
        if 0 <= L < N:
            mapping_dict[L] = P

    free_phys = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for L in range(N):
        if mapping_dict[L] == -1:
            while fp_idx < len(free_phys) and free_phys[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx < len(free_phys):
                p = free_phys[fp_idx]
                fp_idx += 1
                mapping_dict[L] = p
                used_phys.add(p)
            else:
                for q in range(N):
                    if q not in used_phys:
                        mapping_dict[L] = q
                        used_phys.add(q)
                        break

    if len(set(mapping_dict)) != N or -1 in mapping_dict:
        mapping_dict = list(range(N))

    reverse_mapping_dict = [0] * N
    for L in range(N):
        reverse_mapping_dict[mapping_dict[L]] = L

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)