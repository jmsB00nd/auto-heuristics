def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    edge_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1

    sorted_edges = sorted(edge_weight.items(), key=lambda x: -x[1])

    phys_adj = defaultdict(set)
    for (u, v) in self.backend_connections:
        phys_adj[u].add(v)
        phys_adj[v].add(u)

    L2P = {}
    P2L = {}

    phys_by_degree = sorted(range(N), key=lambda p: -len(phys_adj[p]))

    def free_phys_list():
        return [p for p in range(N) if p not in P2L]

    for (a, b), w in sorted_edges:
        a_mapped = a in L2P
        b_mapped = b in L2P

        if a_mapped and b_mapped:
            continue

        if not a_mapped and not b_mapped:
            placed = False
            for pa in phys_by_degree:
                if pa in P2L:
                    continue
                neighbors = sorted(phys_adj[pa], key=lambda p: -len(phys_adj[p]))
                for pb in neighbors:
                    if pb not in P2L:
                        L2P[a] = pa; P2L[pa] = a
                        L2P[b] = pb; P2L[pb] = b
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                free = free_phys_list()
                if len(free) >= 2:
                    L2P[a] = free[0]; P2L[free[0]] = a
                    L2P[b] = free[1]; P2L[free[1]] = b
                elif len(free) == 1:
                    L2P[a] = free[0]; P2L[free[0]] = a
            continue

        mapped_q, unmapped_q = (a, b) if a_mapped else (b, a)
        pm = L2P[mapped_q]

        placed = False
        neighbors = sorted(phys_adj[pm], key=lambda p: -len(phys_adj[p]))
        for pn in neighbors:
            if pn not in P2L:
                L2P[unmapped_q] = pn
                P2L[pn] = unmapped_q
                placed = True
                break

        if not placed:
            free = free_phys_list()
            if free:
                best = min(free, key=lambda p: self.distance_matrix[pm][p])
                L2P[unmapped_q] = best
                P2L[best] = unmapped_q

    unmapped_logicals = [L for L in range(N) if L not in L2P]
    free_phys = free_phys_list()

    for L, P in zip(unmapped_logicals, free_phys):
        L2P[L] = P
        P2L[P] = L

    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N
    for L in range(N):
        self.mapping_dict[L] = L2P[L]
    for L in range(N):
        self.reverse_mapping_dict[self.mapping_dict[L]] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)