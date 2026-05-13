def init_mapping(self):
    N = self.num_qubits
    D = self.distance_matrix

    neighbors = [dict() for _ in range(N)]
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if 0 <= a < N and 0 <= b < N and a != b:
                neighbors[a][b] = neighbors[a].get(b, 0) + 1
                neighbors[b][a] = neighbors[b].get(a, 0) + 1

    mapping = list(range(N))
    reverse = list(range(N))

    try:
        from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
        md, rd = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, N
        )
        md_l = list(md) if md is not None else None
        if md_l is not None and len(md_l) == N and len(set(md_l)) == N and all(
            0 <= p < N for p in md_l
        ):
            mapping = md_l
            reverse = [0] * N
            for L, P in enumerate(mapping):
                reverse[P] = L
    except Exception:
        pass

    def payoff_at(L, P, mp):
        s = 0.0
        for nbr, w in neighbors[L].items():
            s -= w * D[P][mp[nbr]]
        return s

    import random
    rng = random.Random(0xC0FFEE)
    order = list(range(N))
    rng.shuffle(order)

    max_iters = max(20, 4 * N)
    for _ in range(max_iters):
        improved = False
        for L in order:
            cur_P = mapping[L]
            cur_pay = payoff_at(L, cur_P, mapping)
            best_P = cur_P
            best_pay = cur_pay
            nbr_L = neighbors[L]
            for P in range(N):
                if P == cur_P:
                    continue
                other_L = reverse[P]
                s = 0.0
                for nbr, w in nbr_L.items():
                    if nbr == other_L:
                        s -= w * D[P][cur_P]
                    else:
                        s -= w * D[P][mapping[nbr]]
                if s > best_pay + 1e-12:
                    best_pay = s
                    best_P = P
            if best_P != cur_P:
                other_L = reverse[best_P]
                mapping[L], mapping[other_L] = best_P, cur_P
                reverse[cur_P], reverse[best_P] = other_L, L
                improved = True
        if not improved:
            break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)