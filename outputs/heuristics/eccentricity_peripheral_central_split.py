def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    activity = defaultdict(int)
    logical_set = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_set.add(q)
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            activity[a] += 1
            activity[b] += 1

    logicals_sorted = sorted(
        logical_set,
        key=lambda L: (-activity[L], L),
    )

    ecc = [0] * N
    for p in range(N):
        row = self.distance_matrix[p]
        m = 0
        for q in range(N):
            d = row[q]
            if d > m:
                m = d
        ecc[p] = m

    physicals_sorted = sorted(range(N), key=lambda p: (ecc[p], p))

    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_phys = set()
    used_log = set()

    for L, P in zip(logicals_sorted, physicals_sorted):
        if L >= N:
            continue
        mapping_dict[L] = P
        reverse_mapping_dict[P] = L
        used_phys.add(P)
        used_log.add(L)

    remaining_phys = [p for p in range(N) if p not in used_phys]
    rp_idx = 0
    for L in range(N):
        if L in used_log:
            continue
        if rp_idx >= len(remaining_phys):
            break
        P = remaining_phys[rp_idx]
        rp_idx += 1
        mapping_dict[L] = P
        reverse_mapping_dict[P] = L

    if -1 in mapping_dict:
        free_phys = [p for p in range(N) if p not in set(x for x in mapping_dict if x != -1)]
        fi = 0
        for L in range(N):
            if mapping_dict[L] == -1:
                mapping_dict[L] = free_phys[fi]
                reverse_mapping_dict[free_phys[fi]] = L
                fi += 1

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)