def init_mapping(self):
    N = self.num_qubits

    qig = self.qubit_interaction_graph
    activity = self.logical_activity
    centrality = self.physical_centrality
    backend = self.backend

    logical_qubits_in_circuit = set()
    for _, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits_in_circuit.add(q)

    def _activity(q):
        return activity.get(q, 0) if hasattr(activity, "get") else activity[q]

    def _centrality(p):
        return centrality.get(p, 0.0) if hasattr(centrality, "get") else 0.0

    if logical_qubits_in_circuit:
        seed_logical = max(logical_qubits_in_circuit, key=lambda q: (_activity(q), -q))
    else:
        seed_logical = 0

    logical_path = [seed_logical]
    visited_log = {seed_logical}

    current = seed_logical
    while len(logical_path) < len(logical_qubits_in_circuit):
        neighbors = qig.get(current, {}) if hasattr(qig, "get") else qig[current]
        best = None
        best_key = None
        for nb, w in neighbors.items():
            if nb in visited_log or nb not in logical_qubits_in_circuit:
                continue
            key = (w, _activity(nb), -nb)
            if best_key is None or key > best_key:
                best_key = key
                best = nb
        if best is None:
            remaining = logical_qubits_in_circuit - visited_log
            if not remaining:
                break
            best = max(remaining, key=lambda q: (_activity(q), -q))
        logical_path.append(best)
        visited_log.add(best)
        current = best

    for q in range(N):
        if q not in visited_log:
            logical_path.append(q)
            visited_log.add(q)

    all_phys = list(range(N))
    if all_phys:
        seed_phys = max(all_phys, key=lambda p: (_centrality(p), -p))
    else:
        seed_phys = 0

    phys_path = [seed_phys]
    visited_phys = {seed_phys}
    current_p = seed_phys
    while len(phys_path) < N:
        nbrs = backend.get(current_p, set()) if hasattr(backend, "get") else set()
        best_p = None
        best_pk = None
        for nb in nbrs:
            if nb in visited_phys or not (0 <= nb < N):
                continue
            key = (_centrality(nb), -nb)
            if best_pk is None or key > best_pk:
                best_pk = key
                best_p = nb
        if best_p is None:
            remaining_p = [p for p in all_phys if p not in visited_phys]
            if not remaining_p:
                break
            best_p = max(remaining_p, key=lambda p: (_centrality(p), -p))
        phys_path.append(best_p)
        visited_phys.add(best_p)
        current_p = best_p

    mapping = [0] * N
    reverse = [0] * N
    assigned_log = set()
    assigned_phys = set()

    pair_len = min(len(logical_path), len(phys_path))
    for i in range(pair_len):
        L = logical_path[i]
        P = phys_path[i]
        if L in assigned_log or P in assigned_phys:
            continue
        mapping[L] = P
        reverse[P] = L
        assigned_log.add(L)
        assigned_phys.add(P)

    remaining_logicals = [q for q in range(N) if q not in assigned_log]
    remaining_physicals = [p for p in range(N) if p not in assigned_phys]
    remaining_physicals.sort(key=lambda p: (-_centrality(p), p))
    remaining_logicals.sort(key=lambda q: (-_activity(q), q))
    for L, P in zip(remaining_logicals, remaining_physicals):
        mapping[L] = P
        reverse[P] = L
        assigned_log.add(L)
        assigned_phys.add(P)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)