def init_mapping(self):
    import heapq
    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_seen = set()
    edge_weights = {}
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig:
        for u, neigh in qig.items():
            logical_seen.add(u)
            for v, w in neigh.items():
                logical_seen.add(v)
                if u == v or w <= 0:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in edge_weights:
                    edge_weights[(a, b)] = w
    if not edge_weights:
        for gate_id, qubits in self.access.items():
            for q in qubits:
                logical_seen.add(q)
            if len(qubits) == 2:
                u, v = qubits[0], qubits[1]
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1
    else:
        for gate_id, qubits in self.access.items():
            for q in qubits:
                logical_seen.add(q)

    edges_sorted = sorted(edge_weights.items(), key=lambda kv: -kv[1])

    free_phys = set(range(N))
    centrality = getattr(self, "physical_centrality", {}) or {}
    dist = self.distance_matrix

    def central_score(p):
        return centrality.get(p, 0.0)

    def assign(log_q, phys_q):
        if log_q >= N or phys_q >= N:
            return False
        if self.mapping_dict[log_q] != -1 or self.reverse_mapping_dict[phys_q] != -1:
            return False
        self.mapping_dict[log_q] = phys_q
        self.reverse_mapping_dict[phys_q] = log_q
        free_phys.discard(phys_q)
        return True

    def best_free_pair():
        best = None
        best_key = None
        for p in free_phys:
            for q in free_phys:
                if p >= q:
                    continue
                d = dist[p][q]
                key = (d, -(central_score(p) + central_score(q)))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (p, q)
        return best

    def best_free_neighbor(anchor):
        best = None
        best_key = None
        for p in free_phys:
            d = dist[anchor][p]
            key = (d, -central_score(p))
            if best_key is None or key < best_key:
                best_key = key
                best = p
        return best

    for (u, v), w in edges_sorted:
        if u >= N or v >= N:
            continue
        u_mapped = self.mapping_dict[u] != -1
        v_mapped = self.mapping_dict[v] != -1
        if u_mapped and v_mapped:
            continue
        if not u_mapped and not v_mapped:
            pair = best_free_pair()
            if pair is None:
                break
            p, q = pair
            pu, pv = (p, q) if central_score(p) >= central_score(q) else (q, p)
            assign(u, pu)
            assign(v, pv)
        elif u_mapped and not v_mapped:
            anchor = self.mapping_dict[u]
            p = best_free_neighbor(anchor)
            if p is None:
                continue
            assign(v, p)
        else:
            anchor = self.mapping_dict[v]
            p = best_free_neighbor(anchor)
            if p is None:
                continue
            assign(u, p)

    remaining_logicals = sorted(
        [l for l in logical_seen if l < N and self.mapping_dict[l] == -1],
        key=lambda l: -self.logical_activity.get(l, 0) if hasattr(self, "logical_activity") and self.logical_activity is not None else 0,
    )
    free_sorted = sorted(free_phys, key=lambda p: -central_score(p))
    for l in remaining_logicals:
        if not free_sorted:
            break
        p = free_sorted.pop(0)
        assign(l, p)
        free_phys.discard(p)

    leftover_logicals = [l for l in range(N) if self.mapping_dict[l] == -1]
    leftover_phys = sorted(free_phys)
    for l, p in zip(leftover_logicals, leftover_phys):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)