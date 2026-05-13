def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    pair_weight = defaultdict(int)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_qubits.add(a)
                continue
            logical_qubits.add(a)
            logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            pair_weight[key] += 1

    log_activity = defaultdict(float)
    log_neighbors = defaultdict(lambda: defaultdict(float))
    for (a, b), w in pair_weight.items():
        log_activity[a] += w
        log_activity[b] += w
        log_neighbors[a][b] += w
        log_neighbors[b][a] += w

    hw_graph = nx.Graph()
    hw_graph.add_nodes_from(range(N))
    for u, neigh in self.backend.items():
        for v in neigh:
            if u != v:
                hw_graph.add_edge(u, v)

    used_physical = set()
    placed_logical = set()

    if logical_qubits:
        seed_logical = max(logical_qubits, key=lambda q: (log_activity.get(q, 0.0), -q))
        seed_physical = max(range(N), key=lambda p: (self.physical_centrality.get(p, 0.0), -p))
        self.mapping_dict[seed_logical] = seed_physical
        self.reverse_mapping_dict[seed_physical] = seed_logical
        used_physical.add(seed_physical)
        placed_logical.add(seed_logical)

        remaining_logical = set(logical_qubits) - placed_logical

        while remaining_logical:
            best_l, best_pull = None, -1.0
            for l in remaining_logical:
                pull = 0.0
                nbrs = log_neighbors[l]
                for pl in placed_logical:
                    pull += nbrs.get(pl, 0.0)
                if pull > best_pull or (pull == best_pull and (best_l is None or l < best_l)):
                    best_pull = pull
                    best_l = l
            if best_l is None:
                break
            if best_pull <= 0.0:
                best_l = max(
                    remaining_logical,
                    key=lambda q: (log_activity.get(q, 0.0), -q),
                )

            personalization = {p: 0.0 for p in range(N)}
            total_w = 0.0
            nbrs = log_neighbors[best_l]
            for pl in placed_logical:
                w = nbrs.get(pl, 0.0)
                if w > 0.0:
                    phys = self.mapping_dict[pl]
                    personalization[phys] += w
                    total_w += w
            if total_w <= 0.0:
                for pl in placed_logical:
                    personalization[self.mapping_dict[pl]] += 1.0
                    total_w += 1.0
            for k in list(personalization.keys()):
                personalization[k] /= total_w

            try:
                ppr = nx.pagerank(
                    hw_graph,
                    alpha=0.85,
                    personalization=personalization,
                    max_iter=200,
                    tol=1e-6,
                )
            except Exception:
                ppr = {p: float(self.physical_centrality.get(p, 0.0)) for p in range(N)}

            best_p, best_v = None, -float("inf")
            for p in range(N):
                if p in used_physical:
                    continue
                v = ppr.get(p, 0.0)
                if v > best_v or (v == best_v and (best_p is None or p < best_p)):
                    best_v = v
                    best_p = p
            if best_p is None:
                break

            self.mapping_dict[best_l] = best_p
            self.reverse_mapping_dict[best_p] = best_l
            used_physical.add(best_p)
            placed_logical.add(best_l)
            remaining_logical.discard(best_l)

    for l in range(N):
        if self.mapping_dict[l] == -1 and l not in used_physical:
            self.mapping_dict[l] = l
            self.reverse_mapping_dict[l] = l
            used_physical.add(l)

    unused_phys = sorted(set(range(N)) - used_physical)
    idx = 0
    for l in range(N):
        if self.mapping_dict[l] == -1:
            p = unused_phys[idx]
            idx += 1
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l
            used_physical.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)