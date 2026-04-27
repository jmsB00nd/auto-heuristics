def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_weights = defaultdict(int)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if a >= N or b >= N:
                continue
            edge = (min(a, b), max(a, b))
            logical_weights[edge] += 1
            logical_qubits.add(a)
            logical_qubits.add(b)

    if not logical_weights:
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    G_log = nx.Graph()
    for (a, b), w in logical_weights.items():
        G_log.add_edge(a, b, weight=w)
    try:
        log_matching = nx.max_weight_matching(G_log, maxcardinality=False)
    except Exception:
        log_matching = set()

    log_matched_edges = []
    for (u, v) in log_matching:
        w = logical_weights[(min(u, v), max(u, v))]
        log_matched_edges.append((w, u, v))
    log_matched_edges.sort(reverse=True)

    G_phys = nx.Graph()
    for i in range(N):
        G_phys.add_node(i)
    seen_phys_edges = set()
    for (p, q) in self.backend_connections:
        if p == q:
            continue
        e = (min(p, q), max(p, q))
        if e in seen_phys_edges:
            continue
        seen_phys_edges.add(e)

    total_dist = [0.0] * N
    for i in range(N):
        s = 0.0
        row = self.distance_matrix[i]
        for j in range(N):
            d = row[j]
            if d > 0:
                s += d
        total_dist[i] = s if s > 0 else 1.0
    max_total = max(total_dist) if total_dist else 1.0

    edge_centrality = {}
    for (a, b) in seen_phys_edges:
        c = (max_total - total_dist[a]) + (max_total - total_dist[b]) + 1e-6
        edge_centrality[(a, b)] = c
        G_phys.add_edge(a, b, weight=c)

    try:
        phys_matching = nx.max_weight_matching(G_phys, maxcardinality=True)
    except Exception:
        phys_matching = set()

    phys_matched_edges = []
    for (u, v) in phys_matching:
        e = (min(u, v), max(u, v))
        c = edge_centrality.get(e, 0.0)
        phys_matched_edges.append((c, u, v))
    phys_matched_edges.sort(reverse=True)

    used_phys = set()
    placed_log = set()

    pair_count = min(len(log_matched_edges), len(phys_matched_edges))
    for k in range(pair_count):
        _, lu, lv = log_matched_edges[k]
        _, pu, pv = phys_matched_edges[k]
        if pu in used_phys or pv in used_phys:
            continue
        if lu in placed_log or lv in placed_log:
            continue
        deg_lu = G_log.degree(lu, weight='weight') if G_log.has_node(lu) else 0
        deg_lv = G_log.degree(lv, weight='weight') if G_log.has_node(lv) else 0
        deg_pu = G_phys.degree(pu) if G_phys.has_node(pu) else 0
        deg_pv = G_phys.degree(pv) if G_phys.has_node(pv) else 0
        if (deg_lu >= deg_lv) == (deg_pu >= deg_pv):
            self.mapping_dict[lu] = pu
            self.mapping_dict[lv] = pv
            self.reverse_mapping_dict[pu] = lu
            self.reverse_mapping_dict[pv] = lv
        else:
            self.mapping_dict[lu] = pv
            self.mapping_dict[lv] = pu
            self.reverse_mapping_dict[pv] = lu
            self.reverse_mapping_dict[pu] = lv
        used_phys.add(pu)
        used_phys.add(pv)
        placed_log.add(lu)
        placed_log.add(lv)

    log_strength = defaultdict(int)
    for (a, b), w in logical_weights.items():
        log_strength[a] += w
        log_strength[b] += w

    remaining_logical = sorted(
        [q for q in logical_qubits if q not in placed_log],
        key=lambda q: -log_strength[q]
    )

    for lq in remaining_logical:
        if lq in placed_log:
            continue
        best_partner = None
        best_w = -1
        for (a, b), w in logical_weights.items():
            other = None
            if a == lq and b in placed_log:
                other = b
            elif b == lq and a in placed_log:
                other = a
            if other is not None and w > best_w:
                best_w = w
                best_partner = other

        candidates = [p for p in range(N) if p not in used_phys]
        if not candidates:
            break
        if best_partner is not None:
            anchor = self.mapping_dict[best_partner]
            candidates.sort(key=lambda p: (self.distance_matrix[anchor][p], total_dist[p]))
        else:
            candidates.sort(key=lambda p: total_dist[p])
        chosen = candidates[0]
        self.mapping_dict[lq] = chosen
        self.reverse_mapping_dict[chosen] = lq
        used_phys.add(chosen)
        placed_log.add(lq)

    free_phys = [p for p in range(N) if p not in used_phys]
    fi = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            if fi < len(free_phys):
                p = free_phys[fi]
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                used_phys.add(p)
                fi += 1

    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            for p in range(N):
                if p not in used_phys:
                    self.mapping_dict[lq] = p
                    self.reverse_mapping_dict[p] = lq
                    used_phys.add(p)
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)