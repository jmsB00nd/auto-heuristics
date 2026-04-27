def init_mapping(self):
    import networkx as nx
    from collections import defaultdict, Counter

    N = self.num_qubits

    # --- Build physical coupling graph ---
    G_phys = nx.Graph()
    G_phys.add_nodes_from(range(N))
    for (a, b) in self.backend_connections:
        if a != b:
            G_phys.add_edge(a, b)

    # --- Extract logical 2q interactions from self.access ---
    logical_weight = Counter()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            u, v = qubits[0], qubits[1]
            if u != v:
                logical_weight[(min(u, v), max(u, v))] += 1

    logical_qubits = set()
    for (u, v) in logical_weight:
        logical_qubits.add(u); logical_qubits.add(v)

    G_log = nx.Graph()
    G_log.add_nodes_from(logical_qubits)
    for (u, v), w in logical_weight.items():
        G_log.add_edge(u, v, weight=w)

    # --- Articulation points, bridges, biconnected components on G_phys ---
    try:
        artic_points = set(nx.articulation_points(G_phys))
    except Exception:
        artic_points = set()
    try:
        bridges = {(min(a, b), max(a, b)) for (a, b) in nx.bridges(G_phys)}
    except Exception:
        bridges = set()
    try:
        bicomps = [set(c) for c in nx.biconnected_components(G_phys)]
    except Exception:
        bicomps = []
    if not bicomps:
        bicomps = [set(range(N))]

    # --- Centrality ---
    try:
        phys_bet = nx.betweenness_centrality(G_phys)
    except Exception:
        phys_bet = {n: 0.0 for n in G_phys.nodes()}
    try:
        log_bet = nx.betweenness_centrality(G_log, weight='weight') if len(G_log) > 0 else {}
    except Exception:
        log_bet = {n: 0.0 for n in G_log.nodes()}

    log_strength = defaultdict(float)
    for (u, v), w in logical_weight.items():
        log_strength[u] += w
        log_strength[v] += w

    def phys_score(p):
        return (1 if p in artic_points else 0, phys_bet.get(p, 0.0))

    # --- Logical connected components, heaviest first ---
    log_components = [set(c) for c in nx.connected_components(G_log)] if len(G_log) > 0 else []

    def comp_weight(c):
        return sum(w for (u, v), w in logical_weight.items() if u in c and v in c)

    log_components.sort(key=comp_weight, reverse=True)
    bicomps_by_size = sorted(bicomps, key=lambda c: len(c), reverse=True)

    used_phys = set()
    mapping = {}

    # --- Greedy placement: each logical comp -> best biconnected comp ---
    for comp in log_components:
        comp_logicals = sorted(
            comp,
            key=lambda q: (log_bet.get(q, 0.0), log_strength.get(q, 0.0)),
            reverse=True
        )

        best_bi = None
        best_score = (-1, -1, -1)
        for bc in bicomps_by_size:
            avail = bc - used_phys
            if not avail:
                continue
            artic_count = sum(1 for p in avail if p in artic_points)
            fits = 1 if len(avail) >= len(comp) else 0
            score = (fits, artic_count, len(avail))
            if score > best_score:
                best_score = score
                best_bi = bc

        if best_bi is None:
            avail_list = [p for p in range(N) if p not in used_phys]
        else:
            avail_list = list(best_bi - used_phys)
        avail_list.sort(key=phys_score, reverse=True)

        for lq in comp_logicals:
            pq = None
            while avail_list:
                cand = avail_list.pop(0)
                if cand not in used_phys:
                    pq = cand
                    break
            if pq is None:
                spill = [p for p in range(N) if p not in used_phys]
                spill.sort(key=phys_score, reverse=True)
                if not spill:
                    break
                pq = spill[0]
            if 0 <= lq < N:
                mapping[lq] = pq
                used_phys.add(pq)

    # --- Materialize lists ---
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N
    for lq, pq in mapping.items():
        if 0 <= lq < N and self.mapping_dict[lq] is None:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    # --- Identity-style fallback for unmapped logical ids ---
    remaining = [p for p in range(N) if p not in used_phys]
    rp_idx = 0
    for lq in range(N):
        if self.mapping_dict[lq] is None:
            if lq not in used_phys:
                pq = lq
            else:
                while rp_idx < len(remaining) and remaining[rp_idx] in used_phys:
                    rp_idx += 1
                if rp_idx >= len(remaining):
                    continue
                pq = remaining[rp_idx]
                rp_idx += 1
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_phys.add(pq)

    # --- Last-resort sweep for any remaining None ---
    for lq in range(N):
        if self.mapping_dict[lq] is None:
            for p in range(N):
                if p not in used_phys:
                    self.mapping_dict[lq] = p
                    self.reverse_mapping_dict[p] = lq
                    used_phys.add(p)
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)