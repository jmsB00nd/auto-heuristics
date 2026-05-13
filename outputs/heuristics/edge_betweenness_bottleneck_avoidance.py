def init_mapping(self):
    import networkx as nx
    import heapq

    N = self.num_qubits

    # --- Build coupling graph ---
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u, neigh in self.backend.items():
        for v in neigh:
            if u < v:
                G.add_edge(u, v)

    # --- Edge betweenness on coupling graph ---
    try:
        eb = nx.edge_betweenness_centrality(G, normalized=True)
    except Exception:
        eb = {e: 0.0 for e in G.edges()}

    # --- Aggregate bottleneck score per physical qubit ---
    bottleneck = [0.0] * N
    for (a, b), val in eb.items():
        bottleneck[a] += val
        bottleneck[b] += val

    # --- Collect logical qubits and interactions from self.access ---
    logical_set = set()
    interactions = []
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_set.add(q)
        if len(qubits) == 2:
            interactions.append((qubits[0], qubits[1]))

    # Activity score per logical (fall back to local count if logical_activity missing)
    activity = {}
    for L in logical_set:
        activity[L] = self.logical_activity.get(L, 0) if hasattr(self, "logical_activity") else 0
        if activity[L] == 0:
            # rebuild from interactions
            for (a, b) in interactions:
                if a == L or b == L:
                    activity[L] += 1

    # --- Order logicals by activity desc ---
    ordered_logicals = sorted(logical_set, key=lambda L: -activity.get(L, 0))

    # --- Initialize mapping containers ---
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N
    used_phys = set()
    assigned_log = set()

    centrality = getattr(self, "physical_centrality", {}) or {}

    def score_phys(p, placed_partners):
        # primary: low bottleneck. secondary: proximity to already-placed partners. tertiary: high centrality
        prox = 0.0
        for (partner_phys, weight) in placed_partners:
            d = self.distance_matrix[p][partner_phys]
            prox += weight * d
        return (bottleneck[p] + 0.5 * prox, -centrality.get(p, 0.0))

    qig = getattr(self, "qubit_interaction_graph", None)

    for L in ordered_logicals:
        # Compute placed partners for this logical
        placed_partners = []
        if qig is not None and L in qig:
            for partner_log, w in qig[L].items():
                if partner_log in assigned_log:
                    placed_partners.append((self.mapping_dict[partner_log], w))

        best_p = None
        best_score = None
        for p in range(N):
            if p in used_phys:
                continue
            s = score_phys(p, placed_partners)
            if best_score is None or s < best_score:
                best_score = s
                best_p = p
        if best_p is None:
            break
        self.mapping_dict[L] = best_p
        self.reverse_mapping_dict[best_p] = L
        used_phys.add(best_p)
        assigned_log.add(L)

    # --- Back-fill unmapped logicals with remaining physicals (most central first) ---
    remaining_phys = [p for p in range(N) if p not in used_phys]
    remaining_phys.sort(key=lambda p: (bottleneck[p], -centrality.get(p, 0.0)))
    remaining_log = [L for L in range(N) if self.mapping_dict[L] == -1]

    for L, p in zip(remaining_log, remaining_phys):
        self.mapping_dict[L] = p
        self.reverse_mapping_dict[p] = L
        used_phys.add(p)

    # --- Identity safety fallback for any slot still -1 ---
    if any(x == -1 for x in self.mapping_dict):
        free = [p for p in range(N) if p not in used_phys]
        for L in range(N):
            if self.mapping_dict[L] == -1:
                if free:
                    p = free.pop(0)
                else:
                    p = L
                self.mapping_dict[L] = p
                self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)