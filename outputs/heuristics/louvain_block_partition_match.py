def init_mapping(self):
    import networkx as nx
    from collections import defaultdict, deque

    N = self.num_qubits

    logical_qubits = set()
    edge_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_qubits.add(a)
                continue
            logical_qubits.add(a)
            logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1

    for q in range(N):
        if q in self.qubit_interaction_graph:
            for nb, w in self.qubit_interaction_graph[q].items():
                if nb == q:
                    continue
                if w <= 0:
                    continue
                logical_qubits.add(q)
                logical_qubits.add(nb)
                key = (q, nb) if q < nb else (nb, q)
                if key not in edge_weight:
                    edge_weight[key] = w

    logical_qubits = {q for q in logical_qubits if 0 <= q < N}

    LG = nx.Graph()
    LG.add_nodes_from(logical_qubits)
    for (u, v), w in edge_weight.items():
        if 0 <= u < N and 0 <= v < N:
            LG.add_edge(u, v, weight=float(w))

    communities = []
    try:
        if LG.number_of_nodes() > 0:
            from networkx.algorithms.community import louvain_communities
            communities = list(louvain_communities(LG, weight="weight", seed=42))
    except Exception:
        communities = []
    if not communities:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = [set(c) for c in greedy_modularity_communities(LG, weight="weight")]
        except Exception:
            communities = []
    if not communities and LG.number_of_nodes() > 0:
        for comp in nx.connected_components(LG):
            communities.append(set(comp))
    communities = [set(c) for c in communities if len(c) > 0]

    def logical_score(q):
        act = self.logical_activity.get(q, 0) if hasattr(self.logical_activity, "get") else 0
        deg = LG.degree(q, weight="weight") if q in LG else 0
        return (act, deg)

    communities.sort(key=lambda c: (-sum(logical_score(q)[0] for q in c), -len(c)))

    HG = nx.Graph()
    HG.add_nodes_from(range(N))
    for p, nbrs in self.backend.items():
        for nb in nbrs:
            if 0 <= p < N and 0 <= nb < N and p != nb:
                HG.add_edge(p, nb)

    centrality = {}
    for p in range(N):
        try:
            centrality[p] = float(self.physical_centrality.get(p, 0.0))
        except Exception:
            centrality[p] = 0.0

    used_physical = set()
    mapping = [-1] * N

    def grow_block(size, forbidden):
        candidates = [p for p in range(N) if p not in forbidden]
        if not candidates:
            return []
        candidates.sort(key=lambda p: -centrality.get(p, 0.0))
        for seed in candidates:
            block = [seed]
            visited = {seed}
            queue = deque()
            for nb in HG.neighbors(seed):
                if nb not in forbidden and nb not in visited:
                    queue.append(nb)
            frontier = []
            for nb in queue:
                frontier.append(nb)
            frontier_set = set(frontier)
            while len(block) < size and frontier:
                frontier.sort(key=lambda p: -centrality.get(p, 0.0))
                chosen = frontier.pop(0)
                frontier_set.discard(chosen)
                if chosen in visited or chosen in forbidden:
                    continue
                visited.add(chosen)
                block.append(chosen)
                for nb in HG.neighbors(chosen):
                    if nb not in visited and nb not in forbidden and nb not in frontier_set:
                        frontier.append(nb)
                        frontier_set.add(nb)
            if len(block) >= 1:
                return block
        return []

    for comm in communities:
        comm_size = len(comm)
        if comm_size == 0:
            continue
        block = grow_block(comm_size, used_physical)
        if not block:
            continue

        logicals_sorted = sorted(comm, key=lambda q: -logical_score(q)[0])
        physicals_sorted = sorted(block, key=lambda p: -centrality.get(p, 0.0))

        for i, lq in enumerate(logicals_sorted):
            if i >= len(physicals_sorted):
                break
            pq = physicals_sorted[i]
            if 0 <= lq < N and pq not in used_physical and mapping[lq] == -1:
                mapping[lq] = pq
                used_physical.add(pq)

    unmapped_logicals = [q for q in range(N) if mapping[q] == -1]
    unmapped_logicals.sort(key=lambda q: -logical_score(q)[0])
    available_physicals = sorted([p for p in range(N) if p not in used_physical],
                                 key=lambda p: -centrality.get(p, 0.0))

    ai = 0
    for lq in unmapped_logicals:
        if ai >= len(available_physicals):
            break
        pq = available_physicals[ai]
        mapping[lq] = pq
        used_physical.add(pq)
        ai += 1

    remaining_phys = [p for p in range(N) if p not in used_physical]
    remaining_log = [q for q in range(N) if mapping[q] == -1]
    for lq, pq in zip(remaining_log, remaining_phys):
        mapping[lq] = pq
        used_physical.add(pq)

    if -1 in mapping:
        leftover_phys = [p for p in range(N) if p not in used_physical]
        idx = 0
        for q in range(N):
            if mapping[q] == -1 and idx < len(leftover_phys):
                mapping[q] = leftover_phys[idx]
                used_physical.add(leftover_phys[idx])
                idx += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [-1] * N
    for lq, pq in enumerate(self.mapping_dict):
        if 0 <= pq < N:
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)