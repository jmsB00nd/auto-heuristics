def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    # ---- 1. Build weighted logical interaction graph from self.access ----
    logical_adj = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a][b] += 1.0
            logical_adj[b][a] += 1.0
            logical_qubits.add(a)
            logical_qubits.add(b)

    # ---- 2. Build physical graph ----
    phys_adj = defaultdict(set)
    for (u, v) in self.backend_connections:
        if 0 <= u < N and 0 <= v < N and u != v:
            phys_adj[u].add(v)
            phys_adj[v].add(u)

    # ---- 3. Weighted k-core peeling (returns coreness dict) ----
    def weighted_coreness(nodes, adj_weighted):
        # adj_weighted: node -> {neighbor: weight}
        coreness = {}
        deg = {n: sum(adj_weighted[n].values()) for n in nodes}
        active = set(nodes)
        # min-heap of (degree, node); lazy deletion
        heap = [(deg[n], n) for n in nodes]
        heapq.heapify(heap)
        current_core = 0.0
        while active:
            while heap and (heap[0][1] not in active or heap[0][0] != deg[heap[0][1]]):
                heapq.heappop(heap)
            if not heap:
                break
            d, node = heapq.heappop(heap)
            if node not in active:
                continue
            current_core = max(current_core, d)
            coreness[node] = current_core
            active.discard(node)
            for nb, w in adj_weighted[node].items():
                if nb in active:
                    deg[nb] -= w
                    heapq.heappush(heap, (deg[nb], nb))
        return coreness

    # Unweighted coreness for physical graph
    def unweighted_coreness(nodes, adj_set):
        coreness = {}
        deg = {n: len(adj_set[n]) for n in nodes}
        active = set(nodes)
        heap = [(deg[n], n) for n in nodes]
        heapq.heapify(heap)
        current_core = 0
        while active:
            while heap and (heap[0][1] not in active or heap[0][0] != deg[heap[0][1]]):
                heapq.heappop(heap)
            if not heap:
                break
            d, node = heapq.heappop(heap)
            if node not in active:
                continue
            current_core = max(current_core, d)
            coreness[node] = current_core
            active.discard(node)
            for nb in adj_set[node]:
                if nb in active:
                    deg[nb] -= 1
                    heapq.heappush(heap, (deg[nb], nb))
        return coreness

    log_core = weighted_coreness(list(logical_qubits), logical_adj) if logical_qubits else {}
    phys_nodes = list(range(N))
    phys_core = unweighted_coreness(phys_nodes, phys_adj)

    # ---- 4. Order qubits by (coreness desc, weighted-degree desc) ----
    log_wdeg = {q: sum(logical_adj[q].values()) for q in logical_qubits}
    logical_order = sorted(
        logical_qubits,
        key=lambda q: (-log_core.get(q, 0.0), -log_wdeg.get(q, 0.0), q),
    )
    phys_deg = {p: len(phys_adj[p]) for p in phys_nodes}
    phys_order_base = sorted(
        phys_nodes,
        key=lambda p: (-phys_core.get(p, 0), -phys_deg.get(p, 0), p),
    )

    # ---- 5. Greedy concentric placement ----
    mapping = [None] * N
    reverse = [None] * N
    used_phys = set()
    placed_logical = []  # list of (logical_q, physical_q)

    for lq in logical_order:
        # Compute candidate cost: prefer physical qubits with high coreness
        # AND close to already-placed neighbors of lq in logical graph.
        neighbors_placed = [
            (mapping[nb], w)
            for nb, w in logical_adj[lq].items()
            if mapping[nb] is not None
        ]

        best_phys = None
        best_key = None
        for p in phys_order_base:
            if p in used_phys:
                continue
            if neighbors_placed:
                dist_cost = 0.0
                for (pn, w) in neighbors_placed:
                    dist_cost += w * self.distance_matrix[p][pn]
            else:
                dist_cost = 0.0
            # rank: lower dist_cost first, then higher physical coreness, then higher degree
            key = (
                dist_cost,
                -phys_core.get(p, 0),
                -phys_deg.get(p, 0),
                p,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_phys = p
        if best_phys is None:
            break
        mapping[lq] = best_phys
        reverse[best_phys] = lq
        used_phys.add(best_phys)
        placed_logical.append((lq, best_phys))

    # ---- 6. Identity-style fill for remaining logical/physical ids ----
    remaining_phys = [p for p in range(N) if p not in used_phys]
    rp_iter = iter(remaining_phys)
    for lq in range(N):
        if mapping[lq] is None:
            try:
                p = next(rp_iter)
            except StopIteration:
                break
            mapping[lq] = p
            reverse[p] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)