def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits

    # ---- Step 1: hardware graph + bridges ----
    H = nx.Graph()
    H.add_nodes_from(range(N))
    for u, neighbors in self.backend.items():
        for v in neighbors:
            a, b = (u, v) if u < v else (v, u)
            H.add_edge(a, b)

    try:
        bridges = list(nx.bridges(H))
    except Exception:
        bridges = []

    H_cut = H.copy()
    for u, v in bridges:
        if H_cut.has_edge(u, v):
            H_cut.remove_edge(u, v)

    hw_blocks = [sorted(comp) for comp in nx.connected_components(H_cut)]
    hw_blocks.sort(key=len, reverse=True)
    block_sizes = [len(b) for b in hw_blocks]

    # ---- Step 2: logical interaction graph ----
    logical_qubits = set()
    pair_weights = defaultdict(float)
    for _gid, qubits in self.access.items():
        for q in qubits:
            logical_qubits.add(q)
        if len(qubits) == 2:
            a, b = qubits
            if a != b:
                key = (min(a, b), max(a, b))
                pair_weights[key] += 1.0

    L = nx.Graph()
    L.add_nodes_from(logical_qubits)
    for (a, b), w in pair_weights.items():
        L.add_edge(a, b, weight=w)

    logical_list = sorted(logical_qubits)

    # ---- Step 3: recursive bipartition to match block sizes ----
    def side_weight(q, side_set):
        s = 0.0
        for v, w in self.qubit_interaction_graph[q].items():
            if v in side_set:
                s += w
        return s

    def bipartition(nodes, cap_left, cap_right):
        nodes = list(nodes)
        if not nodes:
            return [], []
        if len(nodes) == 1:
            return (nodes, []) if cap_left >= 1 else ([], nodes)

        sub = L.subgraph(nodes).copy()
        # ensure connected via tiny placeholder edges so min-cut bipartition is defined
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if not sub.has_edge(u, v):
                    sub.add_edge(u, v, weight=1e-12)

        try:
            from networkx.algorithms.community import kernighan_lin_bisection
            part_a, part_b = kernighan_lin_bisection(sub, weight='weight', seed=0)
            left, right = list(part_a), list(part_b)
        except Exception:
            ordered = sorted(nodes, key=lambda q: -self.logical_activity.get(q, 0))
            half = len(ordered) // 2
            left, right = ordered[:half], ordered[half:]

        left_set, right_set = set(left), set(right)
        while len(left) > cap_left and right_set is not None:
            mover = min(left, key=lambda q: side_weight(q, left_set) - side_weight(q, right_set))
            left.remove(mover); left_set.discard(mover)
            right.append(mover); right_set.add(mover)
        while len(right) > cap_right:
            mover = min(right, key=lambda q: side_weight(q, right_set) - side_weight(q, left_set))
            right.remove(mover); right_set.discard(mover)
            left.append(mover); left_set.add(mover)
        return left, right

    def recursive_partition(nodes, sizes):
        if not sizes:
            return []
        if len(sizes) == 1:
            return [list(nodes)]
        if not nodes:
            return [[] for _ in sizes]
        mid = len(sizes) // 2
        ls, rs = sizes[:mid], sizes[mid:]
        left, right = bipartition(nodes, sum(ls), sum(rs))
        return recursive_partition(left, ls) + recursive_partition(right, rs)

    if hw_blocks and logical_list:
        logical_partition = recursive_partition(logical_list, block_sizes)
    else:
        logical_partition = [list(logical_list)] if logical_list else []
        while len(logical_partition) < len(hw_blocks):
            logical_partition.append([])

    # ---- Step 4: greedy placement inside each block ----
    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_physical = [False] * N
    placed_logicals = set()

    for block_idx, hw_block in enumerate(hw_blocks):
        if block_idx >= len(logical_partition):
            break
        chunk = logical_partition[block_idx]
        if not chunk:
            continue
        ordered = sorted(chunk, key=lambda q: -self.logical_activity.get(q, 0))
        available = set(hw_block)

        l0 = ordered[0]
        p0 = max(available, key=lambda p: self.physical_centrality.get(p, 0.0))
        mapping_dict[l0] = p0
        reverse_mapping_dict[p0] = l0
        used_physical[p0] = True
        available.discard(p0)
        placed_logicals.add(l0)

        for lq in ordered[1:]:
            if not available:
                break
            best_p, best_cost = None, float('inf')
            for p in available:
                cost, hit = 0.0, False
                for l2 in placed_logicals:
                    w = self.qubit_interaction_graph[lq].get(l2, 0)
                    if w > 0 and mapping_dict[l2] != -1:
                        cost += w * self.distance_matrix[p][mapping_dict[l2]]
                        hit = True
                if not hit:
                    cost = -self.physical_centrality.get(p, 0.0)
                if cost < best_cost:
                    best_cost, best_p = cost, p
            if best_p is None:
                best_p = next(iter(available))
            mapping_dict[lq] = best_p
            reverse_mapping_dict[best_p] = lq
            used_physical[best_p] = True
            available.discard(best_p)
            placed_logicals.add(lq)

    # ---- Step 5: identity / free-pool fallback for remaining slots ----
    for lq in range(N):
        if mapping_dict[lq] != -1:
            continue
        p = lq if (0 <= lq < N and not used_physical[lq]) else None
        if p is None:
            for cand in range(N):
                if not used_physical[cand]:
                    p = cand
                    break
        mapping_dict[lq] = p
        reverse_mapping_dict[p] = lq
        used_physical[p] = True

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)