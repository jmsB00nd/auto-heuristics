def init_mapping(self):
    from collections import defaultdict, deque

    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    # --- Step 1: Build interaction edges and max spanning tree ---
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits or not interaction_weight:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    qig_nodes = set()
    for q1, q2 in interaction_weight:
        qig_nodes.add(q1)
        qig_nodes.add(q2)

    # Kruskal's for max spanning tree
    edges_sorted = sorted(interaction_weight.items(), key=lambda x: -x[1])
    parent = {q: q for q in qig_nodes}
    rank = {q: 0 for q in qig_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    qig_mst = defaultdict(dict)
    for (q1, q2), w in edges_sorted:
        if union(q1, q2):
            qig_mst[q1][q2] = w
            qig_mst[q2][q1] = w

    # --- Step 2: Build "best" spanning tree of hardware graph (BFS from highest-degree node) ---
    hw_root = max(physical_qubits, key=lambda p: len(self.backend[p]))
    hw_tree = defaultdict(dict)
    hw_visited = {hw_root}
    bfs_q = deque([hw_root])
    while bfs_q:
        node = bfs_q.popleft()
        for nb in sorted(self.backend[node], key=lambda n: -len(self.backend[n])):
            if nb not in hw_visited:
                hw_visited.add(nb)
                hw_tree[node][nb] = 1
                hw_tree[nb][node] = 1
                bfs_q.append(nb)

    # --- Step 3: Root both trees ---
    # Root interaction MST at highest-activity node
    ig_root = max(qig_nodes, key=lambda q: sum(self.qubit_interaction_graph[q].values()))

    # Build children lists (rooted tree)
    def build_children(tree, root, nodes):
        children = defaultdict(list)
        visited = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for nb in tree.get(node, {}):
                if nb not in visited:
                    visited.add(nb)
                    children[node].append(nb)
                    stack.append(nb)
        return children

    ig_children = build_children(qig_mst, ig_root, qig_nodes)
    hw_children = build_children(hw_tree, hw_root, set(physical_qubits))

    # --- Step 4: Compute subtree interaction weights and subtree sizes ---
    def compute_subtree_weights(children, root, tree):
        weights = defaultdict(float)
        order = []
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for ch in children[node]:
                stack.append(ch)
        for node in reversed(order):
            w = 0.0
            for ch in children[node]:
                w += tree.get(node, {}).get(ch, 0.0) + weights[ch]
            weights[node] = w
        return weights

    def compute_subtree_sizes(children, root):
        sizes = defaultdict(int)
        order = []
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for ch in children[node]:
                stack.append(ch)
        for node in reversed(order):
            s = 1
            for ch in children[node]:
                s += sizes[ch]
            sizes[node] = s
        return sizes

    ig_subtree_w = compute_subtree_weights(ig_children, ig_root, qig_mst)
    hw_subtree_s = compute_subtree_sizes(hw_children, hw_root)

    # --- Step 5: Simultaneous rooted DFS with child matching ---
    lq_to_phys = {}
    placed_phys = set()
    lq_to_phys[ig_root] = hw_root
    placed_phys.add(hw_root)

    dfs_stack = [(ig_root, hw_root)]
    while dfs_stack:
        ig_node, hw_node = dfs_stack.pop()
        ig_ch = sorted(ig_children[ig_node], key=lambda c: -ig_subtree_w[c])
        hw_ch = sorted(hw_children[hw_node], key=lambda c: -hw_subtree_s[c])

        for i, ig_child in enumerate(ig_ch):
            if i < len(hw_ch):
                phys = hw_ch[i]
            else:
                # Overflow: find closest available physical qubit to hw_node
                available = [p for p in physical_qubits if p not in placed_phys]
                if not available:
                    break
                phys = min(available, key=lambda p: self.distance_matrix[hw_node][p])
            lq_to_phys[ig_child] = phys
            placed_phys.add(phys)
            dfs_stack.append((ig_child, phys))

    # --- Step 6: Place remaining logical qubits (isolated or not in MST) ---
    interaction_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        interaction_neighbors[q1][q2] = w
        interaction_neighbors[q2][q1] = w

    remaining_lq = [lq for lq in logical_qubits if lq not in lq_to_phys]
    for lq in remaining_lq:
        available = [p for p in physical_qubits if p not in placed_phys]
        if not available:
            break
        partners = [(nb, w) for nb, w in interaction_neighbors[lq].items() if nb in lq_to_phys]
        if partners:
            best_phys = min(
                available,
                key=lambda p: sum(w * self.distance_matrix[p][lq_to_phys[nb]] for nb, w in partners)
            )
        else:
            best_phys = max(available, key=lambda p: len(self.backend[p]))
        lq_to_phys[lq] = best_phys
        placed_phys.add(best_phys)

    # --- Step 7: Build strict 1-to-1 bijection via swap-based assignment ---
    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)