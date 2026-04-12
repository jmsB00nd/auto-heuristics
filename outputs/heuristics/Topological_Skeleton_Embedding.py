def init_mapping(self):
    """
    Topological Skeleton Embedding:
    Extracts the MST of the logical interaction graph (weighted by negative interaction
    frequency) and embeds it into the hardware graph's spanning tree using a tree-in-tree
    embedding algorithm. The most frequent interactions are mapped along short physical paths.
    """
    from collections import defaultdict, deque

    num_qubits = self.num_qubits
    mapping_dict = list(range(num_qubits))
    reverse_mapping_dict = list(range(num_qubits))

    # --- Step 1: Build logical interaction graph with frequencies ---
    interaction_freq = defaultdict(int)
    logical_qubits_used = set()
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_freq[key] += 1
            logical_qubits_used.add(q1)
            logical_qubits_used.add(q2)
        elif len(qubits) == 1:
            logical_qubits_used.add(qubits[0])

    if not interaction_freq:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        return

    # --- Step 2: Build logical max spanning tree (Kruskal's, descending frequency) ---
    edges_sorted = sorted(interaction_freq.items(), key=lambda x: -x[1])

    parent = {q: q for q in logical_qubits_used}
    rank = {q: 0 for q in logical_qubits_used}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    logical_mst = defaultdict(list)
    mst_edges = []
    for (q1, q2), freq in edges_sorted:
        if union(q1, q2):
            logical_mst[q1].append((q2, freq))
            logical_mst[q2].append((q1, freq))
            mst_edges.append((q1, q2, freq))

    all_logical = set(logical_qubits_used)

    # --- Step 3: Build hardware spanning tree (BFS from most connected node) ---
    hw_degree = {node: len(neighbors) for node, neighbors in self.backend.items()}
    if not hw_degree:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        return

    hw_root = max(hw_degree, key=hw_degree.get)
    hw_parent = {}
    hw_visited = set()
    queue = deque([hw_root])
    hw_visited.add(hw_root)
    hw_bfs_order = [hw_root]

    while queue:
        node = queue.popleft()
        for neighbor in sorted(self.backend[node], key=lambda n: -hw_degree.get(n, 0)):
            if neighbor not in hw_visited:
                hw_visited.add(neighbor)
                hw_parent[neighbor] = node
                hw_bfs_order.append(neighbor)
                queue.append(neighbor)

    # --- Step 4: Root the logical MST at heaviest-interaction node ---
    logical_weight = defaultdict(int)
    for (q1, q2), freq in interaction_freq.items():
        logical_weight[q1] += freq
        logical_weight[q2] += freq

    logical_nodes_in_mst = set()
    for q1, q2, _ in mst_edges:
        logical_nodes_in_mst.add(q1)
        logical_nodes_in_mst.add(q2)

    if not logical_nodes_in_mst:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        return

    logical_root = max(logical_nodes_in_mst, key=lambda q: logical_weight.get(q, 0))

    # Build rooted logical tree
    logical_children = defaultdict(list)
    lq_visited = set([logical_root])
    lq_queue = deque([logical_root])
    logical_edge_weight = {}

    while lq_queue:
        node = lq_queue.popleft()
        for neighbor, freq in logical_mst[node]:
            if neighbor not in lq_visited:
                lq_visited.add(neighbor)
                logical_children[node].append(neighbor)
                logical_edge_weight[(node, neighbor)] = freq
                lq_queue.append(neighbor)

    # --- Step 5: Tree-in-tree embedding ---
    # Map logical_root -> hw_root, then BFS-embed children greedily
    lq_to_phys = {}
    phys_used = set()

    lq_to_phys[logical_root] = hw_root
    phys_used.add(hw_root)

    embed_queue = deque([logical_root])

    while embed_queue:
        lq_node = embed_queue.popleft()
        phys_node = lq_to_phys[lq_node]
        children = logical_children[lq_node]

        if not children:
            continue

        # Sort children by edge weight descending (heaviest first gets closest spot)
        children_sorted = sorted(
            children, key=lambda c: -logical_edge_weight.get((lq_node, c), 0)
        )

        # BFS from phys_node to find closest available physical qubits
        available_phys = []
        bfs_q = deque([phys_node])
        bfs_visited = set([phys_node])

        while bfs_q and len(available_phys) < len(children):
            p = bfs_q.popleft()
            for pn in self.backend[p]:
                if pn not in bfs_visited:
                    bfs_visited.add(pn)
                    if pn not in phys_used:
                        available_phys.append(pn)
                        if len(available_phys) >= len(children):
                            break
                    bfs_q.append(pn)

        for i, child in enumerate(children_sorted):
            if i < len(available_phys):
                target_phys = available_phys[i]
            else:
                # Fallback: closest unused physical qubit
                best_phys = None
                best_dist = float('inf')
                for pq in range(num_qubits):
                    if pq not in phys_used:
                        d = self.distance_matrix[phys_node][pq]
                        if d < best_dist:
                            best_dist = d
                            best_phys = pq
                if best_phys is not None:
                    target_phys = best_phys
                else:
                    continue

            lq_to_phys[child] = target_phys
            phys_used.add(target_phys)
            embed_queue.append(child)

    # --- Step 6: Assign remaining logical qubits not in MST ---
    unmapped_logical = [q for q in all_logical if q not in lq_to_phys]
    unused_physical = [p for p in range(num_qubits) if p not in phys_used]

    for lq, pq in zip(unmapped_logical, unused_physical):
        lq_to_phys[lq] = pq
        phys_used.add(pq)

    # --- Step 7: Convert to strict bijection via in-place swaps ---
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