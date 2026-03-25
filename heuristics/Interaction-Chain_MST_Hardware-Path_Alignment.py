def init_mapping(self):
    from collections import defaultdict, deque

    # --- Step 1: Build Qubit Interaction Graph (QIG) ---
    # Edge weight = number of 2-qubit gates between the pair (interaction frequency)
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

    # Trivial mapping if nothing to place
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    if not logical_qubits or not interaction_weight:
        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Collect nodes that appear in 2-qubit interactions
    qig_nodes = set()
    for q1, q2 in interaction_weight:
        qig_nodes.add(q1)
        qig_nodes.add(q2)

    # --- Step 2: Maximum Spanning Tree of QIG (Kruskal's) ---
    # Heavier edges = more frequent interactions → kept in the MST skeleton
    edges_sorted = sorted(interaction_weight.items(), key=lambda x: -x[1])

    parent = {q: q for q in qig_nodes}
    rank   = {q: 0 for q in qig_nodes}

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

    qig_mst = defaultdict(dict)   # qig_mst[u][v] = weight
    for (q1, q2), w in edges_sorted:
        if union(q1, q2):
            qig_mst[q1][q2] = w
            qig_mst[q2][q1] = w

    # --- Step 3: Find heaviest path (weighted diameter) in QIG MST ---
    # Double-DFS on the tree: first pass finds one endpoint, second gives the full path.
    def find_weighted_diameter_path(tree, nodes):
        if len(nodes) == 1:
            return list(nodes)

        def dfs_farthest(root):
            """Return (farthest_node, path_from_root_to_farthest)."""
            best_node, best_dist = root, -1.0
            parent_map = {root: None}
            stack = [(root, None, 0.0)]
            while stack:
                node, par, dist = stack.pop()
                if dist > best_dist:
                    best_dist, best_node = dist, node
                for nb, w in tree.get(node, {}).items():
                    if nb not in parent_map:
                        parent_map[nb] = node
                        stack.append((nb, node, dist + w))
            # Reconstruct path root → best_node
            path, cur = [], best_node
            while cur is not None:
                path.append(cur)
                cur = parent_map.get(cur)
            return best_node, path[::-1]

        start = next(iter(nodes))
        end1, _    = dfs_farthest(start)
        end2, path = dfs_farthest(end1)
        return path

    qig_diameter_path = find_weighted_diameter_path(qig_mst, qig_nodes)

    # --- Step 4: Find hardware diameter path (unweighted BFS double-sweep) ---
    def find_hw_diameter_path(hw_graph, hw_nodes):
        if len(hw_nodes) == 1:
            return list(hw_nodes)

        def bfs_farthest(root):
            dist = {root: 0}
            parent_map = {root: None}
            queue = deque([root])
            farthest, max_d = root, 0
            while queue:
                node = queue.popleft()
                for nb in hw_graph[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        parent_map[nb] = node
                        queue.append(nb)
                        if dist[nb] > max_d:
                            max_d, farthest = dist[nb], nb
            path, cur = [], farthest
            while cur is not None:
                path.append(cur)
                cur = parent_map[cur]
            return farthest, path[::-1]

        start = next(iter(hw_nodes))
        end1, _    = bfs_farthest(start)
        end2, path = bfs_farthest(end1)
        return path

    hw_diameter_path = find_hw_diameter_path(self.backend, set(physical_qubits))

    # --- Step 5: Center-align QIG interaction chain onto hardware diameter path ---
    lq_path = qig_diameter_path
    hw_path = hw_diameter_path
    lq_to_phys  = {}
    placed_phys = set()

    offset = max(0, (len(hw_path) - len(lq_path)) // 2)
    for i, lq in enumerate(lq_path):
        hw_idx = offset + i
        if hw_idx < len(hw_path):
            phys = hw_path[hw_idx]
        else:
            # hw_path is shorter than lq_path: spill to closest available node
            anchor = hw_path[-1]
            available = [p for p in physical_qubits if p not in placed_phys]
            if not available:
                break
            phys = min(available, key=lambda p: self.distance_matrix[anchor][p])
        lq_to_phys[lq] = phys
        placed_phys.add(phys)

    # --- Step 6: BFS subtree expansion — match QIG-MST branches to hardware neighbors ---
    visited_lq = set(lq_to_phys.keys())
    bfs_queue  = deque(lq_to_phys.items())   # (logical, physical)

    while bfs_queue:
        lq, phys = bfs_queue.popleft()

        # Unplaced QIG-MST children, heaviest edge first
        lq_children = sorted(
            [(nb, w) for nb, w in qig_mst.get(lq, {}).items() if nb not in visited_lq],
            key=lambda x: -x[1]
        )
        if not lq_children:
            continue

        # Direct hardware neighbors first, then globally closest fallback
        hw_direct = [nb for nb in self.backend[phys] if nb not in placed_phys]
        hw_fallback = sorted(
            [p for p in physical_qubits if p not in placed_phys and p not in hw_direct],
            key=lambda p: self.distance_matrix[phys][p]
        )
        hw_candidates = hw_direct + hw_fallback

        for (nb_lq, _), nb_phys in zip(lq_children, hw_candidates):
            lq_to_phys[nb_lq]  = nb_phys
            placed_phys.add(nb_phys)
            visited_lq.add(nb_lq)
            bfs_queue.append((nb_lq, nb_phys))

    # --- Step 7: Greedy placement for remaining qubits (isolated / unreachable) ---
    # Use weighted distance to already-placed interaction partners where possible.
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
                key=lambda p: sum(w * self.distance_matrix[p][lq_to_phys[nb]]
                                  for nb, w in partners)
            )
        else:
            # Truly isolated qubit: place on highest-degree available node
            best_phys = max(available, key=lambda p: len(self.backend[p]))
        lq_to_phys[lq] = best_phys
        placed_phys.add(best_phys)

    # --- Step 8: Build strict 1-to-1 bijection via swap-based assignment ---
    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]            = target_phys
        mapping_dict[displaced_lq]  = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)