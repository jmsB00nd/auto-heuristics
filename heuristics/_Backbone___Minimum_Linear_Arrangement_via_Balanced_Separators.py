def init_mapping(self):
    from collections import defaultdict, deque
    import math

    num_q = self.num_qubits
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    # ── Step 1: Build temporal-decay weighted interaction graph ──
    interaction_weight = defaultdict(float)
    total_gates = len(self.access)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            pair = (min(q1, q2), max(q1, q2))
            decay = 1.0 / (1.0 + gate_id / max(total_gates, 1))
            interaction_weight[pair] += decay

    # Identify logical qubits that participate in 2-qubit gates
    active_logical = set()
    for (q1, q2) in interaction_weight:
        active_logical.add(q1)
        active_logical.add(q2)

    # Build adjacency list for interaction graph
    interaction_adj = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        interaction_adj[q1][q2] = w
        interaction_adj[q2][q1] = w

    # ── Step 2: Critical-path backbone anchoring ──
    # Find the heaviest path through the interaction graph using weighted BFS/DFS
    # Start from the node with highest total interaction weight
    node_weights = defaultdict(float)
    for (q1, q2), w in interaction_weight.items():
        node_weights[q1] += w
        node_weights[q2] += w

    if not active_logical:
        # No 2-qubit gates: trivial mapping
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        return

    # Find backbone: longest weighted path via BFS from heaviest node
    start_node = max(active_logical, key=lambda q: node_weights[q])

    def weighted_bfs_farthest(start, nodes, adj):
        """BFS weighted by interaction to find farthest node."""
        dist = {start: 0.0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v, w in adj[u].items():
                if v in nodes and v not in dist:
                    dist[v] = dist[u] + 1.0 / (w + 1e-9)
                    queue.append(v)
        if not dist:
            return start, {start: 0.0}
        farthest = max(dist, key=dist.get)
        return farthest, dist

    # Two-pass BFS to find diameter endpoints
    end1, _ = weighted_bfs_farthest(start_node, active_logical, interaction_adj)
    end2, dist_from_end1 = weighted_bfs_farthest(end1, active_logical, interaction_adj)

    # Reconstruct backbone path via BFS
    def bfs_path(start, end, nodes, adj):
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                break
            for v in adj[u]:
                if v in nodes and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if end not in parent:
            return [start]
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    backbone = bfs_path(end1, end2, active_logical, interaction_adj)

    # ── Step 3: Anchor backbone onto hardware ──
    # Find a central physical qubit and do BFS to get hardware position ordering
    # Pick physical node with most connections (central hub)
    hw_degree = {pq: len(neighbors) for pq, neighbors in self.backend.items()}
    if hw_degree:
        hw_center = max(hw_degree, key=hw_degree.get)
    else:
        hw_center = 0

    # BFS traversal of hardware from center
    hw_bfs_order = []
    hw_visited = set()
    queue = deque([hw_center])
    hw_visited.add(hw_center)
    while queue:
        node = queue.popleft()
        hw_bfs_order.append(node)
        for neighbor in sorted(self.backend[node]):
            if neighbor not in hw_visited:
                hw_visited.add(neighbor)
                queue.append(neighbor)

    # Place backbone along the middle of hardware BFS order
    backbone_len = len(backbone)
    mid_start = max(0, (len(hw_bfs_order) - backbone_len) // 2)

    assigned_physical = set()
    assigned_logical = set()

    for i, lq in enumerate(backbone):
        pq = hw_bfs_order[mid_start + i]
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
        assigned_physical.add(pq)
        assigned_logical.add(lq)

    # ── Step 4: MinLA via balanced separators for remaining active qubits ──
    remaining_logical = active_logical - assigned_logical
    remaining_physical_ordered = [pq for pq in hw_bfs_order if pq not in assigned_physical]

    if remaining_logical:
        remaining_list = sorted(remaining_logical)

        # Build sub-interaction graph for remaining qubits
        def find_balanced_separator(nodes, adj):
            """Find an approximate balanced vertex separator using Fiedler vector."""
            nodes = list(nodes)
            n = len(nodes)
            if n <= 2:
                mid = n // 2
                return nodes[:mid], nodes[mid:mid+1] if mid < n else [], nodes[mid+1:] if mid+1 < n else []

            idx = {v: i for i, v in enumerate(nodes)}

            # Compute Laplacian and approximate Fiedler vector via power iteration
            # Use inverse power iteration on L (shifted) to get second smallest eigenvector
            # Simpler: use repeated diffusion to approximate Fiedler vector
            degree = [0.0] * n
            for v in nodes:
                for u, w in adj[v].items():
                    if u in idx:
                        degree[idx[v]] += w

            # Initialize random vector orthogonal to all-ones
            import random as rng
            rng.seed(42)
            x = [rng.random() - 0.5 for _ in range(n)]

            # Power iteration on (D - L) = adjacency matrix to find largest eigvec of A
            # Then Fiedler = eigvec of L with 2nd smallest eigenval ≈ complement
            for _ in range(50):
                new_x = [0.0] * n
                for v in nodes:
                    vi = idx[v]
                    for u, w in adj[v].items():
                        if u in idx:
                            new_x[vi] += w * x[idx[u]]
                # Subtract projection onto all-ones vector
                mean_val = sum(new_x) / n
                new_x = [val - mean_val for val in new_x]
                norm = math.sqrt(sum(val * val for val in new_x)) or 1.0
                x = [val / norm for val in new_x]

            # Sort by Fiedler values, split into left/separator/right
            indexed = sorted(range(n), key=lambda i: x[i])

            sep_size = max(1, n // 8)
            mid = n // 2
            sep_start = mid - sep_size // 2
            sep_end = sep_start + sep_size

            left = [nodes[indexed[i]] for i in range(sep_start)]
            separator = [nodes[indexed[i]] for i in range(sep_start, sep_end)]
            right = [nodes[indexed[i]] for i in range(sep_end, n)]

            return left, separator, right

        # Recursive MinLA ordering via balanced separators
        def minla_order(nodes, adj):
            if len(nodes) <= 1:
                return list(nodes)
            if len(nodes) == 2:
                return list(nodes)

            left, sep, right = find_balanced_separator(set(nodes), adj)

            left_order = minla_order(left, adj) if left else []
            right_order = minla_order(right, adj) if right else []

            return left_order + sep + right_order

        ordered_remaining = minla_order(remaining_list, interaction_adj)

        # Map ordered remaining logical qubits to remaining physical positions
        for i, lq in enumerate(ordered_remaining):
            if i < len(remaining_physical_ordered):
                pq = remaining_physical_ordered[i]
                mapping_dict[lq] = pq
                reverse_mapping_dict[pq] = lq
                assigned_physical.add(pq)
                assigned_logical.add(lq)

    # ── Step 5: Assign inactive logical qubits to leftover physical qubits ──
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in hw_bfs_order if pq not in assigned_physical]

    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict