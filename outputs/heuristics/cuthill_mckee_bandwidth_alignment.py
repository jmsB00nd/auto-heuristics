def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # Step 1: weighted logical interaction graph from self.access
    weight = defaultdict(int)
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                active_logicals.add(a)
                continue
            if a > b:
                a, b = b, a
            weight[(a, b)] += 1
            active_logicals.add(a)
            active_logicals.add(b)

    def rcm_order(num_nodes, edges_with_weight, active_nodes=None):
        # Try scipy RCM first, fall back to a manual BFS-based RCM
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee
            if num_nodes == 0:
                return []
            rows, cols, vals = [], [], []
            for (u, v), w in edges_with_weight.items():
                if u >= num_nodes or v >= num_nodes:
                    continue
                rows.append(u); cols.append(v); vals.append(float(w))
                rows.append(v); cols.append(u); vals.append(float(w))
            if not rows:
                # Empty graph: return active nodes first, then the rest
                base = list(active_nodes) if active_nodes is not None else []
                rest = [i for i in range(num_nodes) if i not in set(base)]
                return base + rest
            mat = csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
            perm = reverse_cuthill_mckee(mat, symmetric_mode=True)
            return [int(x) for x in perm]
        except Exception:
            # Manual RCM fallback
            adj = defaultdict(list)
            for (u, v), w in edges_with_weight.items():
                if u >= num_nodes or v >= num_nodes:
                    continue
                adj[u].append(v)
                adj[v].append(u)
            degree = {i: len(adj[i]) for i in range(num_nodes)}
            visited = [False] * num_nodes
            order = []
            # Process components by ascending degree of the seed
            seeds = sorted(range(num_nodes), key=lambda x: (degree[x], x))
            for seed in seeds:
                if visited[seed]:
                    continue
                # BFS using ascending-degree neighbor expansion (Cuthill-McKee)
                queue = [seed]
                visited[seed] = True
                head = 0
                while head < len(queue):
                    node = queue[head]; head += 1
                    order.append(node)
                    nbrs = sorted(adj[node], key=lambda x: (degree[x], x))
                    for nb in nbrs:
                        if not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
            order.reverse()  # Reverse Cuthill-McKee
            return order

    # Step 2: physical coupling edges
    phys_edges = defaultdict(int)
    seen_phys = set()
    for edge in self.backend_connections:
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen_phys:
            continue
        seen_phys.add(key)
        phys_edges[key] = 1

    # Step 3: RCM ordering of logical qubits (only active ones contribute meaningfully)
    # Build a graph sized to N so indices line up; isolated logicals will appear at the end.
    logical_order_full = rcm_order(N, weight, active_nodes=active_logicals)
    # Prioritize active logical qubits in the ordering, preserving RCM-derived relative order
    active_set = set(active_logicals)
    logical_order = [q for q in logical_order_full if q in active_set]
    inactive_logical = [q for q in logical_order_full if q not in active_set]

    # Step 4: RCM ordering of physical qubits
    physical_order = rcm_order(N, phys_edges)

    # Step 5: Align index-by-index
    used_physical = set()
    assigned_logical = set()

    aligned_len = min(len(logical_order), len(physical_order))
    for i in range(aligned_len):
        lq = logical_order[i]
        pq = physical_order[i]
        if lq in assigned_logical or pq in used_physical:
            continue
        if lq < 0 or lq >= N or pq < 0 or pq >= N:
            continue
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        assigned_logical.add(lq)
        used_physical.add(pq)

    # Step 6: Identity-style fallback for any remaining logical qubits
    remaining_physical = [p for p in physical_order if p not in used_physical]
    remaining_physical += [p for p in range(N) if p not in used_physical and p not in remaining_physical]
    rp_idx = 0

    # First place any inactive logical qubits that came out of RCM
    for lq in inactive_logical + [q for q in range(N) if q not in assigned_logical]:
        if lq in assigned_logical:
            continue
        # Try identity first if free
        if 0 <= lq < N and lq not in used_physical:
            self.mapping_dict[lq] = lq
            self.reverse_mapping_dict[lq] = lq
            assigned_logical.add(lq)
            used_physical.add(lq)
            continue
        # Otherwise pull from remaining physical pool
        while rp_idx < len(remaining_physical) and remaining_physical[rp_idx] in used_physical:
            rp_idx += 1
        if rp_idx < len(remaining_physical):
            pq = remaining_physical[rp_idx]
            rp_idx += 1
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            assigned_logical.add(lq)
            used_physical.add(pq)

    # Final safety sweep: any logical still unmapped gets the next free physical
    free_phys = [p for p in range(N) if p not in used_physical]
    fp_idx = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            while fp_idx < len(free_phys) and free_phys[fp_idx] in used_physical:
                fp_idx += 1
            if fp_idx < len(free_phys):
                pq = free_phys[fp_idx]
                fp_idx += 1
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_physical.add(pq)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)