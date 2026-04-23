def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits
    phys_N = N - 1  # physical qubits are indexed 0..phys_N-1 per backend

    # Build logical interaction graph from 2-qubit gates in self.access
    logical_adj = defaultdict(set)
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if q1 == q2:
                continue
            logical_adj[q1].add(q2)
            logical_adj[q2].add(q1)
            logical_nodes.add(q1)
            logical_nodes.add(q2)

    # Build physical coupling graph from self.backend_connections
    physical_adj = defaultdict(set)
    physical_nodes = set(range(phys_N))
    for u, v in self.backend_connections:
        if u == v:
            continue
        if 0 <= u < phys_N and 0 <= v < phys_N:
            physical_adj[u].add(v)
            physical_adj[v].add(u)

    def rcm_order(adj, nodes):
        # Reverse Cuthill-McKee on connected components
        nodes = list(nodes)
        if not nodes:
            return []
        degree = {n: len(adj.get(n, ())) for n in nodes}
        visited = set()
        order = []
        # Process components in order of ascending min-degree start
        remaining = set(nodes)
        while remaining:
            # Pick start: node of minimum degree in remaining
            start = min(remaining, key=lambda n: (degree[n], n))
            # Cuthill-McKee BFS from start
            queue = deque([start])
            comp_order = []
            local_visited = {start}
            while queue:
                node = queue.popleft()
                comp_order.append(node)
                neighbors = [m for m in adj.get(node, ()) if m not in local_visited and m in remaining]
                neighbors.sort(key=lambda m: (degree[m], m))
                for m in neighbors:
                    local_visited.add(m)
                    queue.append(m)
            # Any unreached nodes in this component (shouldn't happen if connected)
            order.extend(comp_order)
            remaining -= local_visited
            visited |= local_visited
        # Reverse to get RCM
        order.reverse()
        return order

    logical_order = rcm_order(logical_adj, logical_nodes)
    physical_order = rcm_order(physical_adj, physical_nodes)

    # Initialize list-based mappings of length N (identity as safe default)
    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    used_physical = set()
    assigned_logical = set()

    # Positional pairing by RCM index
    pair_len = min(len(logical_order), len(physical_order))
    for i in range(pair_len):
        lq = logical_order[i]
        pq = physical_order[i]
        if 0 <= lq < N and 0 <= pq < phys_N and pq not in used_physical and lq not in assigned_logical:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
            assigned_logical.add(lq)

    # Fill remaining logical qubits with unused physical qubits (identity-preferring)
    remaining_physical = [p for p in range(phys_N) if p not in used_physical]
    remaining_physical_set = set(remaining_physical)
    # Prefer identity assignment for unassigned logicals when possible
    unassigned_logicals = [lq for lq in range(N) if lq not in assigned_logical]

    # First pass: identity if available
    identity_taken = []
    for lq in unassigned_logicals:
        if lq in remaining_physical_set:
            self.mapping_dict[lq] = lq
            self.reverse_mapping_dict[lq] = lq
            used_physical.add(lq)
            assigned_logical.add(lq)
            remaining_physical_set.discard(lq)
            identity_taken.append(lq)

    # Second pass: assign remaining physicals to remaining logicals
    remaining_physical = sorted(remaining_physical_set)
    still_unassigned = [lq for lq in range(N) if lq not in assigned_logical]
    idx = 0
    for lq in still_unassigned:
        if idx < len(remaining_physical):
            pq = remaining_physical[idx]
            idx += 1
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
            assigned_logical.add(lq)
        else:
            # Fall back: keep identity slot (pq == lq if within range), guaranteed unique
            # since lq was not used yet (index N slot, typically the padding slot)
            self.mapping_dict[lq] = lq
            if lq < len(self.reverse_mapping_dict):
                self.reverse_mapping_dict[lq] = lq

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            self.isl_mapping = None

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)