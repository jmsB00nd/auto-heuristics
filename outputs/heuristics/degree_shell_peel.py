def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    num_q = self.num_qubits

    # Build logical interaction graph (NetworkX)
    G_logical = nx.Graph()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, w in neighbors.items():
            if q1 < q2:
                G_logical.add_edge(q1, q2, weight=w)
    # Ensure all logical qubits that appear in access are present
    for gate_id, qubits in self.access.items():
        for q in qubits:
            if not G_logical.has_node(q):
                G_logical.add_node(q)

    # Build physical coupling graph (NetworkX)
    G_physical = nx.Graph()
    for node, neighbors in self.backend.items():
        for nb in neighbors:
            if node < nb:
                G_physical.add_edge(node, nb)
    for i in range(num_q):
        if not G_physical.has_node(i):
            G_physical.add_node(i)

    def peel_shells(G):
        """Peel graph into shells of decreasing max degree."""
        shells = []
        remaining = G.copy()
        while remaining.number_of_nodes() > 0:
            if remaining.number_of_edges() == 0:
                # All remaining nodes have degree 0 — one final shell
                shells.append(sorted(remaining.nodes()))
                break
            # Find current max degree
            deg = dict(remaining.degree())
            max_deg = max(deg.values())
            if max_deg == 0:
                shells.append(sorted(remaining.nodes()))
                break
            # Extract nodes with max degree
            shell_nodes = [n for n, d in deg.items() if d == max_deg]
            # Sort by weighted degree descending for logical, or plain degree for physical
            shell_nodes.sort(key=lambda n: deg[n], reverse=True)
            shells.append(shell_nodes)
            remaining.remove_nodes_from(shell_nodes)
        return shells

    logical_shells = peel_shells(G_logical)
    physical_shells = peel_shells(G_physical)

    # Flatten physical shells for fallback
    all_physical_ordered = []
    for s in physical_shells:
        all_physical_ordered.extend(s)

    mapping = [0] * num_q
    reverse_mapping = [0] * num_q
    used_physical = set()
    mapped_logical = set()

    # For neighbor-aware placement within a shell
    physical_adj = defaultdict(set)
    for node, neighbors in self.backend.items():
        for nb in neighbors:
            physical_adj[node].add(nb)

    p_shell_idx = 0
    p_shell_offset = 0

    def get_next_physical_from_shells():
        nonlocal p_shell_idx, p_shell_offset
        while p_shell_idx < len(physical_shells):
            if p_shell_offset < len(physical_shells[p_shell_idx]):
                pq = physical_shells[p_shell_idx][p_shell_offset]
                p_shell_offset += 1
                if pq not in used_physical:
                    return pq
            else:
                p_shell_idx += 1
                p_shell_offset = 0
        return None

    for l_shell in logical_shells:
        # Collect available physical qubits from the current physical shell tier
        available_physical = []
        # Try to pull from physical shells in order
        temp_pq = get_next_physical_from_shells()
        pulled = []
        while temp_pq is not None and len(pulled) < len(l_shell):
            pulled.append(temp_pq)
            temp_pq = get_next_physical_from_shells()
        available_physical = pulled

        # For each logical qubit in this shell, pick the best physical qubit
        # Prefer physical qubits adjacent to already-placed neighbors
        remaining_physical = list(available_physical)
        for lq in l_shell:
            if not remaining_physical:
                break
            # Score each candidate physical qubit by adjacency to already-placed neighbors
            logical_neighbors = set(self.qubit_interaction_graph.get(lq, {}).keys())
            placed_neighbors_physical = set()
            for ln in logical_neighbors:
                if ln in mapped_logical:
                    placed_neighbors_physical.add(mapping[ln])

            best_pq = None
            best_score = -1
            for pq in remaining_physical:
                score = len(physical_adj[pq] & placed_neighbors_physical)
                if score > best_score:
                    best_score = score
                    best_pq = pq

            mapping[lq] = best_pq
            reverse_mapping[best_pq] = lq
            used_physical.add(best_pq)
            mapped_logical.add(lq)
            remaining_physical.remove(best_pq)

        # Put back unused pulled physical qubits — they'll be skipped via used_physical check

    # Fallback: assign any unmapped logical qubits to remaining physical qubits
    remaining_pq = [pq for pq in range(num_q) if pq not in used_physical]
    remaining_lq = [lq for lq in range(num_q) if lq not in mapped_logical]
    for lq, pq in zip(remaining_lq, remaining_pq):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)