def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    num_q = self.num_qubits

    # --- Build logical interaction graph (weighted) ---
    log_edges = {}
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            key = (min(q1, q2), max(q1, q2))
            log_edges[key] = w
    logical_nodes = set()
    for (a, b) in log_edges:
        logical_nodes.add(a)
        logical_nodes.add(b)

    # --- Build physical coupling graph (weighted by connectivity) ---
    phys_edges = {}
    phys_degree = defaultdict(int)
    for node, neighbors in self.backend.items():
        phys_degree[node] = len(neighbors)
    for node, neighbors in self.backend.items():
        for nb in neighbors:
            key = (min(node, nb), max(node, nb))
            if key not in phys_edges:
                phys_edges[key] = phys_degree[node] + phys_degree[nb]
    physical_nodes = set()
    for (a, b) in phys_edges:
        physical_nodes.add(a)
        physical_nodes.add(b)

    # --- Coarsening helper ---
    def coarsen_graph(nodes, edges):
        """
        Iteratively contract the heaviest edge.
        Returns list of layers: each layer is (merged_node, node_a, node_b).
        nodes: set of node ids (each may be a frozenset of original ids)
        edges: dict {(min,max): weight}
        """
        node_to_super = {}
        for n in nodes:
            node_to_super[n] = frozenset([n]) if not isinstance(n, frozenset) else n

        current_nodes = set(nodes)
        current_edges = dict(edges)
        layers = []

        while len(current_nodes) > 1 and current_edges:
            # Find heaviest edge
            best_edge = max(current_edges, key=lambda e: current_edges[e])
            a, b = best_edge
            w = current_edges[best_edge]

            # Merge b into a
            new_super = node_to_super[a] | node_to_super[b]
            layers.append((a, b, node_to_super[a], node_to_super[b]))

            # Update edges: remove edges between a,b; merge edges
            new_edges = {}
            for (u, v), ew in current_edges.items():
                if u == a and v == b:
                    continue
                if u == b and v == a:
                    continue
                nu = a if u == b else u
                nv = a if v == b else v
                if nu == nv:
                    continue
                key = (min(nu, nv), max(nu, nv))
                new_edges[key] = new_edges.get(key, 0) + ew

            current_edges = new_edges
            current_nodes.discard(b)
            node_to_super[a] = new_super
            if b in node_to_super:
                del node_to_super[b]

        remaining = {n: node_to_super[n] for n in current_nodes}
        return layers, remaining

    # --- Coarsen both graphs ---
    log_layers, log_remaining = coarsen_graph(logical_nodes, log_edges)
    phys_layers, phys_remaining = coarsen_graph(physical_nodes, phys_edges)

    # --- Align at coarsest level ---
    # Sort logical super-nodes by size (interaction count) descending
    log_supers = sorted(log_remaining.values(), key=lambda s: len(s), reverse=True)
    # Sort physical super-nodes by size descending (well-connected clusters are bigger)
    phys_supers = sorted(phys_remaining.values(), key=lambda s: len(s), reverse=True)

    # Initial coarse mapping: super_log -> super_phys
    # Map logical super-nodes to physical super-nodes
    coarse_mapping = {}  # frozenset(logical) -> frozenset(physical)
    used_phys_supers = set()
    for i, ls in enumerate(log_supers):
        if i < len(phys_supers):
            coarse_mapping[ls] = phys_supers[i]
            used_phys_supers.add(i)

    # --- Unfold physical layers to get physical hierarchy ---
    # Build a mapping from frozenset -> (child_a_frozenset, child_b_frozenset)
    phys_split = {}
    for (a, b, super_a, super_b) in reversed(phys_layers):
        merged = super_a | super_b
        phys_split[merged] = (super_a, super_b)

    log_split = {}
    for (a, b, super_a, super_b) in reversed(log_layers):
        merged = super_a | super_b
        log_split[merged] = (super_a, super_b)

    # --- Recursive unfolding ---
    # For each coarse pair, recursively split both and assign
    final_mapping = {}  # logical_qubit -> physical_qubit

    def unfold(log_set, phys_set):
        if len(log_set) == 1 and len(phys_set) >= 1:
            lq = next(iter(log_set))
            # Pick the most central physical qubit from phys_set
            best_pq = min(phys_set, key=lambda p: sum(self.distance_matrix[p]) if p < len(self.distance_matrix) else float('inf'))
            final_mapping[lq] = best_pq
            return {best_pq}

        if len(log_set) == 0:
            return set()

        if len(phys_set) <= 1:
            # Assign all logical to the single physical (or handle degeneracy)
            if phys_set:
                pq = next(iter(phys_set))
                # Only first logical gets it
                for lq in log_set:
                    if lq not in final_mapping:
                        final_mapping[lq] = pq
                        return {pq}
            return set()

        # Try to split logical set
        log_child = log_split.get(log_set, None)
        phys_child = phys_split.get(phys_set, None)

        if log_child is None and phys_child is None:
            # Both are leaves or can't be split further — assign greedily
            sorted_log = sorted(log_set, key=lambda q: self.logical_activity.get(q, 0), reverse=True)
            sorted_phys = sorted(phys_set, key=lambda p: self.physical_centrality.get(p, 0), reverse=True)
            used = set()
            for i, lq in enumerate(sorted_log):
                if i < len(sorted_phys):
                    final_mapping[lq] = sorted_phys[i]
                    used.add(sorted_phys[i])
            return used

        if log_child is None:
            # Can't split logical but can split physical — just assign greedily
            sorted_log = sorted(log_set, key=lambda q: self.logical_activity.get(q, 0), reverse=True)
            sorted_phys = sorted(phys_set, key=lambda p: self.physical_centrality.get(p, 0), reverse=True)
            used = set()
            for i, lq in enumerate(sorted_log):
                if i < len(sorted_phys):
                    final_mapping[lq] = sorted_phys[i]
                    used.add(sorted_phys[i])
            return used

        log_a, log_b = log_child

        if phys_child is None:
            # Split physical set roughly in half by centrality
            sorted_phys = sorted(phys_set, key=lambda p: self.physical_centrality.get(p, 0), reverse=True)
            mid = len(sorted_phys) // 2
            phys_a_set = frozenset(sorted_phys[:mid]) if mid > 0 else frozenset(sorted_phys[:1])
            phys_b_set = frozenset(sorted_phys[mid:]) if mid < len(sorted_phys) else frozenset()
        else:
            phys_a_set, phys_b_set = phys_child

        # Decide which logical child maps to which physical child
        # Compute interaction weight between log_a members and within-set coupling
        # Heuristic: try both assignments, pick lower cost
        def assignment_cost(la, pa, lb, pb):
            cost = 0
            for lq in la:
                for lq2, w in self.qubit_interaction_graph.get(lq, {}).items():
                    if lq2 in lb:
                        # Cross-set interaction: estimate distance
                        min_d = float('inf')
                        for pp in pa:
                            for pp2 in pb:
                                if pp < len(self.distance_matrix) and pp2 < len(self.distance_matrix):
                                    d = self.distance_matrix[pp][pp2]
                                    if d < min_d:
                                        min_d = d
                        if min_d < float('inf'):
                            cost += w * min_d
            return cost

        cost1 = assignment_cost(log_a, phys_a_set, log_b, phys_b_set)
        cost2 = assignment_cost(log_a, phys_b_set, log_b, phys_a_set)

        if cost1 <= cost2:
            used = set()
            used |= unfold(log_a, phys_a_set)
            remaining_phys_b = phys_b_set - used
            used |= unfold(log_b, remaining_phys_b if remaining_phys_b else phys_b_set)
            return used
        else:
            used = set()
            used |= unfold(log_a, phys_b_set)
            remaining_phys_a = phys_a_set - used
            used |= unfold(log_b, remaining_phys_a if remaining_phys_a else phys_a_set)
            return used

    for log_super, phys_super in coarse_mapping.items():
        unfold(log_super, phys_super)

    # --- Build final mapping arrays with identity fallback ---
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    used_physical = set(final_mapping.values())
    mapped_logical = set(final_mapping.keys())
    available_physical = [p for p in range(num_q) if p not in used_physical]
    unmapped_logical = [l for l in range(num_q) if l not in mapped_logical]

    # Assign mapped qubits
    for lq, pq in final_mapping.items():
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    # Assign unmapped logical qubits to remaining physical qubits
    for lq, pq in zip(unmapped_logical, available_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)