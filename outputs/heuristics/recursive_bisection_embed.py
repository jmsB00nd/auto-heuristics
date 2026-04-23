def init_mapping(self):
    import networkx as nx
    import numpy as np

    n = self.num_qubits

    # Build logical interaction graph (only qubits with 2-qubit gates)
    logical_graph = nx.Graph()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, weight in neighbors.items():
            if q1 < q2:
                logical_graph.add_edge(q1, q2, weight=weight)

    # Build physical coupling graph
    physical_graph = nx.Graph()
    for (p1, p2) in self.backend_connections:
        if not physical_graph.has_edge(p1, p2):
            physical_graph.add_edge(p1, p2)

    logical_nodes = sorted(logical_graph.nodes())
    physical_nodes = sorted(physical_graph.nodes())

    # Pad so both sets have the same size: use all physical qubits,
    # and add dummy logical qubits if needed
    all_physical = sorted(physical_graph.nodes()) if len(physical_graph.nodes()) > 0 else list(range(n))
    if len(all_physical) < n:
        all_physical = list(range(n))

    # We'll map active logical qubits first, then assign the rest
    assignment = {}

    def fiedler_partition(graph, nodes):
        """Partition nodes using the Fiedler vector of the subgraph."""
        if len(nodes) <= 1:
            return [nodes], []
        
        subgraph = graph.subgraph(nodes).copy()
        
        # Check connectivity - if disconnected, use components as partition
        if not nx.is_connected(subgraph):
            components = list(nx.connected_components(subgraph))
            # Merge components into two roughly equal groups
            components.sort(key=len, reverse=True)
            part_a, part_b = [], []
            size_a, size_b = 0, 0
            for comp in components:
                if size_a <= size_b:
                    part_a.extend(comp)
                    size_a += len(comp)
                else:
                    part_b.extend(comp)
                    size_b += len(comp)
            if len(part_b) == 0:
                mid = len(part_a) // 2
                return [sorted(part_a[:mid]), sorted(part_a[mid:])], []
            return [sorted(part_a), sorted(part_b)], []

        if len(nodes) == 2:
            return [[nodes[0]], [nodes[1]]], []

        try:
            fiedler_vec = nx.fiedler_vector(subgraph, weight='weight')
        except Exception:
            mid = len(nodes) // 2
            return [sorted(nodes[:mid]), sorted(nodes[mid:])], []

        node_list = sorted(subgraph.nodes())
        indexed = sorted(zip(fiedler_vec, node_list), key=lambda x: x[0])
        mid = len(indexed) // 2
        part_a = sorted([node for _, node in indexed[:mid]])
        part_b = sorted([node for _, node in indexed[mid:]])
        return [part_a, part_b], fiedler_vec

    def partition_density(graph, nodes):
        """Count edges within a set of nodes."""
        count = 0
        node_set = set(nodes)
        for u, v in graph.subgraph(nodes).edges():
            count += 1
        return count

    def recursive_bisection(log_nodes, phys_nodes, log_graph, phys_graph):
        if len(log_nodes) == 0:
            return
        if len(log_nodes) == 1:
            if len(phys_nodes) >= 1:
                assignment[log_nodes[0]] = phys_nodes[0]
            return
        if len(phys_nodes) == 1:
            assignment[log_nodes[0]] = phys_nodes[0]
            return

        log_parts, _ = fiedler_partition(log_graph, log_nodes)
        if len(log_parts) < 2:
            for i, lq in enumerate(log_nodes):
                if i < len(phys_nodes):
                    assignment[lq] = phys_nodes[i]
            return

        # Partition physical nodes to match sizes of logical partitions
        phys_parts, _ = fiedler_partition(phys_graph, phys_nodes)
        if len(phys_parts) < 2:
            for i, lq in enumerate(log_nodes):
                if i < len(phys_nodes):
                    assignment[lq] = phys_nodes[i]
            return

        # Match denser logical partition with denser physical partition
        log_density_0 = partition_density(log_graph, log_parts[0])
        log_density_1 = partition_density(log_graph, log_parts[1])
        phys_density_0 = partition_density(phys_graph, phys_parts[0])
        phys_density_1 = partition_density(phys_graph, phys_parts[1])

        # Determine which logical part is denser
        if log_density_0 >= log_density_1:
            log_dense, log_sparse = log_parts[0], log_parts[1]
        else:
            log_dense, log_sparse = log_parts[1], log_parts[0]

        if phys_density_0 >= phys_density_1:
            phys_dense, phys_sparse = phys_parts[0], phys_parts[1]
        else:
            phys_dense, phys_sparse = phys_parts[1], phys_parts[0]

        # Rebalance physical partitions to match logical partition sizes
        size_dense = len(log_dense)
        size_sparse = len(log_sparse)

        all_phys = phys_dense + phys_sparse
        # Re-sort by the Fiedler ordering of the physical graph to keep locality
        phys_sub = phys_graph.subgraph(all_phys).copy()
        if nx.is_connected(phys_sub) and len(all_phys) > 2:
            try:
                fv = nx.fiedler_vector(phys_sub, weight='weight')
                phys_sorted_nodes = sorted(phys_sub.nodes())
                indexed = sorted(zip(fv, phys_sorted_nodes), key=lambda x: x[0])
                ordered_phys = [node for _, node in indexed]
            except Exception:
                ordered_phys = all_phys
        else:
            ordered_phys = all_phys

        # Assign first size_dense physical qubits to dense match, rest to sparse
        # But we want to respect the density matching, so re-partition:
        # Use the original phys_dense/phys_sparse but adjust sizes
        phys_pool_dense = list(phys_dense)
        phys_pool_sparse = list(phys_sparse)

        final_phys_dense = []
        final_phys_sparse = []

        # Fill dense partition first from phys_dense, then overflow from phys_sparse
        for p in phys_pool_dense:
            if len(final_phys_dense) < size_dense:
                final_phys_dense.append(p)
            else:
                final_phys_sparse.append(p)
        for p in phys_pool_sparse:
            if len(final_phys_dense) < size_dense:
                final_phys_dense.append(p)
            else:
                final_phys_sparse.append(p)

        recursive_bisection(log_dense, final_phys_dense, log_graph, phys_graph)
        recursive_bisection(log_sparse, final_phys_sparse, log_graph, phys_graph)

    # Run recursive bisection on active logical qubits
    if len(logical_nodes) > 0 and len(physical_nodes) > 0:
        # Use only as many physical nodes as needed (at least len(logical_nodes))
        phys_to_use = physical_nodes[:max(len(logical_nodes), len(physical_nodes))]
        recursive_bisection(logical_nodes, phys_to_use, logical_graph, physical_graph)

    # Build full mapping with identity fallback
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    used_physical = set(assignment.values())
    assigned_logical = set(assignment.keys())
    free_physical = [p for p in range(n) if p not in used_physical]
    free_idx = 0

    # First, apply the recursive bisection assignments
    for lq, pq in assignment.items():
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    # Then, assign remaining logical qubits to free physical qubits
    for lq in range(n):
        if lq not in assigned_logical:
            self.mapping_dict[lq] = free_physical[free_idx]
            self.reverse_mapping_dict[free_physical[free_idx]] = lq
            free_idx += 1

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)