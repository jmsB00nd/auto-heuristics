def init_mapping(self):
    import networkx as nx
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    n = self.num_qubits

    # Build physical coupling graph
    G_phys = nx.Graph()
    G_phys.add_nodes_from(range(n))
    for (u, v) in self.backend_connections:
        G_phys.add_edge(u, v)

    # Find biconnected components (sets of edges)
    bicomp_edges = list(nx.biconnected_component_edges(G_phys))
    if not bicomp_edges:
        # No edges at all — fallback to trivial
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Convert edge-sets to node-sets for each biconnected component
    bicomp_nodes = []
    for edges in bicomp_edges:
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        bicomp_nodes.append(nodes)

    # Sort components largest-first
    bicomp_nodes.sort(key=lambda s: len(s), reverse=True)

    # Collect logical qubits that participate in 2-qubit gates
    logical_qubits = set()
    for q1, neighbors in self.qubit_interaction_graph.items():
        if neighbors:
            logical_qubits.add(q1)
            for q2 in neighbors:
                logical_qubits.add(q2)
    logical_qubits = sorted(logical_qubits)

    # Build weighted logical interaction graph
    G_log = nx.Graph()
    G_log.add_nodes_from(logical_qubits)
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q1 < q2:
                G_log.add_edge(q1, q2, weight=w)

    # Partition logical qubits into groups matching physical component sizes
    # Use recursive bisection based on Fiedler vector (spectral partitioning)
    def spectral_partition(graph, nodes, sizes):
        """Partition nodes into groups of given sizes using spectral bisection."""
        nodes = list(nodes)
        if len(sizes) == 1:
            return [nodes]

        # Split sizes into two roughly equal halves
        total = sum(sizes)
        cumsum = 0
        split_idx = 0
        for i, s in enumerate(sizes):
            cumsum += s
            if cumsum >= total / 2:
                split_idx = i + 1
                break
        if split_idx == 0:
            split_idx = 1
        if split_idx >= len(sizes):
            split_idx = len(sizes) - 1

        left_sizes = sizes[:split_idx]
        right_sizes = sizes[split_idx:]
        left_count = sum(left_sizes)

        if len(nodes) <= 1 or left_count == 0 or left_count >= len(nodes):
            # Can't split further, just divide sequentially
            result = []
            idx = 0
            for s in sizes:
                take = min(s, len(nodes) - idx)
                result.append(nodes[idx:idx + take])
                idx += take
            return result

        # Build subgraph and compute Fiedler vector for spectral ordering
        subg = graph.subgraph(nodes)
        try:
            if nx.is_connected(subg) and len(nodes) > 2:
                fiedler = nx.fiedler_vector(subg, weight='weight')
                order = sorted(range(len(nodes)), key=lambda i: fiedler[i])
            else:
                # If disconnected, order by degree in interaction graph
                order = sorted(range(len(nodes)),
                               key=lambda i: graph.degree(nodes[i], weight='weight'),
                               reverse=True)
        except Exception:
            order = list(range(len(nodes)))

        ordered_nodes = [nodes[i] for i in order]
        left_nodes = ordered_nodes[:left_count]
        right_nodes = ordered_nodes[left_count:]

        left_groups = spectral_partition(graph, left_nodes, left_sizes)
        right_groups = spectral_partition(graph, right_nodes, right_sizes)
        return left_groups + right_groups

    # Determine target sizes for logical groups — match physical component sizes
    # but cap at the number of available logical qubits
    comp_sizes = [len(c) for c in bicomp_nodes]
    remaining_logical = len(logical_qubits)
    target_sizes = []
    for s in comp_sizes:
        take = min(s, remaining_logical)
        target_sizes.append(take)
        remaining_logical -= take
        if remaining_logical <= 0:
            break

    # Partition logical qubits
    if sum(target_sizes) > 0 and logical_qubits:
        logical_groups = spectral_partition(G_log, logical_qubits, target_sizes)
    else:
        logical_groups = [logical_qubits] if logical_qubits else []

    # Now assign within each (logical_group, physical_component) pair
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))
    used_physical = set()
    mapped_logical = set()

    for group_idx, lgroup in enumerate(logical_groups):
        if not lgroup:
            continue
        if group_idx >= len(bicomp_nodes):
            break

        pcomp = sorted(bicomp_nodes[group_idx] - used_physical)
        if not pcomp:
            continue

        lgroup = list(lgroup)
        nl = len(lgroup)
        np_ = len(pcomp)
        dim = max(nl, np_)

        # Build cost matrix: interaction-weighted distance
        cost = [[0.0] * dim for _ in range(dim)]
        for i in range(nl):
            lq = lgroup[i]
            for j in range(np_):
                pq = pcomp[j]
                # Cost: sum of (weight * distance to where partner would ideally go)
                # Approximate: weighted sum of distances to all interacting partners'
                # best physical locations within this component
                c = 0.0
                for lq2, w in self.qubit_interaction_graph.get(lq, {}).items():
                    # Use distance from this physical qubit to all others in component
                    # weighted by interaction strength
                    min_dist = float('inf')
                    for k in range(np_):
                        if k != j:
                            d = self.distance_matrix[pq][pcomp[k]]
                            if d < min_dist:
                                min_dist = d
                    if min_dist == float('inf'):
                        min_dist = 0
                    c += w * min_dist
                cost[i][j] = c

        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            if i < nl and j < np_:
                lq = lgroup[i]
                pq = pcomp[j]
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_physical.add(pq)
                mapped_logical.add(lq)

    # Map any remaining unmapped logical qubits to unused physical qubits
    all_logical = set(range(n))
    unmapped_logical = sorted(all_logical - mapped_logical)
    unused_physical = sorted(set(range(n)) - used_physical)

    for lq, pq in zip(unmapped_logical, unused_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)