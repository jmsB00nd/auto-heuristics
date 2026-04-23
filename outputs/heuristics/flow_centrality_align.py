def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    n = self.num_qubits

    # --- Logical interaction graph ---
    lg = nx.Graph()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if lg.has_edge(q1, q2):
                lg[q1][q2]['weight'] += 1
            else:
                lg.add_edge(q1, q2, weight=1)

    # Interaction-flow centrality: betweenness on the logical graph
    # Use inverse weight so heavily-interacting pairs are "close"
    if lg.number_of_edges() > 0:
        for u, v, d in lg.edges(data=True):
            d['inv_weight'] = 1.0 / d['weight']
        logical_centrality = nx.betweenness_centrality(lg, weight='inv_weight')
    else:
        logical_centrality = {}

    # Collect all logical qubits that participate in 2-qubit gates
    active_logical = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            active_logical.update(qubits)

    # Sort active logical qubits by descending flow centrality
    sorted_logical = sorted(active_logical, key=lambda q: logical_centrality.get(q, 0.0), reverse=True)

    # --- Physical coupling graph ---
    pg = nx.Graph()
    for (u, v) in self.backend_connections:
        pg.add_edge(u, v)
    # Ensure all physical qubits are nodes
    for i in range(n):
        if i not in pg:
            pg.add_node(i)

    physical_centrality = nx.betweenness_centrality(pg)

    # Sort physical qubits by descending betweenness centrality
    sorted_physical = sorted(range(n), key=lambda q: physical_centrality.get(q, 0.0), reverse=True)

    # --- Align the two ranked sequences ---
    self.mapping_dict = list(range(n))  # start with identity
    self.reverse_mapping_dict = list(range(n))

    used_physical = set()

    for i, lq in enumerate(sorted_logical):
        if i < len(sorted_physical):
            pq = sorted_physical[i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)

    # Assign remaining logical qubits to remaining physical qubits
    remaining_physical = [pq for pq in sorted_physical if pq not in used_physical]
    remaining_logical = [q for q in range(n) if q not in active_logical]
    remaining_logical.sort()

    for i, lq in enumerate(remaining_logical):
        if i < len(remaining_physical):
            pq = remaining_physical[i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)