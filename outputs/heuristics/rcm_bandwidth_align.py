def init_mapping(self):
    import networkx as nx

    num_q = self.num_qubits

    # Build logical interaction graph (only qubits involved in 2-qubit gates)
    logical_graph = nx.Graph()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, weight in neighbors.items():
            if q1 < q2:
                logical_graph.add_edge(q1, q2, weight=weight)

    # Build physical coupling graph
    physical_graph = nx.Graph()
    for u, v in self.backend_connections:
        physical_graph.add_edge(u, v)

    # RCM ordering on logical interaction graph
    if logical_graph.number_of_nodes() > 0:
        logical_rcm = list(nx.utils.reverse_cuthill_mckee_ordering(logical_graph))
    else:
        logical_rcm = []

    # RCM ordering on physical coupling graph
    if physical_graph.number_of_nodes() > 0:
        physical_rcm = list(nx.utils.reverse_cuthill_mckee_ordering(physical_graph))
    else:
        physical_rcm = list(range(num_q))

    # Align by position: logical_rcm[i] -> physical_rcm[i]
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    used_physical = set()
    mapped_logical = set()

    for i in range(min(len(logical_rcm), len(physical_rcm))):
        lq = logical_rcm[i]
        pq = physical_rcm[i]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        mapped_logical.add(lq)

    # Assign remaining unmapped logical qubits to unused physical qubits
    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    remaining_logical = [l for l in range(num_q) if l not in mapped_logical]

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)