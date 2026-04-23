def init_mapping(self):
    import networkx as nx

    n = self.num_qubits

    # Build logical interaction graph from 2-qubit gates
    logical_graph = nx.Graph()
    logical_qubits = set()
    if self.access2q is not None:
        for gate, qubits in self.access2q.items():
            if len(qubits) == 2:
                q1, q2 = qubits
                logical_graph.add_edge(q1, q2)
                logical_qubits.add(q1)
                logical_qubits.add(q2)

    # Build physical coupling graph from backend adjacency list
    physical_graph = nx.Graph()
    for node, neighbors in self.backend.items():
        for nb in neighbors:
            physical_graph.add_edge(node, nb)

    # Compute betweenness centrality
    logical_centrality = nx.betweenness_centrality(logical_graph)
    physical_centrality = nx.betweenness_centrality(physical_graph)

    # Sort logical qubits by centrality descending
    sorted_logical = sorted(logical_qubits, key=lambda q: logical_centrality.get(q, 0.0), reverse=True)

    # Sort all physical qubits by centrality descending
    all_physical = list(range(n))
    sorted_physical = sorted(all_physical, key=lambda q: physical_centrality.get(q, 0.0), reverse=True)

    # Assign k-th most central logical to k-th most central physical
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    used_physical = set()
    mapped_logical = set()

    for i, lq in enumerate(sorted_logical):
        pq = sorted_physical[i]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        mapped_logical.add(lq)

    # Identity fallback for unmapped logical qubits
    remaining_physical = [pq for pq in sorted_physical if pq not in used_physical]
    remaining_logical = [lq for lq in range(n) if lq not in mapped_logical]

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)