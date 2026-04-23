def init_mapping(self):
    import networkx as nx
    import numpy as np

    num_q = self.num_qubits

    # Build logical interaction graph (only qubits with 2-qubit gates)
    lg = nx.Graph()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, w in neighbors.items():
            if q1 < q2:
                lg.add_edge(q1, q2, weight=w)

    # Build physical coupling graph
    pg = nx.Graph()
    for u, v in self.backend_connections:
        pg.add_edge(u, v)

    logical_nodes = sorted(lg.nodes())
    physical_nodes = sorted(pg.nodes())

    # Compute Fiedler vectors; fall back to trivial if graph is too small or disconnected
    def safe_fiedler(graph, nodes):
        if len(nodes) < 2:
            return {nodes[0]: 0.0} if nodes else {}
        subgraph = graph.subgraph(nodes)
        if not nx.is_connected(subgraph):
            # Compute per-component Fiedler vectors, offset by component index
            components = list(nx.connected_components(subgraph))
            fiedler_map = {}
            for idx, comp in enumerate(sorted(components, key=lambda c: min(c))):
                comp_nodes = sorted(comp)
                if len(comp_nodes) == 1:
                    fiedler_map[comp_nodes[0]] = float(idx) * 1e6
                else:
                    comp_sub = subgraph.subgraph(comp_nodes)
                    try:
                        fv = nx.fiedler_vector(comp_sub, weight='weight')
                    except Exception:
                        fv = np.linspace(0, 1, len(comp_nodes))
                    for i, n in enumerate(sorted(comp_sub.nodes())):
                        fiedler_map[n] = fv[i] + float(idx) * 1e6
            return fiedler_map
        try:
            fv = nx.fiedler_vector(subgraph, weight='weight')
        except Exception:
            fv = np.linspace(0, 1, len(nodes))
        return {n: fv[i] for i, n in enumerate(sorted(subgraph.nodes()))}

    logical_fiedler = safe_fiedler(lg, logical_nodes)
    physical_fiedler = safe_fiedler(pg, physical_nodes)

    # Sort by Fiedler coordinate
    sorted_logical = sorted(logical_fiedler.keys(), key=lambda q: logical_fiedler[q])
    sorted_physical = sorted(physical_fiedler.keys(), key=lambda q: physical_fiedler[q])

    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    used_physical = set()
    mapped_logical = set()

    # Rank-by-rank matching
    for i in range(min(len(sorted_logical), len(sorted_physical))):
        lq = sorted_logical[i]
        pq = sorted_physical[i]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        mapped_logical.add(lq)

    # Fallback: assign remaining logical qubits to remaining physical qubits
    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    remaining_logical = [l for l in range(num_q) if l not in mapped_logical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)