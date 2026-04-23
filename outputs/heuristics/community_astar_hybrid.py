def init_mapping(self):
    import numpy as np
    import networkx as nx
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    n = self.num_qubits

    interaction = defaultdict(float)
    logical_qubits = set()
    two_q_gates = []

    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            q1, q2 = qubits
            interaction[(min(q1, q2), max(q1, q2))] += 1.0
            logical_qubits.update(qubits)
            two_q_gates.append((gate_id, q1, q2))

    if not interaction:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    G_int = nx.Graph()
    for q in logical_qubits:
        G_int.add_node(q)
    for (q1, q2), w in interaction.items():
        G_int.add_edge(q1, q2, weight=w)

    try:
        import leidenalg
        import igraph as ig
        nodes = sorted(G_int.nodes())
        nmap = {v: i for i, v in enumerate(nodes)}
        g = ig.Graph(n=len(nodes),
                     edges=[(nmap[u], nmap[v]) for u, v in G_int.edges()],
                     directed=False)
        g.es['weight'] = [G_int[u][v]['weight'] for u, v in G_int.edges()]
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight')
        communities = [set(nodes[i] for i in comm) for comm in part]
    except ImportError:
        communities = list(nx.community.louvain_communities(G_int, weight='weight', seed=42))

    communities = [c for c in communities if c]
    if not communities:
        communities = [logical_qubits]
    num_comm = len(communities)

    G_hw = nx.Graph()
    for u, v in self.backend_connections:
        G_hw.add_edge(u, v)
    hw_nodes = sorted(G_hw.nodes())

    def spectral_bisect(nodes_set, graph):
        nodes_list = sorted(nodes_set)
        if len(nodes_list) <= 1:
            return [set(nodes_list)]
        sg = graph.subgraph(nodes_list)
        if not nx.is_connected(sg):
            return [set(c) for c in nx.connected_components(sg)]
        try:
            fv = nx.fiedler_vector(sg, seed=42)
        except Exception:
            mid = len(nodes_list) // 2
            return [set(nodes_list[:mid]), set(nodes_list[mid:])]
        order = sorted(zip(fv, nodes_list))
        mid = len(order) // 2
        return [set(x[1] for x in order[:mid]), set(x[1] for x in order[mid:])]

    def partition_k(nodes_set, graph, k):
        if k <= 1 or len(nodes_set) <= 1:
            return [set(nodes_set)]
        parts = spectral_bisect(nodes_set, graph)
        if len(parts) < 2:
            nl = sorted(nodes_set)
            mid = len(nl) // 2
            parts = [set(nl[:mid]), set(nl[mid:])]
        total = sum(len(p) for p in parts)
        k1 = max(1, round(k * len(parts[0]) / total))
        k2 = max(1, k - k1)
        if k1 + k2 > k:
            k2 = k - k1
        result = partition_k(parts[0], graph, k1) + partition_k(parts[1], graph, k2)
        return result

    regions = partition_k(set(hw_nodes), G_hw, num_comm)

    while len(regions) < num_comm:
        largest = max(regions, key=len)
        regions.remove(largest)
        regions.extend(spectral_bisect(largest, G_hw))
    while len(regions) > num_comm:
        regions.sort(key=len)
        regions = [regions[0] | regions[1]] + regions[2:]

    def medoid(region):
        rl = sorted(region)
        if len(rl) == 1:
            return rl[0]
        return min(rl, key=lambda u: sum(self.distance_matrix[u][v] for v in rl))

    centroids = [medoid(r) for r in regions]

    comm_of = {}
    for ci, comm in enumerate(communities):
        for q in comm:
            comm_of[q] = ci

    inter_w = np.zeros((num_comm, num_comm))
    for (q1, q2), w in interaction.items():
        c1, c2 = comm_of.get(q1, -1), comm_of.get(q2, -1)
        if c1 >= 0 and c2 >= 0 and c1 != c2:
            inter_w[c1][c2] += w
            inter_w[c2][c1] += w

    cost = np.zeros((num_comm, num_comm))
    for ci in range(num_comm):
        for ri in range(num_comm):
            c = 0.0
            for cj in range(num_comm):
                if cj != ci:
                    avg_dist = np.mean([self.distance_matrix[centroids[ri]][centroids[rj]]
                                        for rj in range(num_comm) if rj != ri]) if num_comm > 1 else 0.0
                    c += inter_w[ci][cj] * avg_dist
            c += abs(len(communities[ci]) - len(regions[ri])) * 5.0
            cost[ci][ri] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    c2r = {int(ci): int(ri) for ci, ri in zip(row_ind, col_ind)}

    qubit_layer_count = defaultdict(int)
    early_gates = []
    for gid, q1, q2 in two_q_gates:
        layer = max(qubit_layer_count[q1], qubit_layer_count[q2])
        if layer < 3:
            early_gates.append((gid, q1, q2))
            qubit_layer_count[q1] = layer + 1
            qubit_layer_count[q2] = layer + 1

    mapping = [-1] * n
    used_phys = set()

    for ci in range(num_comm):
        ri = c2r[ci]
        lqs = sorted(communities[ci])
        available = sorted(set(regions[ri]) - used_phys)

        if not available:
            continue

        comm_set = set(lqs)
        rel_gates = [(g, q1, q2) for g, q1, q2 in early_gates
                     if q1 in comm_set or q2 in comm_set]

        if len(lqs) <= 8 and len(available) <= 12:
            best_cost = [float('inf')]
            best_assign = [None]

            def _search(idx, assign, cost_so_far, avail, _lqs=lqs, _rel=rel_gates,
                        _dm=self.distance_matrix, _bc=best_cost, _ba=best_assign):
                if cost_so_far >= _bc[0]:
                    return
                if idx == len(_lqs):
                    _bc[0] = cost_so_far
                    _ba[0] = dict(assign)
                    return
                lq = _lqs[idx]
                for pq in sorted(avail):
                    added = 0.0
                    for _, gq1, gq2 in _rel:
                        if gq1 == lq and gq2 in assign:
                            added += _dm[pq][assign[gq2]]
                        elif gq2 == lq and gq1 in assign:
                            added += _dm[pq][assign[gq1]]
                    if cost_so_far + added < _bc[0]:
                        assign[lq] = pq
                        _search(idx + 1, assign, cost_so_far + added, avail - {pq})
                        del assign[lq]

            _search(0, {}, 0.0, set(available))

            if best_assign[0]:
                for lq, pq in best_assign[0].items():
                    mapping[lq] = pq
                    used_phys.add(pq)
            else:
                for i, lq in enumerate(lqs):
                    if i < len(available):
                        mapping[lq] = available[i]
                        used_phys.add(available[i])
        else:
            freq = defaultdict(float)
            for _, q1, q2 in rel_gates:
                if q1 in comm_set:
                    freq[q1] += 1
                if q2 in comm_set:
                    freq[q2] += 1
            sorted_lqs = sorted(lqs, key=lambda q: -freq.get(q, 0))
            avail_set = set(available)
            placed = {}

            for lq in sorted_lqs:
                if not avail_set:
                    break
                if not placed:
                    pq = min(avail_set, key=lambda p: sum(
                        self.distance_matrix[p][x] for x in avail_set))
                else:
                    pq = min(avail_set, key=lambda p: sum(
                        interaction.get((min(lq, nq), max(lq, nq)), 0) *
                        self.distance_matrix[p][ppq]
                        for nq, ppq in placed.items()
                    ))
                mapping[lq] = pq
                placed[lq] = pq
                used_phys.add(pq)
                avail_set.discard(pq)

    all_phys = set(range(n))
    remaining_phys = sorted(all_phys - used_phys)
    unmapped = [q for q in range(n) if mapping[q] == -1]
    for i, q in enumerate(unmapped):
        mapping[q] = remaining_phys[i]

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * n
    for lq in range(n):
        self.reverse_mapping_dict[mapping[lq]] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)