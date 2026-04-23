def init_mapping(self):
    import networkx as nx
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    from collections import defaultdict

    n = self.num_qubits

    # Step 1: Build logical interaction graph (only qubits with 2-qubit gates)
    G_logical = nx.Graph()
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q1 < q2:
                G_logical.add_edge(q1, q2, weight=w)

    logical_qubits_with_interactions = set(G_logical.nodes())

    # Step 2: Greedy clique cover
    cliques = []
    remaining = G_logical.copy()
    while remaining.number_of_nodes() > 0:
        # Find a maximal clique greedily (largest first)
        all_cliques = list(nx.find_cliques(remaining))
        if not all_cliques:
            # Isolated nodes
            for node in list(remaining.nodes()):
                cliques.append([node])
            break
        # Pick the largest clique; break ties by total internal weight
        best_clique = max(all_cliques, key=lambda c: (
            len(c),
            sum(remaining[u][v].get('weight', 1)
                for i, u in enumerate(c) for v in c[i+1:]
                if remaining.has_edge(u, v))
        ))
        cliques.append(list(best_clique))
        remaining.remove_nodes_from(best_clique)

    # Step 3: Compute inter-clique weights for placement ordering
    clique_index = {}
    for idx, clique in enumerate(cliques):
        for q in clique:
            clique_index[q] = idx

    num_cliques = len(cliques)
    inter_clique_weight = defaultdict(float)
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q1 < q2:
                c1 = clique_index.get(q1, -1)
                c2 = clique_index.get(q2, -1)
                if c1 != c2 and c1 >= 0 and c2 >= 0:
                    key = (min(c1, c2), max(c1, c2))
                    inter_clique_weight[key] += w

    # Step 4: Build physical backend graph
    G_physical = nx.Graph()
    for u, v in self.backend_connections:
        G_physical.add_edge(u, v)

    physical_nodes = sorted(G_physical.nodes())
    num_physical = len(self.distance_matrix)

    # Helper: find a tight physical subgraph of given size seeded at a node
    def find_tight_subgraph(seed, size, used_physical):
        if size == 0:
            return []
        subgraph = [seed]
        candidates = set()
        for nb in G_physical.neighbors(seed):
            if nb not in used_physical:
                candidates.add(nb)
        while len(subgraph) < size and candidates:
            # Pick candidate minimizing total distance to current subgraph
            best_c = min(candidates, key=lambda c: sum(
                self.distance_matrix[c][s] for s in subgraph))
            subgraph.append(best_c)
            candidates.discard(best_c)
            for nb in G_physical.neighbors(best_c):
                if nb not in used_physical and nb not in subgraph:
                    candidates.add(nb)
        # If not enough candidates from neighbors, grab closest remaining
        if len(subgraph) < size:
            remaining_phys = [p for p in range(num_physical) if p not in used_physical and p not in subgraph]
            remaining_phys.sort(key=lambda p: sum(
                self.distance_matrix[p][s] for s in subgraph))
            subgraph.extend(remaining_phys[:size - len(subgraph)])
        return subgraph

    # Step 5: Place cliques in order of total interaction (heaviest first)
    clique_total_weight = []
    for idx, clique in enumerate(cliques):
        tw = sum(inter_clique_weight.get((min(idx, j), max(idx, j)), 0)
                 for j in range(num_cliques) if j != idx)
        # Also add internal weight
        tw += sum(self.qubit_interaction_graph.get(u, {}).get(v, 0)
                  for i, u in enumerate(clique) for v in clique[i+1:])
        clique_total_weight.append((tw, idx))
    clique_total_weight.sort(reverse=True)

    used_physical = set()
    clique_physical_subgraphs = [None] * num_cliques
    clique_placement_order = [idx for _, idx in clique_total_weight]

    for cidx in clique_placement_order:
        clique = cliques[cidx]
        size = len(clique)

        # Choose seed: closest to already-placed cliques with heavy inter-edges,
        # or most central physical node if first
        available = [p for p in range(num_physical) if p not in used_physical]
        if not available:
            break

        # Find placed cliques this one interacts with
        neighbor_physical = []
        for other_cidx in range(num_cliques):
            if clique_physical_subgraphs[other_cidx] is not None:
                key = (min(cidx, other_cidx), max(cidx, other_cidx))
                w = inter_clique_weight.get(key, 0)
                if w > 0:
                    for p in clique_physical_subgraphs[other_cidx]:
                        neighbor_physical.append((p, w))

        if neighbor_physical:
            # Pick seed that minimizes weighted distance to neighbor cliques' physical qubits
            seed = min(available, key=lambda p: sum(
                w * self.distance_matrix[p][np_] for np_, w in neighbor_physical))
        else:
            # Pick most central available node
            seed = min(available, key=lambda p: sum(
                self.distance_matrix[p][j] for j in range(num_physical)))

        subgraph = find_tight_subgraph(seed, size, used_physical)
        clique_physical_subgraphs[cidx] = subgraph
        used_physical.update(subgraph)

    # Step 6: Within each clique, use Hungarian to assign logical -> physical
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))
    assigned_logical = set()
    assigned_physical = set()

    for cidx in range(num_cliques):
        clique = cliques[cidx]
        phys_sub = clique_physical_subgraphs[cidx]
        if phys_sub is None:
            continue

        csize = len(clique)
        psize = len(phys_sub)
        dim = max(csize, psize)

        # Build cost matrix: for each logical-physical pair,
        # cost = sum of distances to where its interaction partners would go
        # Use a simpler metric: sum of dist * weight for all neighbors in clique
        cost = np.zeros((dim, dim), dtype=np.float64)
        for li, lq in enumerate(clique):
            for pi, pq in enumerate(phys_sub):
                c = 0.0
                for lj, lq2 in enumerate(clique):
                    if lq != lq2:
                        w = self.qubit_interaction_graph.get(lq, {}).get(lq2, 0)
                        if w > 0:
                            # Average distance to all candidate physical positions of lq2
                            min_d = min(self.distance_matrix[pq][phys_sub[pj]]
                                        for pj in range(psize))
                            c += w * self.distance_matrix[pq][phys_sub[lj]] if lj < psize else 0
                cost[li][pi] = c

        row_ind, col_ind = linear_sum_assignment(cost)

        for li, pi in zip(row_ind, col_ind):
            if li < csize and pi < psize:
                lq = clique[li]
                pq = phys_sub[pi]
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                assigned_logical.add(lq)
                assigned_physical.add(pq)

    # Step 7: Identity fallback for remaining qubits
    remaining_logical = [q for q in range(n) if q not in assigned_logical]
    remaining_physical = [p for p in range(n) if p not in assigned_physical]
    remaining_physical.sort()

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)