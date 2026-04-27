def init_mapping(self):
    import networkx as nx
    from networkx.algorithms.community import kernighan_lin_bisection
    import collections

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    # --- weighted logical interaction graph ---
    edge_w = collections.Counter()
    for gid, qs in self.access.items():
        if len(qs) == 2 and qs[0] != qs[1]:
            a, b = (qs[0], qs[1]) if qs[0] < qs[1] else (qs[1], qs[0])
            if 0 <= a < N and 0 <= b < N:
                edge_w[(a, b)] += 1

    G_L = nx.Graph()
    G_L.add_nodes_from(range(N))
    for (a, b), w in edge_w.items():
        G_L.add_edge(a, b, weight=w)

    # --- physical coupling graph ---
    G_P = nx.Graph()
    G_P.add_nodes_from(range(N))
    for u, v in self.backend_connections:
        if u != v and 0 <= u < N and 0 <= v < N:
            G_P.add_edge(u, v)

    def bisect(G, nodes, weighted):
        nodes = list(nodes)
        if len(nodes) <= 1:
            return set(nodes), set()
        sub = G.subgraph(nodes).copy()
        try:
            A, B = kernighan_lin_bisection(
                sub, max_iter=20, weight='weight' if weighted else None
            )
            return set(A), set(B)
        except Exception:
            h = len(nodes) // 2
            return set(nodes[:h]), set(nodes[h:])

    def internal_weight(G, nodes, weighted):
        s = 0
        for u, v, d in G.subgraph(nodes).edges(data=True):
            s += d.get('weight', 1) if weighted else 1
        return s

    def own_half_weight(l, half_set):
        w = 0
        for nb in G_L.neighbors(l):
            if nb in half_set and nb != l:
                w += G_L[l][nb].get('weight', 1)
        return w

    def recurse(L_set, P_set):
        if not L_set or not P_set:
            return
        if len(L_set) == 1:
            l = next(iter(L_set))
            p = next(iter(P_set))
            if self.mapping_dict[l] is None:
                self.mapping_dict[l] = p
            return

        L_A, L_B = bisect(G_L, L_set, weighted=True)
        P_A, P_B = bisect(G_P, P_set, weighted=False)

        wLA = internal_weight(G_L, L_A, True)
        wLB = internal_weight(G_L, L_B, True)
        wPA = internal_weight(G_P, P_A, False)
        wPB = internal_weight(G_P, P_B, False)

        if (wLA - wLB) * (wPA - wPB) >= 0:
            pairs = [[set(L_A), set(P_A)], [set(L_B), set(P_B)]]
        else:
            pairs = [[set(L_A), set(P_B)], [set(L_B), set(P_A)]]

        # rebalance so |L_part| == |P_part| in each pair
        guard = 0
        while guard < 4 * N:
            guard += 1
            d0 = len(pairs[0][0]) - len(pairs[0][1])
            d1 = len(pairs[1][0]) - len(pairs[1][1])
            if d0 > 0 and d1 < 0:
                src, dst = 0, 1
            elif d1 > 0 and d0 < 0:
                src, dst = 1, 0
            else:
                break
            best, best_w = None, None
            for l in pairs[src][0]:
                w = own_half_weight(l, pairs[src][0])
                if best is None or w < best_w:
                    best, best_w = l, w
            if best is None:
                break
            pairs[src][0].discard(best)
            pairs[dst][0].add(best)

        for L_part, P_part in pairs:
            recurse(L_part, P_part)

    recurse(set(range(N)), set(range(N)))

    # identity fallback for any unassigned logical
    used = set(p for p in self.mapping_dict if p is not None)
    free = [p for p in range(N) if p not in used]
    fi = 0
    for l in range(N):
        if self.mapping_dict[l] is None:
            self.mapping_dict[l] = free[fi]
            fi += 1

    for l in range(N):
        self.reverse_mapping_dict[self.mapping_dict[l]] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)