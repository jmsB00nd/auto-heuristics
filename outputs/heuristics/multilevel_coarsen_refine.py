def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    # ---------- build QIG ----------
    qig = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    try:
        for u, nbrs in self.qubit_interaction_graph.items():
            for v, w in nbrs.items():
                if u != v and w > 0:
                    qig[u][v] = float(w)
                    logical_qubits.add(u); logical_qubits.add(v)
    except Exception:
        pass
    if not logical_qubits:
        for _gid, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if a != b:
                    qig[a][b] += 1.0
                    qig[b][a] += 1.0
                    logical_qubits.add(a); logical_qubits.add(b)

    if not logical_qubits:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # ---------- helpers ----------
    def heavy_edge_match(nodes, adj):
        weight_sum = {n: sum(adj[n].values()) for n in nodes}
        order = sorted(nodes, key=lambda n: -weight_sum[n])
        matched = set()
        clusters = []
        for u in order:
            if u in matched:
                continue
            best_v, best_w = None, -1.0
            for v, w in adj[u].items():
                if v in matched or v == u or v not in weight_sum:
                    continue
                if w > best_w:
                    best_w, best_v = w, v
            if best_v is not None:
                clusters.append((u, best_v))
                matched.add(u); matched.add(best_v)
            else:
                clusters.append((u,))
                matched.add(u)
        return clusters

    def contract(adj, clusters):
        node2cluster = {}
        for ci, cl in enumerate(clusters):
            for n in cl:
                node2cluster[n] = ci
        new_adj = defaultdict(lambda: defaultdict(float))
        for u in list(adj.keys()):
            cu = node2cluster.get(u)
            if cu is None:
                continue
            for v, w in adj[u].items():
                cv = node2cluster.get(v)
                if cv is None or cu == cv:
                    continue
                new_adj[cu][cv] += w
        for ci in range(len(clusters)):
            _ = new_adj[ci]
        return new_adj

    # ---------- coarsen QIG ----------
    qig_node_lists = [list(logical_qubits)]
    qig_adjs = [qig]
    qig_clusters_hist = []
    for _ in range(10):
        nodes = qig_node_lists[-1]
        adj = qig_adjs[-1]
        if len(nodes) <= 2:
            break
        clusters = heavy_edge_match(nodes, adj)
        if len(clusters) == len(nodes):
            break
        qig_clusters_hist.append(clusters)
        qig_adjs.append(contract(adj, clusters))
        qig_node_lists.append(list(range(len(clusters))))

    # ---------- coarsen hardware ----------
    hw_adj = defaultdict(lambda: defaultdict(float))
    for p in range(N):
        try:
            nbrs = self.backend.get(p, ())
        except Exception:
            nbrs = ()
        for q in nbrs:
            if p != q:
                hw_adj[p][q] = 1.0
    hw_node_lists = [list(range(N))]
    hw_adjs = [hw_adj]
    hw_clusters_hist = []
    hw_cluster_to_phys = [{i: [i] for i in range(N)}]
    for _ in range(len(qig_node_lists) - 1):
        nodes = hw_node_lists[-1]
        adj = hw_adjs[-1]
        if len(nodes) <= 2:
            break
        clusters = heavy_edge_match(nodes, adj)
        if len(clusters) == len(nodes):
            break
        hw_clusters_hist.append(clusters)
        hw_adjs.append(contract(adj, clusters))
        hw_node_lists.append(list(range(len(clusters))))
        prev = hw_cluster_to_phys[-1]
        new_c2p = {}
        for ci, cl in enumerate(clusters):
            members = []
            for old_id in cl:
                members.extend(prev[old_id])
            new_c2p[ci] = members
        hw_cluster_to_phys.append(new_c2p)

    qig_depth = len(qig_node_lists) - 1
    hw_depth = len(hw_node_lists) - 1

    # ---------- coarsest embedding ----------
    def supernode_logical_weight(lvl, node):
        return sum(qig_adjs[lvl][node].values())

    def supernode_hw_score(lvl, node):
        return sum(self.physical_centrality.get(p, 0.0)
                   for p in hw_cluster_to_phys[lvl][node])

    sorted_l = sorted(qig_node_lists[qig_depth],
                      key=lambda n: -supernode_logical_weight(qig_depth, n))
    sorted_p = sorted(hw_node_lists[hw_depth],
                      key=lambda n: -supernode_hw_score(hw_depth, n))
    assign = {}
    used = set()
    pi = 0
    for l in sorted_l:
        while pi < len(sorted_p) and sorted_p[pi] in used:
            pi += 1
        if pi >= len(sorted_p):
            break
        assign[l] = sorted_p[pi]
        used.add(sorted_p[pi])
        pi += 1

    # ---------- uncoarsen with local refinement ----------
    cur_q = qig_depth
    cur_h = hw_depth
    while cur_q > 0 or cur_h > 0:
        unq = cur_q > 0
        unh = cur_h > 0
        below_q = cur_q - 1 if unq else cur_q
        below_h = cur_h - 1 if unh else cur_h
        new_assign = {}
        used_below = set()
        for coarse_l, coarse_p in assign.items():
            child_l = list(qig_clusters_hist[cur_q - 1][coarse_l]) if unq else [coarse_l]
            child_p = list(hw_clusters_hist[cur_h - 1][coarse_p]) if unh else [coarse_p]
            child_l_sorted = sorted(child_l,
                                    key=lambda x: -sum(qig_adjs[below_q][x].values()))
            child_p_sorted = sorted(child_p,
                                    key=lambda c: -sum(self.physical_centrality.get(p, 0.0)
                                                       for p in hw_cluster_to_phys[below_h][c]))
            pp = 0
            for cl in child_l_sorted:
                while pp < len(child_p_sorted) and child_p_sorted[pp] in used_below:
                    pp += 1
                if pp >= len(child_p_sorted):
                    break
                new_assign[cl] = child_p_sorted[pp]
                used_below.add(child_p_sorted[pp])
                pp += 1
        assign = new_assign
        cur_q = below_q
        cur_h = below_h

        # refinement: boundary swaps with strongest QIG neighbors
        adj_here = qig_adjs[cur_q]
        edges_here = []
        for u in adj_here:
            for v, w in adj_here[u].items():
                if u < v:
                    edges_here.append((u, v, w))
        rep = {}
        for sp in set(assign.values()):
            mem = hw_cluster_to_phys[cur_h].get(sp, [])
            if mem:
                rep[sp] = max(mem, key=lambda p: self.physical_centrality.get(p, 0.0))
            else:
                rep[sp] = sp

        def ecost(a, b, w):
            pa = assign.get(a); pb = assign.get(b)
            if pa is None or pb is None:
                return 0.0
            ra = rep.get(pa, pa); rb = rep.get(pb, pb)
            if not (0 <= ra < N and 0 <= rb < N):
                return 0.0
            return w * self.distance_matrix[ra][rb]

        for _it in range(2):
            improved = False
            for u in list(assign.keys()):
                pu = assign[u]
                neigh = sorted(adj_here[u].items(), key=lambda kv: -kv[1])[:5]
                for v, _w in neigh:
                    if v not in assign:
                        continue
                    pv = assign[v]
                    if pu == pv:
                        continue
                    old = sum(ecost(a, b, w) for (a, b, w) in edges_here
                              if a == u or b == u or a == v or b == v)
                    assign[u], assign[v] = pv, pu
                    new = sum(ecost(a, b, w) for (a, b, w) in edges_here
                              if a == u or b == u or a == v or b == v)
                    if new + 1e-9 < old:
                        improved = True
                        pu = pv
                    else:
                        assign[u], assign[v] = pu, pv
            if not improved:
                break

    # ---------- project to lists ----------
    mapping = [None] * N
    reverse = [None] * N
    used_p = set()
    for l, p in assign.items():
        if 0 <= l < N and 0 <= p < N and p not in used_p and mapping[l] is None:
            mapping[l] = p
            reverse[p] = l
            used_p.add(p)

    unmapped_l = [l for l in range(N) if mapping[l] is None]
    unused_p = [p for p in range(N) if p not in used_p]
    try:
        unmapped_l.sort(key=lambda l: -self.logical_activity.get(l, 0))
    except Exception:
        pass
    unused_p.sort(key=lambda p: -self.physical_centrality.get(p, 0.0))
    for l, p in zip(unmapped_l, unused_p):
        mapping[l] = p
        reverse[p] = l
        used_p.add(p)

    # absolute fallback (identity-style fill for any leftovers)
    remaining = [p for p in range(N) if p not in used_p]
    ri = 0
    for l in range(N):
        if mapping[l] is None and ri < len(remaining):
            mapping[l] = remaining[ri]
            reverse[remaining[ri]] = l
            ri += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)