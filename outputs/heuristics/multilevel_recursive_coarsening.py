def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---------- collect logical edges ----------
    logical_edges = defaultdict(float)  # (min,max) -> weight
    logical_nodes = set()
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig:
        for u, nbrs in qig.items():
            logical_nodes.add(u)
            for v, w in nbrs.items():
                if u == v or w <= 0:
                    continue
                a, b = (u, v) if u < v else (v, u)
                logical_edges[(a, b)] = max(logical_edges[(a, b)], float(w))
                logical_nodes.add(v)
    if not logical_edges:
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                u, v = qubits
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                logical_edges[(a, b)] += 1.0
                logical_nodes.add(u); logical_nodes.add(v)

    if not logical_nodes:
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # ---------- collect hardware edges with weights ----------
    hw_edges = {}
    centrality = getattr(self, "physical_centrality", {}) or {}
    for p, nbrs in self.backend.items():
        for q in nbrs:
            if p == q:
                continue
            a, b = (p, q) if p < q else (q, p)
            if (a, b) in hw_edges:
                continue
            cw = centrality.get(a, 0.0) + centrality.get(b, 0.0)
            deg = len(self.backend[a]) + len(self.backend[b])
            hw_edges[(a, b)] = 1.0 + cw + 0.01 * deg

    # ---------- generic heavy-edge matching coarsener ----------
    def coarsen(nodes, edges, target):
        # nodes: set of node ids; edges: dict (a,b)->weight
        levels = []  # list of dicts: super_id -> (childA, childB or None)
        cur_nodes = set(nodes)
        cur_edges = dict(edges)
        next_id = (max(cur_nodes) + 1) if cur_nodes else 0
        while len(cur_nodes) > target:
            sorted_e = sorted(cur_edges.items(), key=lambda kv: -kv[1])
            matched = set()
            level_map = {}  # new -> (a,b)
            new_edges = {}
            mapping_old_to_new = {}
            for (a, b), w in sorted_e:
                if a in matched or b in matched:
                    continue
                sid = next_id; next_id += 1
                level_map[sid] = (a, b)
                mapping_old_to_new[a] = sid
                mapping_old_to_new[b] = sid
                matched.add(a); matched.add(b)
            if not level_map:
                break
            # carry over unmatched nodes as singletons
            for n in cur_nodes:
                if n not in matched:
                    sid = next_id; next_id += 1
                    level_map[sid] = (n, None)
                    mapping_old_to_new[n] = sid
            # rebuild edges
            for (a, b), w in cur_edges.items():
                na = mapping_old_to_new[a]
                nb = mapping_old_to_new[b]
                if na == nb:
                    continue
                key = (na, nb) if na < nb else (nb, na)
                new_edges[key] = new_edges.get(key, 0.0) + w
            levels.append(level_map)
            cur_nodes = set(level_map.keys())
            cur_edges = new_edges
        return cur_nodes, cur_edges, levels

    COARSE = 6
    L_nodes, L_edges, L_levels = coarsen(logical_nodes, logical_edges, COARSE)
    H_target = max(len(L_nodes), COARSE)
    hw_nodes = set(range(N))
    H_nodes, H_edges, H_levels = coarsen(hw_nodes, hw_edges, H_target)

    L_list = list(L_nodes)
    H_list = list(H_nodes)
    K = max(len(L_list), len(H_list))
    # padding
    while len(L_list) < K:
        L_list.append(("PAD", len(L_list)))
    while len(H_list) < K:
        H_list.append(("PAD", len(H_list)))

    # ---------- coarse-level assignment via Hungarian (or brute) ----------
    # Build quick distance lookup between coarse hardware nodes by averaging
    # representatives (use any descendant leaf physical qubit).
    def descend_leaves(node, levels_idx, levels):
        # collapse a coarse node back to its set of leaf ids
        stack = [(node, levels_idx)]
        out = []
        while stack:
            n, idx = stack.pop()
            if idx < 0 or not isinstance(n, int) or n not in levels[idx]:
                out.append(n)
                continue
            a, b = levels[idx][n]
            stack.append((a, idx - 1))
            if b is not None:
                stack.append((b, idx - 1))
        return [x for x in out if isinstance(x, int)]

    H_top_idx = len(H_levels) - 1
    L_top_idx = len(L_levels) - 1

    H_leaves = {h: (descend_leaves(h, H_top_idx, H_levels) if isinstance(h, int) else [])
                for h in H_list}
    L_leaves = {l: (descend_leaves(l, L_top_idx, L_levels) if isinstance(l, int) else [])
                for l in L_list}

    # cost matrix: weighted distance between coarse-cluster representative centroids
    def cluster_dist(h1_leaves, h2_leaves):
        if not h1_leaves or not h2_leaves:
            return 0.0
        s = 0.0; c = 0
        for a in h1_leaves:
            for b in h2_leaves:
                s += self.distance_matrix[a][b]; c += 1
        return s / c if c else 0.0

    # logical "weight" between coarse logical clusters comes from L_edges
    cost = [[0.0] * K for _ in range(K)]
    big = 1e9
    for i, lnode in enumerate(L_list):
        for j, hnode in enumerate(H_list):
            if not isinstance(lnode, int) or not isinstance(hnode, int):
                cost[i][j] = big * 0.0  # padding rows/cols cost zero
                continue
            # sum over edges in L_edges incident to lnode of weight * dist(hnode, partner_h)
            c = 0.0
            for (a, b), w in L_edges.items():
                if a == lnode or b == lnode:
                    other = b if a == lnode else a
                    # we don't know `other`'s assignment yet, so use centrality penalty:
                    # prefer central hardware nodes for high-activity logicals
                    centr = sum(centrality.get(p, 0.0) for p in H_leaves.get(hnode, [])) \
                            / max(1, len(H_leaves.get(hnode, [])))
                    c += w * (1.0 - centr)
            cost[i][j] = c

    # Hungarian
    def hungarian(C):
        n = len(C)
        INF = float('inf')
        u = [0.0] * (n + 1); v = [0.0] * (n + 1)
        p = [0] * (n + 1); way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]; delta = INF; j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur; way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]; j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta; v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while j0 != 0:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
        ans = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                ans[p[j] - 1] = j - 1
        return ans

    assign = hungarian(cost)  # L_list[i] -> H_list[assign[i]]

    # ---------- uncoarsen ----------
    # current mapping: coarse logical node -> coarse hardware node
    cur_map = {}
    for i, lnode in enumerate(L_list):
        if isinstance(lnode, int):
            hnode = H_list[assign[i]]
            if isinstance(hnode, int):
                cur_map[lnode] = hnode

    used_phys = set()

    def split_logical(level_idx):
        nonlocal cur_map
        new_map = {}
        for lnode, hnode in cur_map.items():
            if level_idx >= 0 and lnode in L_levels[level_idx]:
                a, b = L_levels[level_idx][lnode]
                # split hardware too if possible
                h_children = None
                if level_idx < len(H_levels) and hnode in H_levels[level_idx]:
                    ha, hb = H_levels[level_idx][hnode]
                    h_children = (ha, hb)
                if b is None:
                    # singleton carry-over
                    new_map[a] = hnode if h_children is None else h_children[0]
                    if h_children is not None and h_children[1] is not None:
                        # leave hb available for matching at this level
                        # store as a "free" hardware node by adding into new_map's
                        # complement set later — simplest: drop hb (it'll be re-used
                        # as identity backfill)
                        pass
                else:
                    if h_children is None or h_children[1] is None:
                        # only one physical child; assign one logical, defer other
                        target = hnode if h_children is None else h_children[0]
                        new_map[a] = target
                        # b unassigned -> handled by backfill
                    else:
                        ha, hb = h_children
                        # 2x2 assignment by weighted distance
                        wa = sum(L_edges.get((min(a, x), max(a, x)), 0.0) for x in cur_map)
                        wb = sum(L_edges.get((min(b, x), max(b, x)), 0.0) for x in cur_map)
                        # pick mapping minimizing wa*dist(ha,*) heuristic via centrality
                        ca = centrality.get(ha, 0.0); cb = centrality.get(hb, 0.0)
                        if (wa - wb) * (ca - cb) >= 0:
                            new_map[a] = ha; new_map[b] = hb
                        else:
                            new_map[a] = hb; new_map[b] = ha
            else:
                new_map[lnode] = hnode
        cur_map = new_map

    for lvl in range(min(len(L_levels), len(H_levels)) - 1, -1, -1):
        split_logical(lvl)

    # ---------- commit assignments ----------
    for lnode, hnode in cur_map.items():
        if not isinstance(lnode, int) or not isinstance(hnode, int):
            continue
        if 0 <= lnode < N and 0 <= hnode < N and hnode not in used_phys and self.mapping_dict[lnode] == -1:
            self.mapping_dict[lnode] = hnode
            self.reverse_mapping_dict[hnode] = lnode
            used_phys.add(hnode)

    # ---------- structure-aware fallback for unassigned logicals ----------
    unassigned = [q for q in range(N) if self.mapping_dict[q] == -1]
    if unassigned:
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            sa_map, _ = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits)
            for lq in unassigned:
                pq = sa_map[lq] if lq < len(sa_map) else -1
                if pq != -1 and pq not in used_phys:
                    self.mapping_dict[lq] = pq
                    self.reverse_mapping_dict[pq] = lq
                    used_phys.add(pq)
        except Exception:
            pass

    # ---------- final identity backfill ----------
    free_phys = [p for p in range(N) if p not in used_phys]
    fi = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            if lq not in used_phys:
                self.mapping_dict[lq] = lq
                self.reverse_mapping_dict[lq] = lq
                used_phys.add(lq)
                if lq in free_phys:
                    free_phys.remove(lq)
            else:
                while fi < len(free_phys) and free_phys[fi] in used_phys:
                    fi += 1
                if fi < len(free_phys):
                    p = free_phys[fi]; fi += 1
                    self.mapping_dict[lq] = p
                    self.reverse_mapping_dict[p] = lq
                    used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)