def init_mapping(self):
    from collections import defaultdict, deque
    import heapq

    N = self.num_qubits

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. Build local logical interaction graph from self.access ---
    logical_adj = defaultdict(lambda: defaultdict(float))
    logicals_seen = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a][b] += 1.0
            logical_adj[b][a] += 1.0
            logicals_seen.add(a)
            logicals_seen.add(b)

    # Fold in prebuilt QIG weights if available (more accurate weights)
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig is not None:
        for u in list(qig.keys()):
            for v, w in qig[u].items():
                if u != v and w:
                    logical_adj[u][v] = max(logical_adj[u][v], float(w))
                    logicals_seen.add(u)
                    logicals_seen.add(v)

    activity = getattr(self, "logical_activity", None)

    def logical_weight(q):
        if activity is not None and q in activity:
            return float(activity[q])
        return float(sum(logical_adj[q].values()))

    # --- 2. Greedy heavy walk to extract the heavy backbone (spine) ---
    spine = []
    if logicals_seen:
        # Seed: heaviest edge endpoint, else highest activity
        best_edge = None
        best_w = -1.0
        for u in logical_adj:
            for v, w in logical_adj[u].items():
                if w > best_w:
                    best_w = w
                    best_edge = (u, v)
        if best_edge is None:
            seed = max(logicals_seen, key=logical_weight)
            spine = [seed]
        else:
            u, v = best_edge
            if logical_weight(u) >= logical_weight(v):
                spine = [u, v]
            else:
                spine = [v, u]

        visited_logical = set(spine)

        # Extend forward from spine[-1]
        while True:
            tail = spine[-1]
            best_nb, best_nb_w = None, -1.0
            for nb, w in logical_adj[tail].items():
                if nb in visited_logical:
                    continue
                if w > best_nb_w:
                    best_nb_w = w
                    best_nb = nb
            if best_nb is None:
                break
            spine.append(best_nb)
            visited_logical.add(best_nb)

        # Extend backward from spine[0]
        while True:
            head = spine[0]
            best_nb, best_nb_w = None, -1.0
            for nb, w in logical_adj[head].items():
                if nb in visited_logical:
                    continue
                if w > best_nb_w:
                    best_nb_w = w
                    best_nb = nb
            if best_nb is None:
                break
            spine.insert(0, best_nb)
            visited_logical.add(best_nb)

    # --- 3. Find a long geodesic in the hardware coupling graph (double BFS) ---
    backend = self.backend

    def bfs_farthest(src):
        dist = {src: 0}
        parent = {src: None}
        order = [src]
        dq = deque([src])
        while dq:
            x = dq.popleft()
            for y in backend.get(x, ()):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    parent[y] = x
                    order.append(y)
                    dq.append(y)
        far = max(dist, key=lambda k: dist[k]) if dist else src
        return far, parent, dist

    physical_path = []
    if N > 0:
        # Pick a starting node: any node with at least one neighbor, else 0
        start = 0
        for p in range(N):
            if backend.get(p):
                start = p
                break
        a, _, _ = bfs_farthest(start)
        b, parent_b, _ = bfs_farthest(a)
        # Reconstruct path a -> ... -> b
        path = []
        cur = b
        while cur is not None:
            path.append(cur)
            cur = parent_b[cur]
        physical_path = list(reversed(path))  # from a to b

    used_physical = set()
    placed_logical = set()

    # --- 4. Embed spine onto the geodesic ---
    spine_len = len(spine)
    geo_len = len(physical_path)
    take = min(spine_len, geo_len)
    for i in range(take):
        lq = spine[i]
        pq = physical_path[i]
        if 0 <= lq < N and pq not in used_physical and lq not in placed_logical:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
            placed_logical.add(lq)

    # If spine is longer than the geodesic, extend by hopping to nearest free neighbor
    if spine_len > geo_len and geo_len > 0:
        for i in range(geo_len, spine_len):
            lq = spine[i]
            if lq in placed_logical or not (0 <= lq < N):
                continue
            anchor_lq = spine[i - 1]
            anchor_pq = self.mapping_dict[anchor_lq]
            # BFS from anchor_pq to find nearest free physical
            dq = deque([anchor_pq])
            seen_b = {anchor_pq}
            found = None
            while dq:
                x = dq.popleft()
                if x not in used_physical:
                    found = x
                    break
                for y in backend.get(x, ()):
                    if y not in seen_b:
                        seen_b.add(y)
                        dq.append(y)
            if found is not None:
                self.mapping_dict[lq] = found
                self.reverse_mapping_dict[found] = lq
                used_physical.add(found)
                placed_logical.add(lq)

    # --- 5. Attach side branches: rank remaining logicals by total weight to spine ---
    centrality = getattr(self, "physical_centrality", {}) or {}

    remaining_logicals = [q for q in logicals_seen if q not in placed_logical and 0 <= q < N]

    # Score each remaining logical by its summed weight to already-placed logicals
    def attach_score(q):
        s = 0.0
        for nb, w in logical_adj[q].items():
            if nb in placed_logical:
                s += w
        return s

    remaining_logicals.sort(key=lambda q: (-attach_score(q), -logical_weight(q)))

    for lq in remaining_logicals:
        if lq in placed_logical:
            continue
        # Find strongest placed partner
        best_partner, best_pw = None, -1.0
        for nb, w in logical_adj[lq].items():
            if nb in placed_logical and w > best_pw:
                best_pw = w
                best_partner = nb
        if best_partner is None:
            continue  # handled in back-fill
        anchor_pq = self.mapping_dict[best_partner]
        # BFS to nearest free physical (prefer one-hop neighbors)
        dq = deque([(anchor_pq, 0)])
        seen_b = {anchor_pq}
        chosen = None
        best_depth = None
        candidates = []
        while dq:
            x, d = dq.popleft()
            if x != anchor_pq and x not in used_physical:
                candidates.append((d, -centrality.get(x, 0.0), x))
                if best_depth is None:
                    best_depth = d
                elif d > best_depth:
                    break
            for y in backend.get(x, ()):
                if y not in seen_b:
                    seen_b.add(y)
                    dq.append((y, d + 1))
        if candidates:
            candidates.sort()
            chosen = candidates[0][2]
        if chosen is not None:
            self.mapping_dict[lq] = chosen
            self.reverse_mapping_dict[chosen] = lq
            used_physical.add(chosen)
            placed_logical.add(lq)

    # --- 6. Back-fill all unplaced logicals onto remaining physicals by centrality ---
    free_physicals = [p for p in range(N) if p not in used_physical]
    free_physicals.sort(key=lambda p: -centrality.get(p, 0.0))

    unplaced = [lq for lq in range(N) if self.mapping_dict[lq] == -1]
    # Place logicals seen in circuit first (higher activity), then idle ones
    unplaced.sort(key=lambda lq: (-(1 if lq in logicals_seen else 0), -logical_weight(lq), lq))

    fi = 0
    for lq in unplaced:
        if fi >= len(free_physicals):
            break
        pq = free_physicals[fi]
        fi += 1
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        placed_logical.add(lq)

    # Final safety: identity fallback for any still-unassigned slot
    if any(x == -1 for x in self.mapping_dict):
        free_physicals = [p for p in range(N) if p not in used_physical]
        for lq in range(N):
            if self.mapping_dict[lq] == -1:
                if free_physicals:
                    pq = free_physicals.pop(0)
                else:
                    pq = lq
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_physical.add(pq)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)