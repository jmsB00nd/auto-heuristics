def init_mapping(self):
    import math, random, heapq
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    # ---------- 1. Build weighted logical interaction graph ----------
    weight = defaultdict(float)        # (u,v) with u<v -> weight
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            weight[(u, v)] += 1.0
            logical_nodes.add(a); logical_nodes.add(b)

    logical_nodes = sorted(logical_nodes)
    if not logical_nodes:
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    adj = defaultdict(lambda: defaultdict(float))
    for (u, v), w in weight.items():
        adj[u][v] += w
        adj[v][u] += w
    deg = {n: sum(adj[n].values()) for n in logical_nodes}
    m2 = sum(deg.values())  # = 2m
    if m2 <= 0:
        m2 = 1.0

    # ---------- 2. Louvain (single-level, repeated passes) ----------
    community = {n: i for i, n in enumerate(logical_nodes)}
    def louvain_pass(nodes, adj_local, deg_local, m2_local):
        comm = {n: n for n in nodes}
        comm_deg = dict(deg_local)
        improved = True
        order = list(nodes)
        rng = random.Random(0xC0FFEE)
        passes = 0
        while improved and passes < 20:
            improved = False
            passes += 1
            rng.shuffle(order)
            for n in order:
                cur = comm[n]
                k_i = deg_local[n]
                # weights to each neighbor community
                w_to_c = defaultdict(float)
                for nb, w in adj_local[n].items():
                    w_to_c[comm[nb]] += w
                # remove n from its community
                comm_deg[cur] -= k_i
                w_to_cur = w_to_c.get(cur, 0.0)
                best_c = cur
                best_gain = 0.0
                for c, k_in in w_to_c.items():
                    gain = k_in - comm_deg[c] * k_i / m2_local
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_c = c
                # default-stay gain reference
                stay_gain = w_to_cur - comm_deg[cur] * k_i / m2_local
                if best_gain <= stay_gain + 1e-12:
                    best_c = cur
                comm[n] = best_c
                comm_deg[best_c] = comm_deg.get(best_c, 0.0) + k_i
                if best_c != cur:
                    improved = True
        return comm

    comm_map = louvain_pass(logical_nodes, adj, deg, m2)
    # normalize community ids
    cid_remap = {}
    communities = defaultdict(list)
    for n in logical_nodes:
        c = comm_map[n]
        if c not in cid_remap:
            cid_remap[c] = len(cid_remap)
        communities[cid_remap[c]].append(n)
    community_list = [communities[i] for i in range(len(communities))]
    K = len(community_list)

    # ---------- 3. Pick K well-separated physical seeds, grow balanced regions ----------
    K_eff = min(K, N)
    # farthest-point sampling using distance_matrix
    dm = self.distance_matrix
    seeds = [0]
    while len(seeds) < K_eff:
        best_node, best_d = -1, -1
        for p in range(N):
            if p in seeds:
                continue
            d_min = min(dm[p][s] for s in seeds)
            if d_min > best_d:
                best_d = d_min
                best_node = p
        if best_node < 0:
            break
        seeds.append(best_node)

    # target region sizes proportional to community sizes
    comm_sizes = [len(c) for c in community_list[:K_eff]]
    total_c = sum(comm_sizes) if sum(comm_sizes) > 0 else 1
    target = [max(1, int(round(s * N / total_c))) for s in comm_sizes]
    # adjust to sum to N
    diff = N - sum(target)
    i = 0
    while diff != 0 and target:
        if diff > 0:
            target[i % len(target)] += 1; diff -= 1
        else:
            if target[i % len(target)] > 1:
                target[i % len(target)] -= 1; diff += 1
        i += 1
        if i > 10 * N:
            break

    # multi-source balanced BFS
    region_of = [-1] * N
    queues = [deque([s]) for s in seeds]
    region_members = [[] for _ in range(K_eff)]
    for r, s in enumerate(seeds):
        region_of[s] = r
        region_members[r].append(s)
    remaining = N - K_eff
    active = True
    while remaining > 0 and active:
        active = False
        for r in range(K_eff):
            if len(region_members[r]) >= target[r]:
                continue
            grew = False
            while queues[r]:
                cur = queues[r].popleft()
                for nb in self.backend[cur]:
                    if region_of[nb] == -1:
                        region_of[nb] = r
                        region_members[r].append(nb)
                        queues[r].append(nb)
                        remaining -= 1
                        grew = True
                        active = True
                        break
                if grew:
                    break
        if not active and remaining > 0:
            # assign leftovers to nearest region
            for p in range(N):
                if region_of[p] == -1:
                    best_r, best_d = 0, float('inf')
                    for r in range(K_eff):
                        for q in region_members[r]:
                            if dm[p][q] < best_d:
                                best_d = dm[p][q]; best_r = r
                    region_of[p] = best_r
                    region_members[best_r].append(p)
                    remaining -= 1
            break

    # ---------- 4. Match communities to regions (Hungarian on size-difference cost) ----------
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((K_eff, K_eff), dtype=float)
        for i in range(K_eff):
            ci = len(community_list[i])
            for j in range(K_eff):
                rj = len(region_members[j])
                cost[i][j] = abs(ci - rj)
        row_ind, col_ind = linear_sum_assignment(cost)
        match = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    except Exception:
        # greedy fallback
        match = {}
        used_r = set()
        order = sorted(range(K_eff), key=lambda i: -len(community_list[i]))
        for ci in order:
            best_r, best_d = -1, float('inf')
            for r in range(K_eff):
                if r in used_r:
                    continue
                d = abs(len(community_list[ci]) - len(region_members[r]))
                if d < best_d:
                    best_d = d; best_r = r
            if best_r >= 0:
                match[ci] = best_r; used_r.add(best_r)

    # ---------- 5. Place qubits inside each matched pair ----------
    used_phys = set()

    def intra_weighted_degree(node, members_set):
        return sum(w for nb, w in adj[node].items() if nb in members_set)

    def intra_region_centrality(p, members_set):
        s = 0.0
        for q in members_set:
            if q == p: continue
            d = dm[p][q]
            if d > 0:
                s += 1.0 / d
        return s

    for ci, ri in match.items():
        comm_nodes = community_list[ci]
        comm_set = set(comm_nodes)
        region_nodes = list(region_members[ri])
        # sort logical by descending intra-community weighted degree
        sorted_log = sorted(comm_nodes,
                            key=lambda n: (-intra_weighted_degree(n, comm_set), n))
        region_set = set(region_nodes)
        sorted_phys = sorted(region_nodes,
                             key=lambda p: (-intra_region_centrality(p, region_set), p))
        for L, P in zip(sorted_log, sorted_phys):
            if L < N and self.mapping_dict[L] is None and P not in used_phys:
                self.mapping_dict[L] = P
                self.reverse_mapping_dict[P] = L
                used_phys.add(P)

    # ---------- 6. Fill remaining logical/physical qubits ----------
    free_phys = [p for p in range(N) if p not in used_phys]
    # logical qubits in access not yet placed
    unplaced_log = [L for L in logical_nodes if L < N and self.mapping_dict[L] is None]
    fp_iter = iter(free_phys)
    for L in unplaced_log:
        try:
            P = next(fp_iter)
        except StopIteration:
            break
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L
        used_phys.add(P)

    # remaining logical ids (not in access, but index < N) and any leftover phys
    free_phys = [p for p in range(N) if p not in used_phys]
    fp_iter = iter(free_phys)
    for L in range(N):
        if self.mapping_dict[L] is None:
            try:
                P = next(fp_iter)
            except StopIteration:
                # identity fallback
                for p in range(N):
                    if p not in used_phys:
                        P = p; break
                else:
                    P = L
            self.mapping_dict[L] = P
            self.reverse_mapping_dict[P] = L
            used_phys.add(P)

    # final safety: if any duplicates due to corner cases, repair via identity
    if len(set(self.mapping_dict)) != len(self.mapping_dict):
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)