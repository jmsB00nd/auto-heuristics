def init_mapping(self):
    import networkx as nx
    import numpy as np
    from collections import defaultdict, deque
    from scipy.optimize import linear_sum_assignment

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. Logical interaction graph --------------------------------------
    edge_w = defaultdict(int)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                continue
            logical_qubits.add(a); logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            edge_w[key] += 1

    G_L = nx.Graph()
    G_L.add_nodes_from(logical_qubits)
    for (u, v), w in edge_w.items():
        G_L.add_edge(u, v, weight=w)

    # --- 2. Louvain communities + densities --------------------------------
    communities = []
    if G_L.number_of_nodes() > 0:
        try:
            communities = list(
                nx.community.louvain_communities(G_L, weight="weight", seed=0)
            )
        except Exception:
            try:
                communities = list(
                    nx.algorithms.community.greedy_modularity_communities(
                        G_L, weight="weight"
                    )
                )
            except Exception:
                communities = [{n} for n in G_L.nodes()]
    communities = [set(c) for c in communities if len(c) > 0]
    communities.sort(key=lambda c: -len(c))

    def density(graph, nodes):
        nodes = list(nodes)
        k = len(nodes)
        if k <= 1:
            return 0.0
        sub = graph.subgraph(nodes)
        max_e = k * (k - 1) / 2.0
        return sub.number_of_edges() / max_e if max_e > 0 else 0.0

    comm_density = [density(G_L, c) for c in communities]

    # --- 3. Physical regions: BFS-balls around high-degree seeds -----------
    G_P = nx.Graph()
    G_P.add_nodes_from(range(N))
    seen_pp = set()
    for (p, q) in self.backend_connections:
        if p == q:
            continue
        key = (p, q) if p < q else (q, p)
        if key in seen_pp:
            continue
        seen_pp.add(key)
        G_P.add_edge(p, q)

    phys_degree = {p: G_P.degree(p) for p in G_P.nodes()}
    seed_order = sorted(range(N), key=lambda p: -phys_degree.get(p, 0))

    used_phys = set()

    def grow_region(size):
        if size <= 0:
            return []
        seed = None
        for s in seed_order:
            if s not in used_phys:
                seed = s
                break
        if seed is None:
            return []
        region = []
        visited = {seed}
        dq = deque([seed])
        while dq and len(region) < size:
            x = dq.popleft()
            if x in used_phys:
                continue
            region.append(x)
            nbrs = sorted(G_P.neighbors(x), key=lambda y: -phys_degree.get(y, 0))
            for y in nbrs:
                if y not in visited and y not in used_phys:
                    visited.add(y)
                    dq.append(y)
        if len(region) < size:
            for p in seed_order:
                if len(region) >= size:
                    break
                if p not in used_phys and p not in region:
                    region.append(p)
        for p in region:
            used_phys.add(p)
        return region

    regions = [grow_region(len(c)) for c in communities]
    region_density = [density(G_P, r) for r in regions]

    # --- 4. Match communities to regions by density similarity -------------
    # (regions were already grown in community order; refine by best density match)
    K = len(communities)
    pair_cost = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            size_pen = abs(len(communities[i]) - len(regions[j]))
            dens_pen = abs(comm_density[i] - region_density[j])
            pair_cost[i, j] = size_pen * 10.0 + dens_pen
    if K > 0:
        row_ind, col_ind = linear_sum_assignment(pair_cost)
        assignment = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    else:
        assignment = {}

    # --- 5. Local Hungarian inside each (community, region) ----------------
    used_log = set()
    for ci, ri in assignment.items():
        comm = list(communities[ci])
        reg = list(regions[ri])
        if not comm or not reg:
            continue
        m = max(len(comm), len(reg))
        cost = np.zeros((m, m), dtype=float)
        sub_L = G_L.subgraph(comm)
        sub_P = G_P.subgraph(reg)
        log_deg = {q: sub_L.degree(q, weight="weight") for q in comm}
        phy_deg = {p: sub_P.degree(p) for p in reg}
        BIG = 1e6
        for a in range(m):
            for b in range(m):
                if a < len(comm) and b < len(reg):
                    cost[a, b] = -(log_deg[comm[a]] + 1) * (phy_deg[reg[b]] + 1)
                else:
                    cost[a, b] = BIG
        r_ind, c_ind = linear_sum_assignment(cost)
        for a, b in zip(r_ind, c_ind):
            if a < len(comm) and b < len(reg):
                lq = comm[a]
                pq = reg[b]
                if 0 <= lq < N and 0 <= pq < N and lq not in used_log and pq not in used_phys.union():
                    self.mapping_dict[lq] = pq
                    self.reverse_mapping_dict[pq] = lq
                    used_log.add(lq)

    # --- 6. Fallback for remaining logical / physical qubits ---------------
    placed_phys = {p for p in self.mapping_dict if p != -1}
    free_phys = [p for p in range(N) if p not in placed_phys]
    free_phys_iter = iter(free_phys)

    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            try:
                pq = next(free_phys_iter)
            except StopIteration:
                pq = lq
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    for pq in range(N):
        lq = self.reverse_mapping_dict[pq]
        if lq == -1:
            for cand in range(N):
                if self.mapping_dict[cand] == pq:
                    self.reverse_mapping_dict[pq] = cand
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)