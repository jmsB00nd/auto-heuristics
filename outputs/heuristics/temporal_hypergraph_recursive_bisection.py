def init_mapping(self):
    import networkx as nx
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = []
    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            interactions.append((int(qubits[0]), int(qubits[1])))

    M = len(interactions)

    if M == 0 or N == 0:
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    W_size = 4
    decay = 0.9
    pair_weight = defaultdict(lambda: defaultdict(float))
    num_windows = max(1, M - W_size + 1)
    for w_idx in range(num_windows):
        window = interactions[w_idx:w_idx + W_size]
        weight = decay ** w_idx
        qs = set()
        for a, b in window:
            qs.add(a); qs.add(b)
        qs_list = [q for q in qs if 0 <= q < N]
        if len(qs_list) < 2:
            continue
        for i in range(len(qs_list)):
            for j in range(i + 1, len(qs_list)):
                u, v = qs_list[i], qs_list[j]
                pair_weight[u][v] += weight
                pair_weight[v][u] += weight

    G_log = nx.Graph()
    G_log.add_nodes_from(range(N))
    for u, nbrs in pair_weight.items():
        for v, w in nbrs.items():
            if u < v:
                G_log.add_edge(u, v, weight=float(w))

    G_hw = nx.Graph()
    G_hw.add_nodes_from(range(N))
    for u, neighbors in self.backend.items():
        for v in neighbors:
            if u != v and 0 <= u < N and 0 <= v < N and u < v:
                G_hw.add_edge(int(u), int(v), weight=1.0)

    def bisect(graph, nodes):
        nodes = list(nodes)
        n = len(nodes)
        if n <= 1:
            return list(nodes), []
        if n == 2:
            return [nodes[0]], [nodes[1]]
        idx = {v: i for i, v in enumerate(nodes)}
        Wmat = np.zeros((n, n))
        for u in nodes:
            if u not in graph:
                continue
            for v in graph.neighbors(u):
                if v in idx and v != u:
                    Wmat[idx[u]][idx[v]] = graph[u][v].get('weight', 1.0)
        Dvec = Wmat.sum(axis=1)
        Lmat = np.diag(Dvec) - Wmat
        try:
            _, eigvecs = np.linalg.eigh(Lmat + 1e-9 * np.eye(n))
            fiedler = eigvecs[:, 1]
            order = np.argsort(fiedler)
            half = n // 2
            A = [nodes[i] for i in order[:half]]
            B = [nodes[i] for i in order[half:]]
        except Exception:
            half = n // 2
            A = nodes[:half]
            B = nodes[half:]
        return A, B

    def assign(log_nodes, phys_nodes):
        log_nodes = list(log_nodes)
        phys_nodes = list(phys_nodes)
        if not log_nodes or not phys_nodes:
            return
        if len(log_nodes) == 1:
            l = log_nodes[0]
            best_p = max(phys_nodes, key=lambda p: self.physical_centrality.get(p, 0.0))
            self.mapping_dict[l] = best_p
            self.reverse_mapping_dict[best_p] = l
            return
        if len(phys_nodes) == 1:
            l = log_nodes[0]
            p = phys_nodes[0]
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l
            return

        L1, L2 = bisect(G_log, log_nodes)
        P1, P2 = bisect(G_hw, phys_nodes)

        if not L1 and L2:
            L1 = [L2.pop()]
        if not L2 and L1:
            L2 = [L1.pop()]
        if not P1 and P2:
            P1 = [P2.pop()]
        if not P2 and P1:
            P2 = [P1.pop()]

        act1 = sum(self.logical_activity.get(l, 0) for l in L1)
        act2 = sum(self.logical_activity.get(l, 0) for l in L2)
        cen1 = float(np.mean([self.physical_centrality.get(p, 0.0) for p in P1])) if P1 else 0.0
        cen2 = float(np.mean([self.physical_centrality.get(p, 0.0) for p in P2])) if P2 else 0.0
        if (act1 - act2) * (cen1 - cen2) < 0:
            P1, P2 = P2, P1

        while len(L1) > len(P1) and len(P2) > len(L2):
            P1.append(P2.pop())
        while len(L2) > len(P2) and len(P1) > len(L1):
            P2.append(P1.pop())
        while len(L1) > len(P1) and P2:
            P1.append(P2.pop())
        while len(L2) > len(P2) and P1:
            P2.append(P1.pop())

        assign(L1, P1)
        assign(L2, P2)

    assign(list(range(N)), list(range(N)))

    used = set(p for p in self.mapping_dict if p != -1)
    free = [p for p in range(N) if p not in used]
    for q in range(N):
        if self.mapping_dict[q] == -1:
            if free:
                p = free.pop(0)
                self.mapping_dict[q] = p
                self.reverse_mapping_dict[p] = q

    if -1 in self.mapping_dict or len(set(self.mapping_dict)) != N:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)