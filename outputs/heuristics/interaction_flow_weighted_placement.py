def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    n = self.num_qubits

    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    if self.access2q is None:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_qubits = set()
    for gate, qubits in self.access2q.items():
        for q in qubits:
            logical_qubits.add(q)

    if not logical_qubits:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logical_list = sorted(logical_qubits)
    num_logical = len(logical_list)
    log_idx = {q: i for i, q in enumerate(logical_list)}

    interaction_weight = defaultdict(float)
    qubit_total_weight = defaultdict(float)
    for gate, qubits in self.access2q.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            interaction_weight[(q1, q2)] += 1.0
            interaction_weight[(q2, q1)] += 1.0
            qubit_total_weight[q1] += 1.0
            qubit_total_weight[q2] += 1.0

    num_physical = n
    phys_list = list(range(num_physical))
    dist = self.distance_matrix

    max_p = len(dist)

    centroids = {}
    for lq in logical_list:
        if qubit_total_weight[lq] == 0:
            centroids[lq] = None
            continue
        scores = np.zeros(max_p)
        for (qa, qb), w in interaction_weight.items():
            if qa == lq and qb < max_p:
                for p in range(max_p):
                    scores[p] += w * dist[qb][p] if qb < max_p and p < max_p else 0
        centroids[lq] = scores

    K = min(3, num_physical)
    candidates = {}
    for lq in logical_list:
        if centroids[lq] is None:
            valid_phys = [p for p in phys_list if p < max_p]
            candidates[lq] = valid_phys[:K] if valid_phys else list(range(K))
        else:
            centroid_dist = np.zeros(max_p)
            for p in range(max_p):
                total = 0.0
                tw = qubit_total_weight[lq]
                if tw > 0:
                    for (qa, qb), w in interaction_weight.items():
                        if qa == lq and qb < max_p:
                            total += w * dist[qb][p]
                    centroid_dist[p] = total / tw
                else:
                    centroid_dist[p] = 0
            ranked = np.argsort(centroid_dist)
            candidates[lq] = [int(ranked[i]) for i in range(min(K, len(ranked)))]

    # --- SSP min-cost flow on auxiliary graph ---
    # Nodes: 0=super_source, 1..num_logical=logical, num_logical+1..num_logical+num_physical=physical, last=super_sink
    S = 0
    T = num_logical + num_physical + 1
    total_nodes = T + 1

    graph = defaultdict(list)
    cap = {}
    cost_map = {}

    def add_edge(u, v, c, w):
        graph[u].append(v)
        graph[v].append(u)
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap[(v, u)] = cap.get((v, u), 0)
        cost_map[(u, v)] = w
        cost_map[(v, u)] = -w

    for i, lq in enumerate(logical_list):
        l_node = i + 1
        add_edge(S, l_node, 1, 0)

    for i, lq in enumerate(logical_list):
        l_node = i + 1
        for p in candidates[lq]:
            p_node = num_logical + 1 + p
            w_cost = 0.0
            for (qa, qb), w in interaction_weight.items():
                if qa == lq and qb in log_idx:
                    for pc in candidates[qb]:
                        if p < max_p and pc < max_p:
                            w_cost += w * dist[p][pc] / len(candidates[qb])
            add_edge(l_node, p_node, 1, w_cost)

    for p in range(num_physical):
        p_node = num_logical + 1 + p
        add_edge(p_node, T, 1, 0)

    flow = {}
    for key in cap:
        flow[key] = 0

    assigned_logical = set()
    assigned_physical = set()
    assignment = {}

    for _ in range(num_logical):
        INF = float('inf')
        dist_bf = [INF] * total_nodes
        dist_bf[S] = 0
        parent = [-1] * total_nodes
        in_queue = [False] * total_nodes

        queue = [S]
        in_queue[S] = True

        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for v in graph[u]:
                residual = cap.get((u, v), 0) - flow.get((u, v), 0)
                if residual > 0 and dist_bf[u] + cost_map.get((u, v), 0) < dist_bf[v]:
                    dist_bf[v] = dist_bf[u] + cost_map.get((u, v), 0)
                    parent[v] = u
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if dist_bf[T] == INF:
            break

        v = T
        while v != S:
            u = parent[v]
            flow[(u, v)] = flow.get((u, v), 0) + 1
            flow[(v, u)] = flow.get((v, u), 0) - 1
            v = u

    for i, lq in enumerate(logical_list):
        l_node = i + 1
        for p in candidates[lq]:
            p_node = num_logical + 1 + p
            if flow.get((l_node, p_node), 0) > 0:
                assignment[lq] = p
                assigned_logical.add(lq)
                assigned_physical.add(p)
                break

    remaining_physical = [p for p in phys_list if p not in assigned_physical]
    remaining_idx = 0
    for lq in logical_list:
        if lq not in assigned_logical:
            if remaining_idx < len(remaining_physical):
                assignment[lq] = remaining_physical[remaining_idx]
                assigned_physical.add(remaining_physical[remaining_idx])
                remaining_idx += 1

    used_physical = set(assignment.values())
    remaining_physical_all = [p for p in range(n) if p not in used_physical]
    rp_idx = 0

    for q in range(n):
        if q in assignment:
            self.mapping_dict[q] = assignment[q]
        else:
            if rp_idx < len(remaining_physical_all):
                self.mapping_dict[q] = remaining_physical_all[rp_idx]
                rp_idx += 1

    for q in range(n):
        self.reverse_mapping_dict[self.mapping_dict[q]] = q

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)