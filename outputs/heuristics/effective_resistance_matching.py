def init_mapping(self):
    import numpy as np

    N = self.num_qubits

    # 1) Logical edge weights from access
    edge_weight = {}
    logical_seen = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits
            if a == b:
                continue
            logical_seen.add(a)
            logical_seen.add(b)
            key = (a, b) if a < b else (b, a)
            edge_weight[key] = edge_weight.get(key, 0) + 1

    # 2) Laplacian -> pseudoinverse -> effective-resistance matrix
    L = np.zeros((N, N), dtype=float)
    for u in range(N):
        neighbors = self.backend.get(u, set()) if hasattr(self.backend, "get") else set()
        deg = 0
        for v in neighbors:
            if 0 <= v < N and v != u:
                L[u, v] = -1.0
                deg += 1
        L[u, u] = deg

    R = np.full((N, N), np.inf, dtype=float)
    try:
        Lpinv = np.linalg.pinv(L)
        diag = np.diag(Lpinv)
        Rfull = diag[:, None] + diag[None, :] - 2.0 * Lpinv
        Rfull = np.maximum(Rfull, 0.0)
        for i in range(N):
            for j in range(N):
                if i == j:
                    R[i, j] = 0.0
                else:
                    d = self.distance_matrix[i][j]
                    if d and d > 0:
                        R[i, j] = float(Rfull[i, j])
    except Exception:
        for i in range(N):
            for j in range(N):
                if i == j:
                    R[i, j] = 0.0
                else:
                    d = self.distance_matrix[i][j]
                    if d and d > 0:
                        R[i, j] = float(d)

    # 3) Sort structures
    sorted_edges = sorted(edge_weight.items(), key=lambda kv: -kv[1])
    physical_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r = R[i, j]
            if r != np.inf:
                physical_pairs.append((r, i, j))
    physical_pairs.sort(key=lambda t: t[0])

    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_physical = set()

    def assign(logical, physical):
        if logical < 0 or logical >= N or physical < 0 or physical >= N:
            return False
        if mapping_dict[logical] != -1:
            return False
        if physical in used_physical:
            return False
        mapping_dict[logical] = physical
        reverse_mapping_dict[physical] = logical
        used_physical.add(physical)
        return True

    def best_free_neighbor_by_R(anchor):
        best_p, best_r = -1, np.inf
        for p in range(N):
            if p in used_physical:
                continue
            r = R[anchor, p]
            if r < best_r:
                best_r = r
                best_p = p
        return best_p

    # 4) Greedy resistance-matching of logical edges
    for (l1, l2), _w in sorted_edges:
        if l1 >= N or l2 >= N:
            continue
        m1 = mapping_dict[l1]
        m2 = mapping_dict[l2]
        if m1 != -1 and m2 != -1:
            continue
        if m1 != -1:
            p = best_free_neighbor_by_R(m1)
            if p != -1:
                assign(l2, p)
            continue
        if m2 != -1:
            p = best_free_neighbor_by_R(m2)
            if p != -1:
                assign(l1, p)
            continue
        # both unmapped: take the lowest-R fully-free pair
        chosen = None
        for r, i, j in physical_pairs:
            if i in used_physical or j in used_physical:
                continue
            chosen = (i, j)
            break
        if chosen is None:
            continue
        pi, pj = chosen
        la1 = self.logical_activity.get(l1, 0) if hasattr(self.logical_activity, "get") else 0
        la2 = self.logical_activity.get(l2, 0) if hasattr(self.logical_activity, "get") else 0
        ci = self.physical_centrality.get(pi, 0.0) if hasattr(self.physical_centrality, "get") else 0.0
        cj = self.physical_centrality.get(pj, 0.0) if hasattr(self.physical_centrality, "get") else 0.0
        if (la1 >= la2) == (ci >= cj):
            assign(l1, pi)
            assign(l2, pj)
        else:
            assign(l1, pj)
            assign(l2, pi)

    # 5) Place leftover interacting logicals on most-central free physicals
    leftover = [l for l in logical_seen if l < N and mapping_dict[l] == -1]
    leftover.sort(key=lambda l: -(self.logical_activity.get(l, 0) if hasattr(self.logical_activity, "get") else 0))
    for l in leftover:
        best_p, best_c = -1, -1.0
        for p in range(N):
            if p in used_physical:
                continue
            c = self.physical_centrality.get(p, 0.0) if hasattr(self.physical_centrality, "get") else 0.0
            if c > best_c:
                best_c = c
                best_p = p
        if best_p != -1:
            assign(l, best_p)

    # 6) Identity fallback for any remaining logical slot
    for l in range(N):
        if mapping_dict[l] != -1:
            continue
        if l not in used_physical:
            assign(l, l)
        else:
            for p in range(N):
                if p not in used_physical:
                    assign(l, p)
                    break

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)