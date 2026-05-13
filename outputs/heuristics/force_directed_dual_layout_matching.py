def init_mapping(self):
    import numpy as np
    N = self.num_qubits

    def fr_layout(num_nodes, edges_w, iters=50, seed=0):
        rng = np.random.default_rng(seed)
        if num_nodes <= 0:
            return np.zeros((0, 2))
        pos = rng.uniform(-1.0, 1.0, size=(num_nodes, 2))
        area = 1.0
        k = np.sqrt(area / max(num_nodes, 1))
        t = 0.1
        cooling = t / (iters + 1)
        eps = 1e-9
        for _ in range(iters):
            disp = np.zeros_like(pos)
            for i in range(num_nodes):
                delta = pos[i] - pos
                dist = np.linalg.norm(delta, axis=1)
                dist[i] = 1.0
                rep = (k * k) / (dist + eps)
                rep[i] = 0.0
                disp[i] += (delta.T * (rep / (dist + eps))).sum(axis=1)
            for (u, v, w) in edges_w:
                if u == v:
                    continue
                d = pos[u] - pos[v]
                dn = np.linalg.norm(d) + eps
                attr = (dn * dn) / k * w
                contrib = (d / dn) * attr
                disp[u] -= contrib
                disp[v] += contrib
            dnorm = np.linalg.norm(disp, axis=1) + eps
            limit = np.minimum(dnorm, t) / dnorm
            pos = pos + (disp.T * limit).T
            t = max(t - cooling, 1e-3)
        return pos

    def procrustes_align(A, B):
        if A.shape[0] == 0 or B.shape[0] == 0:
            return A
        ca = A.mean(axis=0)
        cb = B.mean(axis=0)
        A0 = A - ca
        B0 = B - cb
        sa = np.linalg.norm(A0) + 1e-12
        sb = np.linalg.norm(B0) + 1e-12
        A0 /= sa
        B0 /= sb
        H = A0.T @ B0
        try:
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
        except np.linalg.LinAlgError:
            R = np.eye(2)
        aligned = A0 @ R
        return aligned * sb + cb

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    try:
        logical_edges = []
        seen = set()
        active_logicals = set()
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if a >= N or b >= N or a < 0 or b < 0:
                    continue
                active_logicals.add(a)
                active_logicals.add(b)
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                w = float(self.qubit_interaction_graph[a].get(b, 1)) if hasattr(self, "qubit_interaction_graph") else 1.0
                if w <= 0:
                    w = 1.0
                logical_edges.append((a, b, w))

        phys_edges = []
        phys_seen = set()
        for (u, v) in self.backend_connections:
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in phys_seen:
                continue
            phys_seen.add(key)
            phys_edges.append((u, v, 1.0))

        log_pos = fr_layout(N, logical_edges, iters=50, seed=1)
        phys_pos = fr_layout(N, phys_edges, iters=50, seed=2)
        log_aligned = procrustes_align(log_pos, phys_pos)

        diff = log_aligned[:, None, :] - phys_pos[None, :, :]
        cost = np.sqrt((diff * diff).sum(axis=2))

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        used_phys = set()
        for r, c in zip(row_ind, col_ind):
            self.mapping_dict[r] = int(c)
            self.reverse_mapping_dict[int(c)] = int(r)
            used_phys.add(int(c))

        if len(used_phys) != N or any(p < 0 for p in self.mapping_dict):
            raise RuntimeError("incomplete assignment")

    except Exception:
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            md, rmd = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits
            )
            self.mapping_dict = list(md)
            self.reverse_mapping_dict = list(rmd)
        except Exception:
            self.mapping_dict = list(range(N))
            self.reverse_mapping_dict = list(range(N))

        if len(set(self.mapping_dict)) != N:
            used = set()
            fixed = [-1] * N
            free = []
            for L in range(N):
                p = self.mapping_dict[L]
                if 0 <= p < N and p not in used:
                    fixed[L] = p
                    used.add(p)
            for p in range(N):
                if p not in used:
                    free.append(p)
            for L in range(N):
                if fixed[L] == -1:
                    fixed[L] = free.pop()
            self.mapping_dict = fixed
            self.reverse_mapping_dict = [-1] * N
            for L, p in enumerate(self.mapping_dict):
                self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)