def init_mapping(self):
    import numpy as np
    import math
    import random

    N = self.num_qubits
    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    try:
        active_logicals = set()
        interactions = []
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = int(qubits[0]), int(qubits[1])
                if a == b:
                    continue
                active_logicals.add(a)
                active_logicals.add(b)
                interactions.append((a, b))

        if not active_logicals:
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        active_logicals = sorted(active_logicals)
        max_logical = max(active_logicals)
        if max_logical >= N:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            md, rmd = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, N
            )
            self.mapping_dict = list(md)
            self.reverse_mapping_dict = list(rmd)
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        L = len(active_logicals)
        l_index = {q: i for i, q in enumerate(active_logicals)}

        W = np.zeros((L, L), dtype=float)
        for (a, b) in interactions:
            try:
                w = float(self.qubit_interaction_graph[a][b])
            except Exception:
                w = 1.0
            if w <= 0.0:
                w = 1.0
            i, j = l_index[a], l_index[b]
            if i != j:
                W[i, j] = w
                W[j, i] = w

        rng = random.Random(0xC0FFEE)
        pos = np.zeros((L, 2), dtype=float)
        for i in range(L):
            theta = 2.0 * math.pi * i / max(1, L)
            r = 1.0 + 0.01 * rng.random()
            pos[i, 0] = r * math.cos(theta)
            pos[i, 1] = r * math.sin(theta)

        max_w = W.max() if W.size > 0 else 1.0
        if max_w <= 0:
            max_w = 1.0
        W_norm = W / max_w

        area = float(L)
        k = math.sqrt(area / max(1, L)) if L > 0 else 1.0
        k_attr = k
        k_rep = k * k
        iterations = 80
        t0 = 0.1 * math.sqrt(area)
        eps = 1e-9

        for it in range(iterations):
            t = t0 * (1.0 - it / iterations)
            disp = np.zeros_like(pos)

            diff = pos[:, None, :] - pos[None, :, :]
            dist2 = np.sum(diff * diff, axis=2) + eps
            dist = np.sqrt(dist2)
            rep_mag = (k_rep / dist2)
            np.fill_diagonal(rep_mag, 0.0)
            rep_force = (diff / dist[:, :, None]) * rep_mag[:, :, None]
            disp += rep_force.sum(axis=1)

            attr_mag = (dist2 / k_attr) * W_norm
            attr_force = (diff / dist[:, :, None]) * attr_mag[:, :, None]
            disp -= attr_force.sum(axis=1)

            disp_norm = np.sqrt(np.sum(disp * disp, axis=1)) + eps
            scale = np.minimum(disp_norm, t) / disp_norm
            pos = pos + disp * scale[:, None]

            center = pos.mean(axis=0)
            pos = pos - center

        phys_coords = None
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for u in range(N):
                neigh = self.backend.get(u, set()) if hasattr(self.backend, "get") else self.backend[u]
                for v in neigh:
                    if u < v:
                        G.add_edge(u, v)
            if nx.is_connected(G) and N >= 3:
                layout = nx.spectral_layout(G, dim=2)
                phys_coords = np.array([layout[i] for i in range(N)], dtype=float)
            else:
                layout = nx.kamada_kawai_layout(G, dim=2)
                phys_coords = np.array([layout[i] for i in range(N)], dtype=float)
        except Exception:
            phys_coords = None

        if phys_coords is None:
            try:
                D = np.array(self.distance_matrix, dtype=float)
                D2 = D * D
                n = D.shape[0]
                J = np.eye(n) - np.ones((n, n)) / n
                B = -0.5 * J @ D2 @ J
                vals, vecs = np.linalg.eigh(B)
                idx = np.argsort(vals)[::-1][:2]
                phys_coords = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
            except Exception:
                phys_coords = np.zeros((N, 2), dtype=float)
                for p in range(N):
                    theta = 2.0 * math.pi * p / max(1, N)
                    phys_coords[p, 0] = math.cos(theta)
                    phys_coords[p, 1] = math.sin(theta)

        def _normalize(coords):
            mn = coords.min(axis=0)
            mx = coords.max(axis=0)
            rng_ = mx - mn
            rng_[rng_ < eps] = 1.0
            return (coords - mn) / rng_

        logical_coords = _normalize(pos)
        physical_coords = _normalize(phys_coords)

        size = N
        BIG = 1e6
        cost = np.full((size, size), BIG, dtype=float)
        for i in range(L):
            lc = logical_coords[i]
            d = physical_coords - lc[None, :]
            cost[i, :] = np.sqrt(np.sum(d * d, axis=1))

        idle_logicals = [q for q in range(N) if q not in l_index]
        try:
            cent = self.physical_centrality
        except Exception:
            cent = {}
        for row_i, lq in enumerate(idle_logicals, start=L):
            if row_i >= size:
                break
            for p in range(N):
                c = cent.get(p, 0.0) if isinstance(cent, dict) else 0.0
                cost[row_i, p] = BIG - c

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            row_ind = np.arange(size)
            col_ind = np.arange(size)

        mapping = [-1] * N
        reverse = [-1] * N
        used_phys = set()
        assigned_logicals = set()

        for r, c in zip(row_ind, col_ind):
            if r < L:
                lq = active_logicals[r]
            else:
                idx = r - L
                if idx < len(idle_logicals):
                    lq = idle_logicals[idx]
                else:
                    continue
            if 0 <= lq < N and 0 <= c < N and c not in used_phys and lq not in assigned_logicals:
                mapping[lq] = c
                reverse[c] = lq
                used_phys.add(c)
                assigned_logicals.add(lq)

        remaining_phys = [p for p in range(N) if p not in used_phys]
        rp_iter = iter(remaining_phys)
        for lq in range(N):
            if mapping[lq] == -1:
                try:
                    p = next(rp_iter)
                except StopIteration:
                    for p_candidate in range(N):
                        if p_candidate not in used_phys:
                            p = p_candidate
                            break
                    else:
                        p = lq
                mapping[lq] = p
                reverse[p] = lq
                used_phys.add(p)

        self.mapping_dict = mapping
        self.reverse_mapping_dict = reverse

    except Exception:
        self.mapping_dict = [i for i in range(N)]
        self.reverse_mapping_dict = [i for i in range(N)]

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)