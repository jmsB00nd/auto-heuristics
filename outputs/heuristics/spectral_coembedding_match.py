def init_mapping(self):
    import numpy as np
    import math
    N = self.num_qubits

    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    try:
        # ---- 1. Collect logical qubits and interactions ----
        active_logicals = set()
        edge_weights = {}
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = int(qubits[0]), int(qubits[1])
                if a == b:
                    continue
                if a >= N or b >= N or a < 0 or b < 0:
                    continue
                active_logicals.add(a)
                active_logicals.add(b)
                key = (a, b) if a < b else (b, a)
                w = 1.0
                qig = getattr(self, "qubit_interaction_graph", None)
                if qig is not None:
                    try:
                        w = float(qig[a][b])
                        if w <= 0:
                            w = 1.0
                    except Exception:
                        w = 1.0
                edge_weights[key] = edge_weights.get(key, 0.0) + w

        active_logicals = sorted(active_logicals)
        n_log = len(active_logicals)

        if n_log == 0:
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        log_index = {q: i for i, q in enumerate(active_logicals)}

        # ---- 2. Build logical normalized Laplacian ----
        A_log = np.zeros((n_log, n_log), dtype=float)
        for (a, b), w in edge_weights.items():
            i, j = log_index[a], log_index[b]
            A_log[i, j] += w
            A_log[j, i] += w
        d_log = A_log.sum(axis=1)
        d_log_safe = np.where(d_log > 0, d_log, 1.0)
        D_inv_sqrt_log = 1.0 / np.sqrt(d_log_safe)
        L_log = np.eye(n_log) - (A_log * D_inv_sqrt_log[:, None]) * D_inv_sqrt_log[None, :]
        L_log = (L_log + L_log.T) * 0.5

        # ---- 3. Build hardware normalized Laplacian ----
        A_phys = np.zeros((N, N), dtype=float)
        for u, neighbors in self.backend.items():
            for v in neighbors:
                if 0 <= u < N and 0 <= v < N and u != v:
                    A_phys[u, v] = 1.0
                    A_phys[v, u] = 1.0
        d_phys = A_phys.sum(axis=1)
        d_phys_safe = np.where(d_phys > 0, d_phys, 1.0)
        D_inv_sqrt_phys = 1.0 / np.sqrt(d_phys_safe)
        L_phys = np.eye(N) - (A_phys * D_inv_sqrt_phys[:, None]) * D_inv_sqrt_phys[None, :]
        L_phys = (L_phys + L_phys.T) * 0.5

        # ---- 4. Eigendecomposition: bottom-k non-trivial eigenvectors ----
        k_cap = 8
        k = min(k_cap, max(1, n_log - 1), max(1, N - 1))

        def bottom_k_nontrivial(L, k):
            vals, vecs = np.linalg.eigh(L)
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
            # Skip eigenvalues equal to (numerically) zero — connected component nullspace.
            tol = 1e-8
            keep = []
            for idx in range(len(vals)):
                if vals[idx] > tol:
                    keep.append(idx)
                if len(keep) == k:
                    break
            if len(keep) < k:
                # fall back: take last (n - len(keep)) smallest regardless
                extras = [idx for idx in range(len(vals)) if idx not in keep]
                for idx in extras:
                    keep.append(idx)
                    if len(keep) == k:
                        break
            keep = keep[:k]
            return vecs[:, keep]

        X_log = bottom_k_nontrivial(L_log, k)        # n_log x k
        X_phys = bottom_k_nontrivial(L_phys, k)      # N x k

        # ---- 5. Procrustes alignment on zero-padded square logical embedding ----
        X_log_padded = np.zeros((N, k), dtype=float)
        for i in range(n_log):
            X_log_padded[i, :] = X_log[i, :]

        # Align X_log_padded to X_phys via orthogonal Procrustes: find R minimizing ||X_log_padded R - X_phys||
        M_cross = X_log_padded.T @ X_phys
        try:
            U, _, Vt = np.linalg.svd(M_cross, full_matrices=False)
            R = U @ Vt
            X_log_aligned = X_log_padded @ R
        except Exception:
            X_log_aligned = X_log_padded

        # ---- 6. Nearest-neighbor matching with priority by logical activity ----
        # Order logicals by activity (descending) so heavy-traffic qubits pick first.
        activity = getattr(self, "logical_activity", None)
        def act_of(q):
            if activity is None:
                return 0.0
            try:
                return float(activity[q])
            except Exception:
                return 0.0

        order = sorted(active_logicals, key=lambda q: (-act_of(q), q))

        used_phys = set()
        assigned = {}  # logical_q -> physical_q

        for q in order:
            i = log_index[q]
            v = X_log_aligned[i, :]
            # distance to every physical
            diffs = X_phys - v[None, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            # tie-break by centrality (more central first)
            cent = getattr(self, "physical_centrality", None)
            best_p = -1
            best_key = None
            sorted_phys = np.argsort(dists)
            for p in sorted_phys:
                p = int(p)
                if p in used_phys:
                    continue
                c = 0.0
                if cent is not None:
                    try:
                        c = float(cent.get(p, 0.0))
                    except Exception:
                        c = 0.0
                key = (dists[p], -c, p)
                if best_key is None or key < best_key:
                    best_p = p
                    best_key = key
                    break
            if best_p < 0:
                continue
            assigned[q] = best_p
            used_phys.add(best_p)

        # ---- 7. Place idle logicals onto remaining physicals (by centrality) ----
        all_logicals = set(range(N))
        idle_logicals = sorted(all_logicals - set(assigned.keys()))
        remaining_phys = [p for p in range(N) if p not in used_phys]

        cent = getattr(self, "physical_centrality", None)
        def cent_of(p):
            if cent is None:
                return 0.0
            try:
                return float(cent.get(p, 0.0))
            except Exception:
                return 0.0
        remaining_phys.sort(key=lambda p: (-cent_of(p), p))

        for q in idle_logicals:
            if not remaining_phys:
                break
            p = remaining_phys.pop(0)
            assigned[q] = p
            used_phys.add(p)

        # ---- 8. Build the output lists ----
        new_map = [-1] * N
        new_rev = [-1] * N
        for q, p in assigned.items():
            if 0 <= q < N and 0 <= p < N:
                new_map[q] = p
                new_rev[p] = q

        # Fill any remaining unassigned logicals with leftover physicals (identity-ish).
        leftover_phys = [p for p in range(N) if p not in used_phys]
        for q in range(N):
            if new_map[q] == -1:
                if leftover_phys:
                    p = leftover_phys.pop(0)
                else:
                    # find any unused
                    used_now = set(x for x in new_map if x != -1)
                    candidates = [pp for pp in range(N) if pp not in used_now]
                    if not candidates:
                        break
                    p = candidates[0]
                new_map[q] = p
                new_rev[p] = q

        # Validate; if anything is off, fall back to identity.
        if -1 in new_map or len(set(new_map)) != N:
            new_map = [i for i in range(N)]
            new_rev = [i for i in range(N)]

        self.mapping_dict = new_map
        self.reverse_mapping_dict = new_rev

    except Exception:
        self.mapping_dict = [i for i in range(N)]
        self.reverse_mapping_dict = [i for i in range(N)]

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)