def init_mapping(self):
    import math
    import random
    import numpy as np

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- 1. Physical-qubit coordinates via spectral embedding ----
    def _spectral_coords():
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except Exception:
            sp = None
        # Build symmetric adjacency
        A = np.zeros((N, N), dtype=float)
        for u, nbrs in self.backend.items():
            for v in nbrs:
                if 0 <= u < N and 0 <= v < N:
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        deg = A.sum(axis=1)
        D = np.diag(deg)
        L = D - A
        try:
            # Use the two smallest non-trivial eigenvectors
            w, V = np.linalg.eigh(L)
            order = np.argsort(w)
            # Skip the zero eigenvalue (trivial)
            idx = [i for i in order if w[i] > 1e-9][:2]
            if len(idx) < 2:
                raise ValueError
            coords = V[:, idx]
        except Exception:
            # Fallback: random coords
            rng = np.random.default_rng(0)
            coords = rng.standard_normal((N, 2))
        # Normalize to unit box
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        span = np.where(mx - mn > 1e-12, mx - mn, 1.0)
        coords = (coords - mn) / span
        return coords  # shape (N, 2), in [0,1]^2

    phys_coords = _spectral_coords()

    # ---- 2. Logical qubits and weighted interaction edges ----
    logical_set = set()
    for gid, qs in self.access.items():
        for q in qs:
            logical_set.add(q)
    # also include anything from QIG keys (defensive)
    for q in list(self.qubit_interaction_graph.keys()):
        logical_set.add(q)
    logical_qubits = sorted(logical_set)
    L = len(logical_qubits)

    if L == 0:
        # No logical interactions — identity mapping
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # logical -> contiguous index for arrays
    lid = {lq: i for i, lq in enumerate(logical_qubits)}

    edges = []  # (i, j, weight) with i<j
    seen_pairs = set()
    for q1 in logical_qubits:
        nbrs = self.qubit_interaction_graph.get(q1, {})
        for q2, w in nbrs.items():
            if q2 not in lid or q1 == q2:
                continue
            a, b = (q1, q2) if q1 < q2 else (q2, q1)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            edges.append((lid[a], lid[b], float(w)))
    # Fallback: derive from self.access if QIG empty
    if not edges:
        from collections import Counter
        cnt = Counter()
        for gid, qs in self.access.items():
            if len(qs) == 2 and qs[0] != qs[1] and qs[0] in lid and qs[1] in lid:
                a, b = sorted(qs)
                cnt[(a, b)] += 1
        for (a, b), w in cnt.items():
            edges.append((lid[a], lid[b], float(w)))

    # ---- 3. Initialize positions ----
    rng = np.random.default_rng(42)
    pos = np.zeros((L, 2), dtype=float)
    # seed each logical near a random distinct physical coord (or any if L>N)
    seed_perm = rng.permutation(N)
    for i in range(L):
        p = seed_perm[i % N]
        pos[i] = phys_coords[p] + rng.normal(0.0, 0.01, size=2)

    # ---- 4. Fruchterman-Reingold simulation ----
    area = 1.0  # phys_coords lie in [0,1]^2
    k = math.sqrt(area / max(L, 1))
    iterations = 80
    t0 = 0.1  # initial temperature relative to box size
    edges_arr = np.array(edges, dtype=float) if edges else np.zeros((0, 3))
    for it in range(iterations):
        temperature = t0 * (1.0 - it / iterations)

        # Repulsive forces (all pairs, vectorized)
        delta = pos[:, None, :] - pos[None, :, :]            # (L,L,2)
        dist2 = np.sum(delta * delta, axis=2) + 1e-9          # (L,L)
        dist = np.sqrt(dist2)
        # f_rep = k^2 / d  (FR repulsion); inverse-square magnitude on direction
        rep_mag = (k * k) / dist2                             # (L,L)
        np.fill_diagonal(rep_mag, 0.0)
        disp = np.sum((delta / dist[:, :, None]) * rep_mag[:, :, None], axis=1)

        # Attractive forces along weighted QIG edges (Hookean: f_att = w * d^2 / k)
        if edges_arr.shape[0] > 0:
            ii = edges_arr[:, 0].astype(int)
            jj = edges_arr[:, 1].astype(int)
            ww = edges_arr[:, 2]
            d_ij = pos[ii] - pos[jj]
            d_norm = np.linalg.norm(d_ij, axis=1) + 1e-9
            att_mag = ww * (d_norm * d_norm) / k
            unit = d_ij / d_norm[:, None]
            f_ij = unit * att_mag[:, None]
            np.add.at(disp, ii, -f_ij)
            np.add.at(disp, jj,  f_ij)

        # Confining potential: pull toward nearest physical coord
        # Compute nearest physical for each logical
        diff_p = pos[:, None, :] - phys_coords[None, :, :]    # (L,N,2)
        d_p2 = np.sum(diff_p * diff_p, axis=2)
        nearest = np.argmin(d_p2, axis=1)
        target = phys_coords[nearest]
        confine = (target - pos) * 0.05  # weak spring to box of valid sites
        disp += confine

        # Limit displacement by temperature
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-9
        scale = np.minimum(disp_norm, temperature) / disp_norm
        pos = pos + disp * scale[:, None]
        # Keep within unit box
        pos = np.clip(pos, 0.0, 1.0)

    # ---- 5. Greedy nearest-unclaimed assignment ----
    # Order logicals by activity (most interactive first → best location priority)
    activity = []
    for lq in logical_qubits:
        a = self.logical_activity.get(lq, 0) if hasattr(self, "logical_activity") else 0
        activity.append(a)
    order = sorted(range(L), key=lambda i: -activity[i])

    claimed = [False] * N
    for i in order:
        lq = logical_qubits[i]
        # distances from pos[i] to every physical coord
        diffs = phys_coords - pos[i]
        d2 = np.sum(diffs * diffs, axis=1)
        cand = np.argsort(d2)
        for p in cand:
            p = int(p)
            if not claimed[p]:
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                claimed[p] = True
                break

    # ---- 6. Back-fill any unmapped logical slots with leftover physicals ----
    leftover = [p for p in range(N) if not claimed[p]]
    li = 0
    for q in range(N):
        if self.mapping_dict[q] == -1:
            if li < len(leftover):
                p = leftover[li]
                li += 1
                self.mapping_dict[q] = p
                self.reverse_mapping_dict[p] = q
                claimed[p] = True

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)