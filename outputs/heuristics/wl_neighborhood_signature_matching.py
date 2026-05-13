def init_mapping(self):
    import numpy as np
    from collections import defaultdict, Counter

    N = self.num_qubits

    try:
        # ---------- 1. Gather logical interactions ----------
        interactions = []
        active_logicals = set()
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = int(qubits[0]), int(qubits[1])
                if a == b:
                    continue
                interactions.append((a, b))
                active_logicals.add(a)
                active_logicals.add(b)

        active_logicals = sorted(active_logicals)
        L = len(active_logicals)

        # ---------- 2. Build adjacency for both graphs ----------
        # Logical: weighted via qubit_interaction_graph
        log_adj = defaultdict(lambda: defaultdict(float))
        for u in active_logicals:
            row = self.qubit_interaction_graph.get(u, {})
            for v, w in row.items():
                if v != u and v in active_logicals and w > 0:
                    log_adj[u][v] += float(w)

        # Physical: from backend
        phys_adj = defaultdict(set)
        for p in range(N):
            for q in self.backend.get(p, set()):
                if q != p and 0 <= q < N:
                    phys_adj[p].add(q)

        # ---------- 3. WL color refinement ----------
        K_ROUNDS = 3

        def wl_refine(nodes, neighbor_fn, init_label_fn):
            # Returns: list of dicts (one per round, including round 0) -> {node: color_int}
            colors_per_round = []
            label_to_id = {}
            def canon(label):
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                return label_to_id[label]
            # Round 0: init labels
            current = {}
            for n in nodes:
                current[n] = canon(("init", init_label_fn(n)))
            colors_per_round.append(dict(current))
            # Subsequent rounds
            for _ in range(K_ROUNDS):
                nxt = {}
                for n in nodes:
                    nbr_colors = tuple(sorted(neighbor_fn(n, current)))
                    nxt[n] = canon((current[n], nbr_colors))
                current = nxt
                colors_per_round.append(dict(current))
            return colors_per_round

        # Logical neighbor function (weighted: bucket weights into integer bins)
        def log_neighbors(n, current):
            out = []
            for v, w in log_adj.get(n, {}).items():
                if v in current:
                    bin_w = int(round(min(w, 16.0)))
                    out.append((current[v], bin_w))
            return out

        def log_init(n):
            deg = sum(log_adj.get(n, {}).values())
            return int(round(min(deg, 32.0)))

        def phys_neighbors(n, current):
            out = []
            for v in phys_adj.get(n, set()):
                if v in current:
                    out.append(current[v])
            return out

        def phys_init(n):
            return len(phys_adj.get(n, set()))

        log_rounds = wl_refine(active_logicals, log_neighbors, log_init)
        phys_rounds = wl_refine(list(range(N)), phys_neighbors, phys_init)

        # ---------- 4. Build histogram feature vectors per round ----------
        def build_features(nodes, rounds):
            # For each round, count occurrences of each color across all nodes,
            # and represent each node by [round_color_freq for each round]
            feats = {}
            for n in nodes:
                vec = []
                for r_colors in rounds:
                    c = r_colors[n]
                    # frequency of this color in this round (global structural role weight)
                    freq = sum(1 for x in r_colors.values() if x == c)
                    vec.append(freq)
                    vec.append(c % 997)  # color id signature (mod for stability)
                feats[n] = np.array(vec, dtype=float)
            return feats

        log_feats = build_features(active_logicals, log_rounds)
        phys_feats = build_features(list(range(N)), phys_rounds)

        # Normalize features per dimension
        def normalize(feats_dict):
            if not feats_dict:
                return feats_dict
            mat = np.stack(list(feats_dict.values()))
            mn = mat.min(axis=0)
            mx = mat.max(axis=0)
            rng = np.where((mx - mn) > 0, (mx - mn), 1.0)
            return {k: (v - mn) / rng for k, v in feats_dict.items()}

        log_feats = normalize(log_feats)
        phys_feats = normalize(phys_feats)

        # ---------- 5. Cost matrix + Hungarian on square pad ----------
        size = max(L, N)
        BIG = 1e6
        cost = np.full((size, size), BIG, dtype=float)

        # Activity for tiebreak
        act = self.logical_activity if hasattr(self, "logical_activity") else {}
        max_act = max(act.values()) if act else 1.0
        max_act = max_act if max_act > 0 else 1.0

        cent = self.physical_centrality if hasattr(self, "physical_centrality") else {}
        max_cent = max(cent.values()) if cent else 1.0
        max_cent = max_cent if max_cent > 0 else 1.0

        for i, lq in enumerate(active_logicals):
            lf = log_feats[lq]
            a_norm = act.get(lq, 0) / max_act
            for p in range(N):
                pf = phys_feats[p]
                # Pad/truncate to common length
                if lf.shape[0] != pf.shape[0]:
                    m = min(lf.shape[0], pf.shape[0])
                    diff = np.abs(lf[:m] - pf[:m]).sum()
                else:
                    diff = np.abs(lf - pf).sum()
                # Tiebreak: high-activity logicals prefer central physicals
                c_norm = cent.get(p, 0.0) / max_cent
                penalty = a_norm * (1.0 - c_norm) * 0.1
                cost[i, p] = diff + penalty

        # Hungarian
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            row_ind = np.arange(size)
            col_ind = np.arange(size)

        # ---------- 6. Build mapping lists ----------
        mapping = [-1] * N
        reverse = [-1] * N
        used_phys = set()

        for r, c in zip(row_ind, col_ind):
            if r < L and c < N:
                lq = active_logicals[r]
                pq = int(c)
                if pq not in used_phys and 0 <= lq < N:
                    mapping[lq] = pq
                    reverse[pq] = lq
                    used_phys.add(pq)

        # Fill remaining logicals (idle ones) with identity-then-first-free
        for lq in range(N):
            if mapping[lq] == -1:
                if lq not in used_phys:
                    mapping[lq] = lq
                    reverse[lq] = lq
                    used_phys.add(lq)
                else:
                    for pq in range(N):
                        if pq not in used_phys:
                            mapping[lq] = pq
                            reverse[pq] = lq
                            used_phys.add(pq)
                            break

        self.mapping_dict = mapping
        self.reverse_mapping_dict = reverse

    except Exception:
        # Fallback: identity
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    # Final safety: ensure no -1 and a valid permutation
    if -1 in self.mapping_dict or len(set(self.mapping_dict)) != N:
        used = set()
        m = [-1] * N
        for lq in range(N):
            pq = self.mapping_dict[lq] if 0 <= self.mapping_dict[lq] < N else -1
            if pq != -1 and pq not in used:
                m[lq] = pq
                used.add(pq)
        for lq in range(N):
            if m[lq] == -1:
                for pq in range(N):
                    if pq not in used:
                        m[lq] = pq
                        used.add(pq)
                        break
        self.mapping_dict = m
        self.reverse_mapping_dict = [-1] * N
        for lq, pq in enumerate(self.mapping_dict):
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)