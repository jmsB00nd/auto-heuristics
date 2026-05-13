def init_mapping(self):
    N = self.num_qubits
    backend = self.backend
    dist = self.distance_matrix

    # 1) Collect ordered 2-qubit interactions from self.access (access2q may be None here)
    try:
        gate_ids = sorted(self.access.keys())
    except Exception:
        gate_ids = list(self.access.keys()) if getattr(self, "access", None) else []

    interactions = []
    for gid in gate_ids:
        qs = self.access[gid]
        if qs is not None and len(qs) == 2:
            q1, q2 = qs[0], qs[1]
            if q1 != q2 and 0 <= q1 < N and 0 <= q2 < N:
                interactions.append((q1, q2))

    # 2) Arbitrary "final" mapping = identity
    current_mapping = list(range(N))
    reverse_mapping = list(range(N))

    # 3) Replay gates in REVERSE, routing greedily as if forward
    reversed_interactions = list(reversed(interactions))
    swap_budget_per_gate = max(8, 2 * N)

    for (lq1, lq2) in reversed_interactions:
        guard = 0
        while guard < swap_budget_per_gate:
            p1 = current_mapping[lq1]
            p2 = current_mapping[lq2]
            d12 = dist[p1][p2]
            if d12 <= 1:
                break

            # 4) Greedy SWAP: among neighbors of p1 or p2, pick one that
            #    maximally reduces the distance between the gate's endpoints.
            best_swap = None
            best_new_d = d12
            for nb in backend[p1]:
                nd = dist[nb][p2]
                if nd < best_new_d:
                    best_new_d = nd
                    best_swap = (p1, nb)
            for nb in backend[p2]:
                nd = dist[p1][nb]
                if nd < best_new_d:
                    best_new_d = nd
                    best_swap = (p2, nb)

            if best_swap is None:
                break

            pa, pb = best_swap
            la = reverse_mapping[pa]
            lb = reverse_mapping[pb]
            current_mapping[la], current_mapping[lb] = current_mapping[lb], current_mapping[la]
            reverse_mapping[pa], reverse_mapping[pb] = reverse_mapping[pb], reverse_mapping[pa]
            guard += 1

    # 5) Commit result; identity-backfilled by construction
    self.mapping_dict = list(current_mapping)
    self.reverse_mapping_dict = list(reverse_mapping)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)