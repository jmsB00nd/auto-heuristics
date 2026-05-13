def init_mapping(self):
    import math
    N = self.num_qubits

    mapping = [-1] * N
    reverse = [-1] * N

    # 1) Collect logical edges with weights from access (fallback to QIG).
    edge_weight = {}
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                active_logicals.add(a)
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] = edge_weight.get(key, 0) + 1
            active_logicals.add(a)
            active_logicals.add(b)

    if not edge_weight and getattr(self, "qubit_interaction_graph", None):
        for u, nbrs in self.qubit_interaction_graph.items():
            for v, w in nbrs.items():
                if u == v or w <= 0:
                    continue
                key = (u, v) if u < v else (v, u)
                if key not in edge_weight:
                    edge_weight[key] = w
                active_logicals.add(u)
                active_logicals.add(v)

    # 2) Sort logical edges by descending weight.
    sorted_edges = sorted(edge_weight.items(), key=lambda kv: -kv[1])

    # 3) Precompute physical pair list sorted by (distance asc, -centrality_sum).
    centrality = getattr(self, "physical_centrality", {}) or {}
    def cscore(p):
        return centrality.get(p, 0.0)

    dist = self.distance_matrix
    phys_pairs = []
    for p in range(N):
        for q in range(p + 1, N):
            d = dist[p][q]
            if d <= 0:
                d = math.inf
            phys_pairs.append((d, -(cscore(p) + cscore(q)), p, q))
    phys_pairs.sort()

    used_phys = set()
    placed_log = set()

    def best_unused_pair():
        for d, _, p, q in phys_pairs:
            if p not in used_phys and q not in used_phys:
                return p, q
        return None

    def best_unused_neighbor_of(phys_anchor):
        # Pick unused physical with smallest distance to anchor (ties: higher centrality).
        best = None
        best_key = None
        for cand in range(N):
            if cand == phys_anchor or cand in used_phys:
                continue
            d = dist[phys_anchor][cand]
            if d <= 0:
                continue
            key = (d, -cscore(cand))
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        return best

    def assign(log_q, phys_q):
        mapping[log_q] = phys_q
        reverse[phys_q] = log_q
        used_phys.add(phys_q)
        placed_log.add(log_q)

    # 4) Greedy edge processing in descending weight.
    for (u, v), _w in sorted_edges:
        u_placed = u in placed_log
        v_placed = v in placed_log
        if u_placed and v_placed:
            continue
        if not u_placed and not v_placed:
            pair = best_unused_pair()
            if pair is None:
                break
            p, q = pair
            # Orient by centrality: heavier-activity logical to more central physical.
            au = self.logical_activity.get(u, 0) if hasattr(self, "logical_activity") else 0
            av = self.logical_activity.get(v, 0) if hasattr(self, "logical_activity") else 0
            if cscore(p) >= cscore(q):
                hi_p, lo_p = p, q
            else:
                hi_p, lo_p = q, p
            if au >= av:
                assign(u, hi_p)
                assign(v, lo_p)
            else:
                assign(v, hi_p)
                assign(u, lo_p)
        else:
            anchor_log = u if u_placed else v
            other_log = v if u_placed else u
            anchor_phys = mapping[anchor_log]
            cand = best_unused_neighbor_of(anchor_phys)
            if cand is None:
                continue
            assign(other_log, cand)

    # 5) Place remaining active logicals onto most-central unused physicals.
    remaining_active = [l for l in active_logicals if l not in placed_log]
    remaining_active.sort(
        key=lambda l: -(self.logical_activity.get(l, 0) if hasattr(self, "logical_activity") else 0)
    )
    unused_sorted = sorted(
        (p for p in range(N) if p not in used_phys),
        key=lambda p: -cscore(p),
    )
    for log_q in remaining_active:
        if not unused_sorted:
            break
        phys_q = unused_sorted.pop(0)
        assign(log_q, phys_q)

    # 6) Identity-style backfill for any still-unmapped logical slot.
    unused_phys_list = [p for p in range(N) if p not in used_phys]
    idx = 0
    for log_q in range(N):
        if mapping[log_q] == -1:
            # Prefer identity when free, else next unused physical.
            if log_q not in used_phys:
                phys_q = log_q
                # remove from unused list if present
                if phys_q in unused_phys_list:
                    unused_phys_list.remove(phys_q)
            else:
                if idx >= len(unused_phys_list):
                    break
                phys_q = unused_phys_list[idx]
                idx += 1
            assign(log_q, phys_q)

    # Final safety pass: any remaining -1 gets next free physical.
    leftover_phys = [p for p in range(N) if p not in used_phys]
    li = 0
    for log_q in range(N):
        if mapping[log_q] == -1:
            if li >= len(leftover_phys):
                break
            phys_q = leftover_phys[li]
            li += 1
            mapping[log_q] = phys_q
            reverse[phys_q] = log_q
            used_phys.add(phys_q)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)