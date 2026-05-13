def init_mapping(self):
    import math
    from collections import defaultdict, Counter

    N = self.num_qubits

    # Collect logical interactions from self.access (DAG not yet built)
    logical_adj = defaultdict(lambda: defaultdict(int))
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a][b] += 1
            logical_adj[b][a] += 1
            active_logicals.add(a)
            active_logicals.add(b)
        elif len(qubits) == 1:
            active_logicals.add(qubits[0])

    # Physical adjacency
    phys_adj = {p: list(self.backend.get(p, set())) for p in range(N)}

    # Logical node universe (cap to N)
    logical_nodes = [q for q in range(N)]

    # Initial colors: based on degree
    K_ROUNDS = 4
    DIM = max(32, N)  # histogram dimension

    def initial_color_logical(q):
        deg = len(logical_adj.get(q, {}))
        wdeg = sum(logical_adj.get(q, {}).values())
        return hash(("L0", deg, wdeg)) & 0xFFFFFFFF

    def initial_color_physical(p):
        deg = len(phys_adj.get(p, []))
        return hash(("P0", deg)) & 0xFFFFFFFF

    log_colors = {q: initial_color_logical(q) for q in logical_nodes}
    phy_colors = {p: initial_color_physical(p) for p in range(N)}

    # Per-round signature accumulators (histogram of size DIM per round)
    log_sig = {q: [0] * (DIM * K_ROUNDS) for q in logical_nodes}
    phy_sig = {p: [0] * (DIM * K_ROUNDS) for p in range(N)}

    def bump(sig, round_idx, color):
        idx = round_idx * DIM + (color % DIM)
        sig[idx] += 1

    # Record round 0
    for q in logical_nodes:
        bump(log_sig[q], 0, log_colors[q])
    for p in range(N):
        bump(phy_sig[p], 0, phy_colors[p])

    # WL refinement rounds
    for r in range(1, K_ROUNDS):
        new_log = {}
        for q in logical_nodes:
            neigh = logical_adj.get(q, {})
            multiset = tuple(sorted(log_colors[n] for n in neigh.keys()))
            new_log[q] = hash((log_colors[q], multiset)) & 0xFFFFFFFF
        new_phy = {}
        for p in range(N):
            multiset = tuple(sorted(phy_colors[n] for n in phys_adj.get(p, [])))
            new_phy[p] = hash((phy_colors[p], multiset)) & 0xFFFFFFFF
        log_colors = new_log
        phy_colors = new_phy
        for q in logical_nodes:
            bump(log_sig[q], r, log_colors[q])
        for p in range(N):
            bump(phy_sig[p], r, phy_colors[p])

    # Precompute physical signature norms
    def norm(v):
        s = 0.0
        for x in v:
            s += x * x
        return math.sqrt(s) if s > 0 else 1.0

    phy_norm = {p: norm(phy_sig[p]) for p in range(N)}

    # Order logicals by activity (descending), physicals by centrality (descending)
    activity = {q: sum(logical_adj.get(q, {}).values()) for q in logical_nodes}
    logical_order = sorted(active_logicals, key=lambda q: -activity.get(q, 0))
    # Append inactive logicals at the end
    inactive = [q for q in logical_nodes if q not in active_logicals]
    logical_order.extend(inactive)

    centrality = getattr(self, "physical_centrality", {})
    phys_pool = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))

    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_phys = set()

    for q in logical_order:
        if q >= N:
            continue
        sig_q = log_sig[q]
        nq = norm(sig_q)
        best_p = -1
        best_score = -1.0
        for p in phys_pool:
            if p in used_phys:
                continue
            sp = phy_sig[p]
            dot = 0
            for i in range(len(sig_q)):
                dot += sig_q[i] * sp[i]
            score = dot / (nq * phy_norm[p])
            if score > best_score:
                best_score = score
                best_p = p
        if best_p == -1:
            continue
        mapping_dict[q] = best_p
        reverse_mapping_dict[best_p] = q
        used_phys.add(best_p)

    # Fill any unmapped logicals with leftover physicals
    leftover = [p for p in range(N) if p not in used_phys]
    for q in range(N):
        if mapping_dict[q] == -1:
            if leftover:
                p = leftover.pop(0)
                mapping_dict[q] = p
                reverse_mapping_dict[p] = q
                used_phys.add(p)

    # Identity fallback for any remaining (should not happen)
    for q in range(N):
        if mapping_dict[q] == -1:
            for p in range(N):
                if p not in used_phys:
                    mapping_dict[q] = p
                    reverse_mapping_dict[p] = q
                    used_phys.add(p)
                    break

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)