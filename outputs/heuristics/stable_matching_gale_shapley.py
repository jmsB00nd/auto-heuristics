def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    # --- gather logical qubits and interactions from self.access ---
    logical_set = set()
    interactions = []
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_set.add(q)
        if len(qubits) == 2:
            interactions.append(tuple(qubits))

    active_logicals = sorted(logical_set)

    # --- logical activity / degree proxies ---
    activity = {}
    for q in range(N):
        activity[q] = float(self.logical_activity.get(q, 0)) if hasattr(self.logical_activity, "get") else float(self.logical_activity[q])

    max_act = max(activity.values()) if activity else 1.0
    if max_act <= 0:
        max_act = 1.0

    # --- physical degree from backend adjacency ---
    phys_degree = {p: len(self.backend[p]) for p in range(N)}
    max_pdeg = max(phys_degree.values()) if phys_degree else 1
    if max_pdeg <= 0:
        max_pdeg = 1

    centrality = {p: float(self.physical_centrality.get(p, 0.0)) for p in range(N)}
    max_cent = max(centrality.values()) if centrality else 1.0
    if max_cent <= 0:
        max_cent = 1.0

    # --- logical neighbor-weight totals (for utility) ---
    logical_neighbor_weight = defaultdict(float)
    for q in range(N):
        row = self.qubit_interaction_graph.get(q, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[q]
        s = 0.0
        for nb, w in row.items():
            s += float(w)
        logical_neighbor_weight[q] = s

    # --- Build logical preference list: each logical ranks physicals (best first) ---
    # utility(q, p) = centrality(p) + alpha * (phys_degree(p)/max_pdeg) * (activity(q)/max_act)
    alpha = 1.0
    logical_pref = {}
    for q in range(N):
        scores = []
        a_q = activity[q] / max_act
        nb_w = logical_neighbor_weight[q]
        for p in range(N):
            util = (centrality[p] / max_cent) + alpha * (phys_degree[p] / max_pdeg) * (0.5 + a_q) + 0.01 * nb_w * (phys_degree[p] / max_pdeg)
            # negate for max-heap via min-heap; tie-break by p
            scores.append((-util, p))
        scores.sort()
        logical_pref[q] = [p for _, p in scores]

    # --- Build physical preference: each physical ranks logicals by interaction load * structural compat ---
    # score(p, q) = activity(q) * (1 + phys_degree(p)/max_pdeg) + centrality(p) * (logical_neighbor_weight(q))
    phys_rank = {}
    for p in range(N):
        compat = 1.0 + (phys_degree[p] / max_pdeg)
        c_p = centrality[p] / max_cent
        ranks = {}
        scored = []
        for q in range(N):
            s = activity[q] * compat + c_p * logical_neighbor_weight[q] * 0.1
            scored.append((-s, q))
        scored.sort()
        for rank_idx, (_, q) in enumerate(scored):
            ranks[q] = rank_idx
        phys_rank[p] = ranks

    # --- Gale-Shapley: logical-proposing (square N x N market, all logicals participate) ---
    next_proposal = [0] * N            # index into logical_pref[q]
    phys_holder = [-1] * N             # physical -> current logical (-1 if free)
    free_logicals = list(range(N))

    while free_logicals:
        q = free_logicals.pop()
        if next_proposal[q] >= N:
            continue  # exhausted (shouldn't happen in square market)
        p = logical_pref[q][next_proposal[q]]
        next_proposal[q] += 1
        cur = phys_holder[p]
        if cur == -1:
            phys_holder[p] = q
        else:
            # physical prefers lower rank index
            if phys_rank[p][q] < phys_rank[p][cur]:
                phys_holder[p] = q
                free_logicals.append(cur)
            else:
                free_logicals.append(q)

    # --- Build mapping from matching ---
    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_phys = set()
    for p in range(N):
        q = phys_holder[p]
        if q != -1 and 0 <= q < N:
            mapping_dict[q] = p
            reverse_mapping_dict[p] = q
            used_phys.add(p)

    # --- Fallback: assign any unmapped logical to remaining physicals (identity-preferred) ---
    unmapped_logicals = [q for q in range(N) if mapping_dict[q] == -1]
    free_physicals = [p for p in range(N) if p not in used_phys]
    # try identity first
    free_set = set(free_physicals)
    remaining = []
    for q in unmapped_logicals:
        if q in free_set:
            mapping_dict[q] = q
            reverse_mapping_dict[q] = q
            free_set.discard(q)
        else:
            remaining.append(q)
    free_physicals = sorted(free_set)
    for q, p in zip(remaining, free_physicals):
        mapping_dict[q] = p
        reverse_mapping_dict[p] = q

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)