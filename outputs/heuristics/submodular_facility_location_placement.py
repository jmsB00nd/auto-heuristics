def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    pair_w = defaultdict(int)
    log_activity = defaultdict(int)
    logicals = set()
    for gid, qs in self.access.items():
        if len(qs) == 2:
            a, b = qs[0], qs[1]
            if a == b:
                logicals.add(a)
                continue
            key = (a, b) if a < b else (b, a)
            pair_w[key] += 1
            log_activity[a] += 1
            log_activity[b] += 1
            logicals.add(a)
            logicals.add(b)
        elif len(qs) == 1:
            logicals.add(qs[0])

    centrality = {}
    if getattr(self, "physical_centrality", None):
        for p in range(N):
            centrality[p] = self.physical_centrality.get(p, 0.0)
    else:
        for p in range(N):
            centrality[p] = 0.0

    sorted_log = sorted(logicals, key=lambda q: (-log_activity.get(q, 0), q))
    num_active = sum(1 for q in sorted_log if log_activity.get(q, 0) > 0)
    num_hubs = min(max(num_active, 1 if sorted_log else 0), N, len(sorted_log))

    placed_log = set()
    used_phys = set()

    if num_hubs > 0:
        l0 = sorted_log[0]
        p0 = max(range(N), key=lambda p: (centrality.get(p, 0.0), -p))
        self.mapping_dict[l0] = p0
        self.reverse_mapping_dict[p0] = l0
        placed_log.add(l0)
        used_phys.add(p0)

    for i in range(1, num_hubs):
        log_q = sorted_log[i]
        best_phy = None
        best_gain = -1.0
        for phy in range(N):
            if phy in used_phys:
                continue
            gain = 0.0
            for pl in placed_log:
                pair = (log_q, pl) if log_q < pl else (pl, log_q)
                w = pair_w.get(pair, 0)
                if w == 0:
                    continue
                pp = self.mapping_dict[pl]
                d = self.distance_matrix[phy][pp]
                gain += w / (1.0 + d)
            gain += centrality.get(phy, 0.0) * 1e-9
            if gain > best_gain:
                best_gain = gain
                best_phy = phy
        if best_phy is None:
            break
        self.mapping_dict[log_q] = best_phy
        self.reverse_mapping_dict[best_phy] = log_q
        placed_log.add(log_q)
        used_phys.add(best_phy)

    for log_q in sorted_log[num_hubs:]:
        if log_q in placed_log:
            continue
        best_hub_phy = None
        best_w = -1
        for pl in placed_log:
            pair = (log_q, pl) if log_q < pl else (pl, log_q)
            w = pair_w.get(pair, 0)
            if w > best_w:
                best_w = w
                best_hub_phy = self.mapping_dict[pl]
        if best_hub_phy is None and used_phys:
            best_hub_phy = max(used_phys, key=lambda p: centrality.get(p, 0.0))
        if best_hub_phy is None:
            best_hub_phy = max(range(N), key=lambda p: centrality.get(p, 0.0))
        candidates = [p for p in range(N) if p not in used_phys]
        if not candidates:
            break
        candidates.sort(key=lambda p: (self.distance_matrix[best_hub_phy][p], -centrality.get(p, 0.0), p))
        chosen = candidates[0]
        self.mapping_dict[log_q] = chosen
        self.reverse_mapping_dict[chosen] = log_q
        placed_log.add(log_q)
        used_phys.add(chosen)

    for L in range(N):
        if self.mapping_dict[L] is None and L not in used_phys:
            self.mapping_dict[L] = L
            self.reverse_mapping_dict[L] = L
            used_phys.add(L)

    free_phys = [p for p in range(N) if p not in used_phys]
    fi = 0
    for L in range(N):
        if self.mapping_dict[L] is None:
            while fi < len(free_phys) and free_phys[fi] in used_phys:
                fi += 1
            if fi >= len(free_phys):
                break
            p = free_phys[fi]
            fi += 1
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)