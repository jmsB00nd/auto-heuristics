def init_mapping(self):
    import math
    import random
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- collect logical interactions ----
    interactions = []  # (l1, l2, weight)
    seen_pairs = {}
    logicals_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logicals_set.add(a)
            logicals_set.add(b)
            key = (a, b) if a < b else (b, a)
            seen_pairs[key] = seen_pairs.get(key, 0) + 1
        elif len(qubits) == 1:
            logicals_set.add(qubits[0])

    qig = getattr(self, "qubit_interaction_graph", None)
    for (a, b), cnt in seen_pairs.items():
        w = cnt
        if qig is not None:
            try:
                w = max(w, qig[a][b])
            except Exception:
                pass
        interactions.append((a, b, float(w)))

    # adjacency over logicals
    ladj = defaultdict(dict)
    for a, b, w in interactions:
        ladj[a][b] = w
        ladj[b][a] = w

    # logical ordering: by activity desc, then degree desc
    activity = getattr(self, "logical_activity", None)
    def lkey(q):
        act = activity[q] if (activity is not None and q in activity) else 0
        return (-act, -len(ladj.get(q, {})), q)
    logicals_ordered = sorted(logicals_set, key=lkey)
    L = len(logicals_ordered)
    pos_of_logical = {q: i for i, q in enumerate(logicals_ordered)}

    # physical ordering by centrality (best first)
    cent = getattr(self, "physical_centrality", None)
    def pkey(p):
        c = cent[p] if (cent is not None and p in cent) else 0.0
        return -c
    phys_sorted = sorted(range(N), key=pkey)

    dist = self.distance_matrix

    # ---- helpers ----
    def partial_cost(mapping_partial):
        # cost over interacting pairs both placed
        c = 0.0
        for i in range(len(mapping_partial)):
            li = logicals_ordered[i]
            pi = mapping_partial[i]
            if pi < 0:
                continue
            nbrs = ladj.get(li, {})
            for lj, w in nbrs.items():
                j = pos_of_logical.get(lj, -1)
                if 0 <= j < i:
                    pj = mapping_partial[j]
                    if pj >= 0:
                        c += w * (dist[pi][pj] - 1)
        return c

    def candidate_physicals(mapping_partial, used, idx):
        # next logical to place
        li = logicals_ordered[idx]
        nbrs = ladj.get(li, {})
        placed_partner_phys = []
        for lj, w in nbrs.items():
            j = pos_of_logical.get(lj, -1)
            if 0 <= j < idx:
                pj = mapping_partial[j]
                if pj >= 0:
                    placed_partner_phys.append((pj, w))
        avail = [p for p in phys_sorted if p not in used]
        if not placed_partner_phys:
            # use centrality order, capped
            return avail[: min(len(avail), max(4, N // 2))]
        # rank by weighted distance to partners
        def score(p):
            s = 0.0
            for pj, w in placed_partner_phys:
                s += w * dist[p][pj]
            return s
        avail.sort(key=score)
        # cap branching
        cap = min(len(avail), 6)
        return avail[:cap]

    def rollout(mapping_partial, used, idx):
        mp = list(mapping_partial)
        u = set(used)
        for k in range(idx, L):
            lk = logicals_ordered[k]
            nbrs = ladj.get(lk, {})
            placed_partners = []
            for lj, w in nbrs.items():
                j = pos_of_logical.get(lj, -1)
                if 0 <= j < k and mp[j] >= 0:
                    placed_partners.append((mp[j], w))
            avail = [p for p in range(N) if p not in u]
            if not avail:
                break
            if placed_partners:
                weights = []
                for p in avail:
                    s = 0.0
                    for pj, w in placed_partners:
                        s += w * dist[p][pj]
                    weights.append(1.0 / (1.0 + s))
                total = sum(weights)
                if total <= 0:
                    chosen = random.choice(avail)
                else:
                    r = random.random() * total
                    acc = 0.0
                    chosen = avail[-1]
                    for p, ww in zip(avail, weights):
                        acc += ww
                        if acc >= r:
                            chosen = p
                            break
            else:
                # bias toward central
                chosen = avail[0] if random.random() < 0.5 else random.choice(avail)
            mp.append(chosen)
            u.add(chosen)
        # total cost on complete (or as-far-as-possible) mapping
        total = 0.0
        for a, b, w in interactions:
            ia = pos_of_logical.get(a, -1)
            ib = pos_of_logical.get(b, -1)
            if 0 <= ia < len(mp) and 0 <= ib < len(mp):
                pa = mp[ia]
                pb = mp[ib]
                if pa >= 0 and pb >= 0:
                    total += w * (dist[pa][pb] - 1)
        return total, mp

    # ---- MCTS ----
    best_mapping = None
    best_cost = float("inf")

    if L > 0:
        # node: dict with keys: parent, children (dict phys->node), visits, value_sum,
        # untried (list), idx, mapping_partial, used
        root = {
            "parent": None,
            "children": {},
            "visits": 0,
            "value_sum": 0.0,
            "idx": 0,
            "mapping_partial": [],
            "used": set(),
            "untried": None,
            "from_phys": None,
        }
        root["untried"] = candidate_physicals(root["mapping_partial"], root["used"], 0) if L > 0 else []

        iterations = min(300, max(50, 10 * L))
        scale = max(1.0, float(L))
        C = 1.4

        for _ in range(iterations):
            node = root
            # SELECTION
            while node["idx"] < L and not node["untried"] and node["children"]:
                best_child = None
                best_ucb = -float("inf")
                ln_n = math.log(max(1, node["visits"]))
                for ph, ch in node["children"].items():
                    if ch["visits"] == 0:
                        ucb = float("inf")
                    else:
                        exploit = ch["value_sum"] / ch["visits"]
                        explore = C * math.sqrt(ln_n / ch["visits"])
                        ucb = exploit + explore
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = ch
                if best_child is None:
                    break
                node = best_child

            # EXPANSION
            if node["idx"] < L and node["untried"]:
                ph = node["untried"].pop(0)
                new_partial = node["mapping_partial"] + [ph]
                new_used = set(node["used"])
                new_used.add(ph)
                new_idx = node["idx"] + 1
                child = {
                    "parent": node,
                    "children": {},
                    "visits": 0,
                    "value_sum": 0.0,
                    "idx": new_idx,
                    "mapping_partial": new_partial,
                    "used": new_used,
                    "untried": None,
                    "from_phys": ph,
                }
                if new_idx < L:
                    child["untried"] = candidate_physicals(new_partial, new_used, new_idx)
                else:
                    child["untried"] = []
                node["children"][ph] = child
                node = child

            # SIMULATION
            cost, full_mp = rollout(node["mapping_partial"], node["used"], node["idx"])
            if cost < best_cost and len(full_mp) == L:
                best_cost = cost
                best_mapping = full_mp

            reward = math.exp(-cost / scale)

            # BACKPROP
            cur = node
            while cur is not None:
                cur["visits"] += 1
                cur["value_sum"] += reward
                cur = cur["parent"]

    # ---- materialize mapping ----
    used_phys = set()
    if best_mapping is not None and len(best_mapping) == L:
        for i, lq in enumerate(logicals_ordered):
            p = best_mapping[i]
            if 0 <= lq < N and 0 <= p < N and p not in used_phys:
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                used_phys.add(p)

    # fallback for any logical in access not yet placed
    leftover = [p for p in phys_sorted if p not in used_phys]
    for lq in logicals_ordered:
        if 0 <= lq < N and self.mapping_dict[lq] == -1:
            if leftover:
                p = leftover.pop(0)
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                used_phys.add(p)

    # fill remaining logical ids (those not in access) with identity-preferred leftovers
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            if lq not in used_phys:
                self.mapping_dict[lq] = lq
                self.reverse_mapping_dict[lq] = lq
                used_phys.add(lq)
            else:
                if leftover:
                    p = leftover.pop(0)
                else:
                    remaining = [p for p in range(N) if p not in used_phys]
                    if not remaining:
                        break
                    p = remaining[0]
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                used_phys.add(p)

    # final safety: if any duplicates somehow, rebuild identity
    if len(set(self.mapping_dict)) != len(self.mapping_dict) or any(p < 0 for p in self.mapping_dict):
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)