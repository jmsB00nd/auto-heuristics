def init_mapping(self):
    import collections

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = collections.defaultdict(int)
    logical_qubits = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_qubits.add(a)
                continue
            logical_qubits.add(a); logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            interactions[key] += 1
        else:
            for q in qubits:
                logical_qubits.add(q)
    logical_qubits = sorted(logical_qubits)

    neighbors = collections.defaultdict(set)
    for (a, b) in interactions.keys():
        neighbors[a].add(b)
        neighbors[b].add(a)

    def run_ac3(hop_budget):
        domains = {L: set(range(N)) for L in logical_qubits}
        queue = collections.deque()
        for (a, b) in interactions.keys():
            queue.append((a, b)); queue.append((b, a))
        while queue:
            Xi, Xj = queue.popleft()
            to_remove = []
            for vi in domains[Xi]:
                ok = False
                for vj in domains[Xj]:
                    if vi != vj and self.distance_matrix[vi][vj] <= hop_budget:
                        ok = True
                        break
                if not ok:
                    to_remove.append(vi)
            if to_remove:
                for v in to_remove:
                    domains[Xi].discard(v)
                if not domains[Xi]:
                    return None
                for Xk in neighbors[Xi]:
                    if Xk != Xj:
                        queue.append((Xk, Xi))
        return domains

    domains = None
    if interactions:
        max_dist = 1
        for i in range(N):
            row = self.distance_matrix[i]
            for j in range(N):
                if row[j] > max_dist:
                    max_dist = row[j]
        for budget in range(1, max_dist + 1):
            domains = run_ac3(budget)
            if domains is not None:
                break
    if domains is None:
        domains = {L: set(range(N)) for L in logical_qubits}

    activity = collections.defaultdict(int)
    for (a, b), w in interactions.items():
        activity[a] += w
        activity[b] += w

    used_physical = set()
    assigned_logical = set()

    order = sorted(logical_qubits, key=lambda L: (len(domains.get(L, set(range(N)))), -activity[L], L))

    for L in order:
        dom = domains.get(L, set(range(N)))
        candidates = [p for p in dom if p not in used_physical]
        if not candidates:
            continue

        def score(p):
            s = 0.0
            for nb in neighbors[L]:
                if nb in assigned_logical:
                    s -= self.distance_matrix[p][self.mapping_dict[nb]]
            s += self.physical_centrality.get(p, 0.0) * 1e-3
            if p == L:
                s += 1e-9
            return s

        best = max(candidates, key=score)
        self.mapping_dict[L] = best
        self.reverse_mapping_dict[best] = L
        used_physical.add(best)
        assigned_logical.add(L)

    unassigned = [L for L in range(N) if self.mapping_dict[L] == -1]
    still_unassigned = []
    for L in unassigned:
        if L not in used_physical:
            self.mapping_dict[L] = L
            self.reverse_mapping_dict[L] = L
            used_physical.add(L)
        else:
            still_unassigned.append(L)
    remaining_physical = [p for p in range(N) if p not in used_physical]
    for L, p in zip(still_unassigned, remaining_physical):
        self.mapping_dict[L] = p
        self.reverse_mapping_dict[p] = L
        used_physical.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)