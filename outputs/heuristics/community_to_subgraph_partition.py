def init_mapping(self):
    import collections
    import random as _random

    N = self.num_qubits
    mapping = [-1] * N
    reverse_mapping = [-1] * N

    # ---- Step 1: weighted label propagation on the QIG ----
    qig = self.qubit_interaction_graph
    logical_in_qig = set()
    for q1, nbrs in qig.items():
        if 0 <= q1 < N:
            logical_in_qig.add(q1)
        for q2 in nbrs:
            if 0 <= q2 < N:
                logical_in_qig.add(q2)

    labels = {q: q for q in logical_in_qig}
    nodes = list(logical_in_qig)
    rng = _random.Random(42)

    for _ in range(20):
        rng.shuffle(nodes)
        changed = False
        for q in nodes:
            nbrs = qig.get(q, {})
            if not nbrs:
                continue
            wlab = collections.defaultdict(float)
            for nb, w in nbrs.items():
                if nb in labels:
                    wlab[labels[nb]] += w
            if not wlab:
                continue
            max_w = max(wlab.values())
            best = [lab for lab, w in wlab.items() if w == max_w]
            new_label = labels[q] if labels[q] in best else best[0]
            if new_label != labels[q]:
                labels[q] = new_label
                changed = True
        if not changed:
            break

    communities = collections.defaultdict(list)
    for q, lab in labels.items():
        communities[lab].append(q)
    community_list = sorted(communities.values(), key=len, reverse=True)

    # ---- Step 2: grow connected backend subgraphs from centrality hubs ----
    backend_adj = self.backend
    centrality = self.physical_centrality
    sorted_hubs = sorted(range(N), key=lambda p: centrality.get(p, 0.0), reverse=True)

    used_phys = set()
    subgraphs = []
    for comm in community_list:
        size = len(comm)
        hub = next((p for p in sorted_hubs if p not in used_phys), None)
        if hub is None:
            subgraphs.append([])
            continue
        sub = []
        visited = {hub}
        queue = collections.deque([hub])
        while queue and len(sub) < size:
            cur = queue.popleft()
            if cur in used_phys:
                continue
            sub.append(cur)
            used_phys.add(cur)
            nbrs_phys = backend_adj.get(cur, [])
            sorted_nbrs = sorted(nbrs_phys, key=lambda p: centrality.get(p, 0.0), reverse=True)
            for nb in sorted_nbrs:
                if nb not in visited and nb not in used_phys and 0 <= nb < N:
                    visited.add(nb)
                    queue.append(nb)
        subgraphs.append(sub)

    # ---- Step 3: assign each community to its subgraph (activity <-> centrality) ----
    for comm, sub in zip(community_list, subgraphs):
        sorted_comm = sorted(comm, key=lambda q: self.logical_activity.get(q, 0), reverse=True)
        sorted_sub = sorted(sub, key=lambda p: centrality.get(p, 0.0), reverse=True)
        for lq, pq in zip(sorted_comm, sorted_sub):
            if 0 <= lq < N and 0 <= pq < N and mapping[lq] == -1 and reverse_mapping[pq] == -1:
                mapping[lq] = pq
                reverse_mapping[pq] = lq

    # ---- Step 4: identity + leftover fallback for any unmapped logical qubits ----
    used_phys_final = {p for p in mapping if p != -1}
    available_set = set(range(N)) - used_phys_final

    for lq in range(N):
        if mapping[lq] == -1 and lq in available_set:
            mapping[lq] = lq
            reverse_mapping[lq] = lq
            available_set.discard(lq)

    remaining_unmapped = [q for q in range(N) if mapping[q] == -1]
    remaining_avail = sorted(available_set)
    for lq, pq in zip(remaining_unmapped, remaining_avail):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)