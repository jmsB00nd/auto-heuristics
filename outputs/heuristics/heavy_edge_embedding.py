def init_mapping(self):
    num_qubits = self.num_qubits

    self.mapping_dict = [-1] * num_qubits
    self.reverse_mapping_dict = [-1] * num_qubits

    edge_weight = {}
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            try:
                w = self.qubit_interaction_graph[key[0]][key[1]]
            except Exception:
                w = 0
            if not w:
                edge_weight[key] = edge_weight.get(key, 0) + 1
            else:
                if key not in edge_weight:
                    edge_weight[key] = w

    logical_edges = sorted(edge_weight.items(), key=lambda kv: -kv[1])

    centrality = getattr(self, "physical_centrality", {}) or {}

    def cscore(p):
        return centrality.get(p, 0.0)

    seen_hw = set()
    hw_edges = []
    for (u, v) in self.backend_connections:
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen_hw:
            continue
        seen_hw.add(key)
        hw_edges.append(key)
    hw_edges.sort(key=lambda e: -(cscore(e[0]) + cscore(e[1])))

    used_phys = set()
    placed_log = {}

    def free_neighbor_edges(phys):
        candidates = []
        for nb in self.backend.get(phys, ()):
            if nb in used_phys:
                continue
            candidates.append((nb, cscore(nb)))
        candidates.sort(key=lambda x: -x[1])
        return [c[0] for c in candidates]

    for (lu, lv), _w in logical_edges:
        u_placed = lu in placed_log
        v_placed = lv in placed_log

        if u_placed and v_placed:
            continue

        if not u_placed and not v_placed:
            chosen = None
            for (pa, pb) in hw_edges:
                if pa in used_phys or pb in used_phys:
                    continue
                chosen = (pa, pb)
                break
            if chosen is None:
                continue
            pa, pb = chosen
            if cscore(pa) >= cscore(pb):
                placed_log[lu] = pa
                placed_log[lv] = pb
            else:
                placed_log[lu] = pb
                placed_log[lv] = pa
            used_phys.add(pa)
            used_phys.add(pb)
        else:
            if u_placed:
                anchor_log, anchor_phys, free_log = lu, placed_log[lu], lv
            else:
                anchor_log, anchor_phys, free_log = lv, placed_log[lv], lu
            neighbors = free_neighbor_edges(anchor_phys)
            if not neighbors:
                continue
            target = neighbors[0]
            placed_log[free_log] = target
            used_phys.add(target)

    logical_in_circuit = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_in_circuit.add(q)

    remaining_logical = [q for q in logical_in_circuit if q not in placed_log]
    activity = getattr(self, "logical_activity", {}) or {}
    remaining_logical.sort(key=lambda q: -activity.get(q, 0))

    free_phys_sorted = sorted(
        (p for p in range(num_qubits) if p not in used_phys),
        key=lambda p: -cscore(p),
    )

    for lq in remaining_logical:
        if not free_phys_sorted:
            break
        target = free_phys_sorted.pop(0)
        placed_log[lq] = target
        used_phys.add(target)

    for lq, pq in placed_log.items():
        if 0 <= lq < num_qubits and 0 <= pq < num_qubits:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    unused_logical = [q for q in range(num_qubits) if self.mapping_dict[q] == -1]
    unused_physical = [p for p in range(num_qubits) if self.reverse_mapping_dict[p] == -1]

    identity_first = [q for q in unused_logical if q in unused_physical and self.reverse_mapping_dict[q] == -1 and self.mapping_dict[q] == -1]
    identity_set = set(identity_first)
    for q in identity_first:
        self.mapping_dict[q] = q
        self.reverse_mapping_dict[q] = q

    unused_logical = [q for q in range(num_qubits) if self.mapping_dict[q] == -1]
    unused_physical = [p for p in range(num_qubits) if self.reverse_mapping_dict[p] == -1]

    for lq, pq in zip(unused_logical, unused_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)