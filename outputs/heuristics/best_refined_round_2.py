def init_mapping(self):
    import collections, math
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    distance_matrix = self.distance_matrix
    num_qubits = self.num_qubits
    backend = self.backend
    access = self.access
    dag_predecessors2q = getattr(self, 'dag_predecessors2q', {}) or {}
    dag2q = getattr(self, 'dag2q', {}) or {}
    dag_dependencies_count = getattr(self, 'dag_dependencies_count', []) or []

    # --- 1. Logical qubits in use ---
    logical_qubits_used = set()
    for qubits in access.values():
        logical_qubits_used.update(qubits)
    if not logical_qubits_used:
        self.mapping_dict = list(range(num_qubits))
        self.reverse_mapping_dict = list(range(num_qubits))
        if getattr(self, 'use_isl', False):
            self.isl_mapping = dict_to_isl_map(dict(enumerate(self.mapping_dict)))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # --- 2. P2 primary: SabreLayout reference anchors ---
    ref_anchor = {}
    try:
        circuit = QuantumCircuit.from_qasm_str(self.data["qasm_code"])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap(list(self.backend_connections))
        sl = SabreLayout(coupling_map, seed=21)
        sl.run(dag_circuit)
        layout = sl.property_set["layout"]
        for v, pq in layout._v2p.items():
            if v._register.name == "ancilla":
                continue
            lq = v._index
            if 0 <= lq < num_qubits and 0 <= pq < num_qubits:
                ref_anchor[lq] = pq
    except Exception:
        ref_anchor = {}

    # --- 3. P1 primary: floored depth-decay + frontier-boosted interaction weights ---
    all_2q_gates = {g: qs for g, qs in access.items() if len(qs) == 2}
    if dag_predecessors2q and dag2q:
        in_count = {g: sum(1 for p in dag_predecessors2q.get(g, set()) if p in all_2q_gates) for g in all_2q_gates}
        gate_depth = {g: 0 for g in all_2q_gates}
        bfs = collections.deque(g for g in all_2q_gates if in_count[g] == 0)
        while bfs:
            g = bfs.popleft()
            d = gate_depth[g]
            for succ in dag2q.get(g, set()):
                if succ not in all_2q_gates:
                    continue
                if d + 1 > gate_depth[succ]:
                    gate_depth[succ] = d + 1
                in_count[succ] -= 1
                if in_count[succ] == 0:
                    bfs.append(succ)
    else:
        gate_depth = {g: 0 for g in all_2q_gates}
    max_depth = max(gate_depth.values(), default=1) or 1

    ALPHA = 4.0
    DEPTH_FLOOR = 0.18  # MUTATION: prevent exp(-ALPHA*d/D) from crushing late-layer weights
    interaction = collections.defaultdict(float)
    for gate_id, qubits in access.items():
        if len(qubits) != 2:
            continue
        q0, q1 = qubits[0], qubits[1]
        crit = dag_dependencies_count[gate_id] if gate_id < len(dag_dependencies_count) else 1
        depth = gate_depth.get(gate_id, 0)
        frontier = not dag_predecessors2q.get(gate_id)
        decay = max(math.exp(-ALPHA * depth / max_depth), DEPTH_FLOOR)
        w = math.sqrt(max(crit, 1)) * decay * (4.0 if frontier else 1.0)
        interaction[(q0, q1)] += w
        interaction[(q1, q0)] += w

    logical_degree = collections.defaultdict(float)
    for (l0, l1), w in interaction.items():
        if l0 < l1:
            logical_degree[l0] += w
            logical_degree[l1] += w

    # --- 4. CROSSOVER: anchor-referenced L×N cost + Hungarian matching ---
    logical_list = sorted(logical_qubits_used)
    L = len(logical_list)
    lq_index = {lq: i for i, lq in enumerate(logical_list)}

    SELF_ANCHOR_WEIGHT = 0.5
    neighbor_anchors = {}
    for lq in logical_list:
        entries = []
        for olq in logical_list:
            if olq == lq:
                continue
            w = interaction.get((lq, olq), 0.0)
            if w > 0 and olq in ref_anchor:
                entries.append((ref_anchor[olq], w))
        neighbor_anchors[lq] = entries

    cost = [[0.0] * num_qubits for _ in range(L)]
    for i, lq in enumerate(logical_list):
        na = neighbor_anchors[lq]
        own_a = ref_anchor.get(lq, -1)
        for pq in range(num_qubits):
            c = 0.0
            for apq, w in na:
                c += w * distance_matrix[pq][apq]
            if own_a >= 0:
                c += SELF_ANCHOR_WEIGHT * distance_matrix[pq][own_a]
            cost[i][pq] = c

    placed, placed_rev = {}, {}
    matched = False
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cm = np.array(cost, dtype=float)
        row_ind, col_ind = linear_sum_assignment(cm)
        for i, pq in zip(row_ind, col_ind):
            lq = logical_list[int(i)]
            placed[lq] = int(pq)
            placed_rev[int(pq)] = lq
        matched = True
    except Exception:
        matched = False

    if not matched:
        remaining_lq = set(logical_list)
        remaining_pq = set(range(num_qubits))
        while remaining_lq:
            best_regret = -1.0
            best_choice = None
            for lq in remaining_lq:
                i = lq_index[lq]
                pairs = sorted((cost[i][pq], pq) for pq in remaining_pq)
                if not pairs:
                    break
                regret = (pairs[1][0] - pairs[0][0]) if len(pairs) > 1 else 0.0
                if regret > best_regret or best_choice is None:
                    best_regret = regret
                    best_choice = (lq, pairs[0][1])
            lq, pq = best_choice
            placed[lq] = pq
            placed_rev[pq] = lq
            remaining_lq.remove(lq)
            remaining_pq.remove(pq)

    # --- 5. P1 refinement: bounded unbiased 2-opt polish ---
    ll = sorted(placed.keys(), key=lambda lq: -logical_degree.get(lq, 0.0))
    nbr = {lq: [(olq, interaction[(lq, olq)]) for olq in ll
                if lq != olq and interaction[(lq, olq)] > 0] for lq in ll}
    nbr_sets = {lq: {olq for olq, _ in nbr[lq]} for lq in ll}
    swap_pairs = [(l0, l1) for i, l0 in enumerate(ll) for l1 in ll[i + 1:]
                  if l1 in nbr_sets[l0] or (nbr_sets[l0] & nbr_sets[l1])]

    for _ in range(40):
        improved = False
        for l0, l1 in swap_pairs:
            p0, p1 = placed[l0], placed[l1]
            delta = 0.0
            for lq in ll:
                if lq == l0 or lq == l1:
                    continue
                pq = placed[lq]
                w0 = interaction.get((l0, lq), 0.0)
                w1 = interaction.get((l1, lq), 0.0)
                if w0:
                    delta += w0 * (distance_matrix[p1][pq] - distance_matrix[p0][pq])
                if w1:
                    delta += w1 * (distance_matrix[p0][pq] - distance_matrix[p1][pq])
            if delta < -1e-9:
                placed[l0], placed[l1] = p1, p0
                placed_rev[p0], placed_rev[p1] = l1, l0
                improved = True
        for lq in ll:
            pq_cur = placed[lq]
            cost_cur = sum(w * distance_matrix[pq_cur][placed[olq]] for olq, w in nbr[lq] if olq in placed)
            best_pq, best_cost_lq = pq_cur, cost_cur
            for pq_new in range(num_qubits):
                if pq_new == pq_cur or pq_new in placed_rev:
                    continue
                c = sum(w * distance_matrix[pq_new][placed[olq]] for olq, w in nbr[lq] if olq in placed)
                if c < best_cost_lq - 1e-9:
                    best_cost_lq = c
                    best_pq = pq_new
            if best_pq != pq_cur:
                del placed_rev[pq_cur]
                placed[lq] = best_pq
                placed_rev[best_pq] = lq
                improved = True
        if not improved:
            break

    # --- 6. Assemble injective list mappings ---
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    for lq, pq in placed.items():
        if 0 <= lq < num_qubits:
            mapping[lq] = pq
            reverse_mapping[pq] = lq
    used_phys = {pq for pq in mapping if pq >= 0}
    free_phys = [p for p in range(num_qubits) if p not in used_phys]
    fi = 0
    for lq in range(num_qubits):
        if mapping[lq] < 0:
            mapping[lq] = free_phys[fi]
            reverse_mapping[free_phys[fi]] = lq
            fi += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping
    if getattr(self, 'use_isl', False):
        self.isl_mapping = dict_to_isl_map(dict(enumerate(self.mapping_dict)))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)