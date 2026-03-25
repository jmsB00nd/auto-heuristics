def init_mapping(self):
    """
    Topological Sort + Hardware Centrality BFS Co-Scheduling (TSHCBS)

    Core insight: logical qubits that appear earliest in a topological
    traversal of the gate DAG form the first scheduling bottleneck.
    Placing them at the most central hardware positions maximises their
    routing flexibility and reduces expected SWAP distance for all
    subsequent gates.

    Algorithm
    ---------
    1. Build a lightweight sequential gate DAG from self.access:
       for each logical qubit, add an edge gate[i] -> gate[i+1]
       (program-order dependency chain per qubit).
    2. Run Kahn's BFS topological sort on that DAG.
    3. Record the *first topological appearance index* of every logical
       qubit: the topo-order index of the first gate that touches it.
    4. Sort logical qubits ascending by first-appearance index
       (ties broken by qubit ID for determinism).
    5. Rank physical qubits ascending by *sum-distance centrality*:
       sum of shortest-path distances to all other physical qubits
       (lower = more central in the hardware graph).
    6. Pair: i-th earliest logical qubit -> i-th most central physical qubit.
       Remaining idle logical qubits fill remaining physical positions.
    7. Build a strict bijection with swap-in-place over the identity
       permutation to guarantee mapping_dict and reverse_mapping_dict
       are consistent 1-to-1 maps over all num_qubits indices.
    """
    from collections import defaultdict, deque

    # ── Step 1: Build sequential per-qubit gate DAG ──────────────────────────
    gate_order_per_qubit = defaultdict(list)
    for gate, qubits in sorted(self.access.items()):   # sorted = program order
        for q in qubits:
            gate_order_per_qubit[q].append(gate)

    all_gates = set(self.access.keys())
    dag_succ = defaultdict(set)
    dag_pred_count = {g: 0 for g in all_gates}

    for gates_on_qubit in gate_order_per_qubit.values():
        for i in range(len(gates_on_qubit) - 1):
            src, dst = gates_on_qubit[i], gates_on_qubit[i + 1]
            if dst not in dag_succ[src]:        # avoid double-counting
                dag_succ[src].add(dst)
                dag_pred_count[dst] += 1

    # ── Step 2: Kahn's BFS topological sort ──────────────────────────────────
    queue = deque(sorted(g for g in all_gates if dag_pred_count[g] == 0))
    topo_order = []
    remaining_pred = dict(dag_pred_count)
    while queue:
        gate = queue.popleft()
        topo_order.append(gate)
        for succ in sorted(dag_succ[gate]):     # sorted for determinism
            remaining_pred[succ] -= 1
            if remaining_pred[succ] == 0:
                queue.append(succ)

    # ── Step 3: First topological appearance index per logical qubit ─────────
    qubit_first_topo = {}
    for order_idx, gate in enumerate(topo_order):
        for q in self.access[gate]:
            if q not in qubit_first_topo:
                qubit_first_topo[q] = order_idx

    # ── Step 4: Sort logical qubits by first appearance ──────────────────────
    circuit_qubits = sorted(
        qubit_first_topo.keys(),
        key=lambda q: (qubit_first_topo[q], q)   # topo index, then qubit ID
    )
    idle_qubits = sorted(
        q for q in range(self.num_qubits) if q not in qubit_first_topo
    )
    logical_order = circuit_qubits + idle_qubits  # full permutation domain

    # ── Step 5: Sort physical qubits by sum-distance centrality ─────────────
    # Lower total distance = more central position in hardware topology
    physical_qubits = sorted(self.backend.keys())
    phys_centrality = {
        p: sum(d for d in self.distance_matrix[p] if d != float('inf'))
        for p in physical_qubits
    }
    physical_order = sorted(physical_qubits, key=lambda p: (phys_centrality[p], p))

    # Include any indices in [0, num_qubits) not in the backend (sparse IDs)
    phys_in_backend = set(physical_order)
    extra_phys = sorted(p for p in range(self.num_qubits) if p not in phys_in_backend)
    physical_order = physical_order + extra_phys

    # ── Step 6: Pair logical (topo order) with physical (centrality order) ───
    lq_to_phys = {}
    for i, logical_q in enumerate(logical_order):
        if i < len(physical_order):
            lq_to_phys[logical_q] = physical_order[i]

    # ── Step 7: Build strict bijection via swap-in-place ─────────────────────
    # Start from identity; for each assignment lq -> target_phys, swap the
    # two colliding entries so both dicts stay mutually consistent at all times.
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)