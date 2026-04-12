def init_mapping(self):
    from collections import defaultdict, deque
    import math
    import random

    num_qubits = self.num_qubits
    distance_matrix = self.distance_matrix
    backend = self.backend

    # Extract logical qubits and 2-qubit gate interactions
    logical_qubit_set = set()
    two_qubit_gates = []
    interaction_weight = defaultdict(float)

    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            two_qubit_gates.append((gate_id, q1, q2))
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(backend.keys())
    n_logical = len(logical_qubits)

    # Trivial case
    if n_logical == 0 or len(two_qubit_gates) == 0:
        self.mapping_dict = list(range(num_qubits))
        self.reverse_mapping_dict = list(range(num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 1: Build simple DAG and assign layers via topological sort ---
    # Build predecessor/successor for 2q gates only
    gate_order = {}
    for idx, (gid, q1, q2) in enumerate(two_qubit_gates):
        gate_order[gid] = idx

    # Track last gate per qubit for dependency chain
    last_gate_on_qubit = {}
    successors = defaultdict(set)
    predecessors = defaultdict(set)

    for gid, q1, q2 in two_qubit_gates:
        for q in [q1, q2]:
            if q in last_gate_on_qubit:
                prev = last_gate_on_qubit[q]
                if prev != gid:
                    successors[prev].add(gid)
                    predecessors[gid].add(prev)
            last_gate_on_qubit[q] = gid

    # Topological layering via BFS
    all_2q_gate_ids = [gid for gid, _, _ in two_qubit_gates]
    in_degree = {gid: len(predecessors[gid]) for gid in all_2q_gate_ids}
    gate_layer = {}
    queue = deque()
    for gid in all_2q_gate_ids:
        if in_degree[gid] == 0:
            queue.append(gid)
            gate_layer[gid] = 0

    max_layer = 0
    while queue:
        gid = queue.popleft()
        curr_layer = gate_layer[gid]
        for succ in successors[gid]:
            in_degree[succ] -= 1
            if succ in gate_layer:
                gate_layer[succ] = max(gate_layer[succ], curr_layer + 1)
            else:
                gate_layer[succ] = curr_layer + 1
            if in_degree[succ] == 0:
                queue.append(succ)
                max_layer = max(max_layer, gate_layer[succ])

    # --- Step 2: Build overlapping layer-window cohorts ---
    W = 8
    stride = 4
    cohorts = []  # list of (set of logical qubits, interaction weight within window)
    cohort_interactions = []  # list of defaultdict(float) per cohort

    layer_start = 0
    while layer_start <= max_layer:
        layer_end = min(layer_start + W - 1, max_layer)
        cohort_qubits = set()
        cohort_weight = defaultdict(float)
        total_weight = 0.0

        for gid, q1, q2 in two_qubit_gates:
            gl = gate_layer.get(gid, 0)
            if layer_start <= gl <= layer_end:
                cohort_qubits.add(q1)
                cohort_qubits.add(q2)
                key = (min(q1, q2), max(q1, q2))
                cohort_weight[key] += 1.0
                total_weight += 1.0

        if cohort_qubits:
            cohorts.append(sorted(cohort_qubits))
            cohort_interactions.append((cohort_weight, total_weight))

        layer_start += stride
        if layer_end == max_layer:
            break

    if not cohorts:
        cohorts.append(logical_qubits)
        cohort_interactions.append((interaction_weight, sum(interaction_weight.values())))

    # --- Step 3: Multi-seed greedy construction ---
    # Build logical adjacency weighted graph
    lq_neighbors = defaultdict(list)
    for (q1, q2), w in interaction_weight.items():
        lq_neighbors[q1].append((q2, w))
        lq_neighbors[q2].append((q1, w))

    # Sort logical qubits by total interaction weight (descending)
    lq_total_weight = {}
    for lq in logical_qubits:
        lq_total_weight[lq] = sum(w for _, w in lq_neighbors[lq])
    sorted_lq = sorted(logical_qubits, key=lambda q: lq_total_weight[q], reverse=True)

    # Physical qubit centrality (lower = more central)
    phys_degree = {pq: len(backend[pq]) for pq in physical_qubits}

    def cost_of_mapping(mapping):
        """Total weighted distance cost."""
        total = 0.0
        for (q1, q2), w in interaction_weight.items():
            total += w * distance_matrix[mapping[q1]][mapping[q2]]
        return total

    def greedy_construct(seed_lq=None, seed_pq=None):
        """Greedy placement starting from a seed."""
        mapping = list(range(num_qubits))
        reverse_mapping = list(range(num_qubits))
        assigned_physical = set()
        assigned_logical = set()

        def place(lq, pq):
            old_pq = mapping[lq]
            old_lq = reverse_mapping[pq]
            mapping[lq] = pq
            mapping[old_lq] = old_pq
            reverse_mapping[pq] = lq
            reverse_mapping[old_pq] = old_lq
            assigned_physical.add(pq)
            assigned_logical.add(lq)

        if seed_lq is not None and seed_pq is not None:
            place(seed_lq, seed_pq)

        # BFS-like expansion from placed qubits
        lq_queue = deque()
        if seed_lq is not None:
            for neighbor, _ in sorted(lq_neighbors[seed_lq], key=lambda x: -x[1]):
                if neighbor not in assigned_logical:
                    lq_queue.append(neighbor)

        # Add remaining by weight order
        for lq in sorted_lq:
            if lq not in assigned_logical and lq not in set(lq_queue):
                lq_queue.append(lq)

        while lq_queue:
            lq = lq_queue.popleft()
            if lq in assigned_logical:
                continue

            # Find best physical qubit: minimize weighted distance to already-placed neighbors
            best_pq = None
            best_cost = float('inf')

            candidates = [pq for pq in physical_qubits if pq not in assigned_physical]
            for pq in candidates:
                cost = 0.0
                for neighbor, w in lq_neighbors[lq]:
                    if neighbor in assigned_logical:
                        cost += w * distance_matrix[pq][mapping[neighbor]]
                if cost < best_cost:
                    best_cost = cost
                    best_pq = pq

            if best_pq is not None:
                place(lq, best_pq)
                for neighbor, _ in sorted(lq_neighbors[lq], key=lambda x: -x[1]):
                    if neighbor not in assigned_logical:
                        lq_queue.append(neighbor)

        return mapping, reverse_mapping

    # Generate multiple seeds
    num_seeds = min(8, n_logical)
    best_mapping = None
    best_reverse = None
    best_cost = float('inf')

    # Seed 1: highest-weight logical -> highest-degree physical
    top_lqs = sorted_lq[:num_seeds]
    top_pqs = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)[:num_seeds]

    for i in range(num_seeds):
        m, r = greedy_construct(top_lqs[i % len(top_lqs)], top_pqs[i % len(top_pqs)])
        c = cost_of_mapping(m)
        if c < best_cost:
            best_cost = c
            best_mapping = m[:]
            best_reverse = r[:]

    # Also try a few random seeds
    for _ in range(4):
        slq = random.choice(logical_qubits)
        spq = random.choice(physical_qubits)
        m, r = greedy_construct(slq, spq)
        c = cost_of_mapping(m)
        if c < best_cost:
            best_cost = c
            best_mapping = m[:]
            best_reverse = r[:]

    # --- Step 4: RSDIWR-like routing simulation for cohort prioritization ---
    def simulate_routing_swaps(mapping):
        """Quick routing simulation to count swaps per cohort."""
        sim_map = mapping[:]
        sim_rev = [0] * num_qubits
        for i in range(num_qubits):
            sim_rev[sim_map[i]] = i

        cohort_swaps = [0.0] * len(cohorts)

        for gid, q1, q2 in two_qubit_gates:
            pq1 = sim_map[q1]
            pq2 = sim_map[q2]
            if (pq1, pq2) in self.backend_connections or (pq2, pq1) in self.backend_connections:
                continue
            # Count as a swap needed, attribute to cohorts containing this gate
            gl = gate_layer.get(gid, 0)
            for ci, cohort_lqs in enumerate(cohorts):
                cw = cohort_interactions[ci][1]
                if cw == 0:
                    continue
                c_start = ci * stride
                c_end = c_start + W - 1
                if c_start <= gl <= c_end:
                    cohort_swaps[ci] += 1.0

        return cohort_swaps

    cohort_swaps = simulate_routing_swaps(best_mapping)

    # Compute cohort selection weights (combine interaction weight + swap feedback)
    cohort_selection_weights = []
    for ci in range(len(cohorts)):
        base_w = cohort_interactions[ci][1]
        swap_w = cohort_swaps[ci] + 1.0
        # Bias toward early cohorts slightly
        time_bias = 1.0 / (1.0 + 0.1 * ci)
        cohort_selection_weights.append(base_w * swap_w * time_bias)

    total_csw = sum(cohort_selection_weights)
    if total_csw > 0:
        cohort_selection_probs = [w / total_csw for w in cohort_selection_weights]
    else:
        cohort_selection_probs = [1.0 / len(cohorts)] * len(cohorts)

    # --- Step 5: SA refinement with cohort-biased swap selection ---
    current_mapping = best_mapping[:]
    current_reverse = best_reverse[:]
    current_cost = best_cost

    T_init = max(1.0, current_cost * 0.05)
    T_min = 0.001
    cooling_rate = 0.995
    max_iters = min(80000, 200 * n_logical * n_logical)

    T = T_init
    no_improve_count = 0
    best_sa_cost = current_cost
    best_sa_mapping = current_mapping[:]
    best_sa_reverse = current_reverse[:]

    # Precompute cohort member pairs for swap selection
    cohort_pairs = []
    for ci, cohort_lqs in enumerate(cohorts):
        pairs = []
        # Physical qubits of cohort members and their neighbors
        for lq in cohort_lqs:
            pq = current_mapping[lq]
            for npq in backend[pq]:
                pairs.append((pq, npq))
        cohort_pairs.append(pairs)

    def pick_cohort_swap():
        """Pick a swap biased toward a cohort."""
        # Select cohort proportional to weights
        r = random.random()
        cumulative = 0.0
        ci = 0
        for i, p in enumerate(cohort_selection_probs):
            cumulative += p
            if r <= cumulative:
                ci = i
                break

        # Get current physical positions of cohort qubits
        cohort_lqs = cohorts[ci]
        phys_set = set()
        for lq in cohort_lqs:
            phys_set.add(current_mapping[lq])
        # Expand to neighbors
        expanded = set(phys_set)
        for pq in phys_set:
            for npq in backend[pq]:
                expanded.add(npq)

        expanded = list(expanded)
        if len(expanded) < 2:
            return None
        random.shuffle(expanded)
        return (expanded[0], expanded[1])

    def delta_cost(mapping, pq1, pq2):
        """Compute cost change from swapping two physical qubits."""
        lq1 = current_reverse[pq1]
        lq2 = current_reverse[pq2]

        delta = 0.0
        affected = set()
        if lq1 in logical_qubit_set:
            for neighbor, w in lq_neighbors.get(lq1, []):
                if neighbor == lq2:
                    continue
                old_d = distance_matrix[pq1][mapping[neighbor]]
                new_d = distance_matrix[pq2][mapping[neighbor]]
                delta += w * (new_d - old_d)
                affected.add(neighbor)

        if lq2 in logical_qubit_set:
            for neighbor, w in lq_neighbors.get(lq2, []):
                if neighbor == lq1:
                    continue
                if neighbor in affected:
                    old_d = distance_matrix[pq2][mapping[neighbor]]
                    new_d = distance_matrix[pq1][mapping[neighbor]]
                    delta += w * (new_d - old_d)
                else:
                    old_d = distance_matrix[pq2][mapping[neighbor]]
                    new_d = distance_matrix[pq1][mapping[neighbor]]
                    delta += w * (new_d - old_d)

        # Direct interaction between lq1 and lq2 doesn't change (same distance)
        return delta

    for iteration in range(max_iters):
        # Cohort-biased swap selection
        if random.random() < 0.7:
            swap = pick_cohort_swap()
            if swap is None:
                continue
            pq1, pq2 = swap
        else:
            # Global random swap
            pq1 = random.choice(physical_qubits)
            pq2 = random.choice(physical_qubits)

        if pq1 == pq2:
            continue

        d_cost = delta_cost(current_mapping, pq1, pq2)

        if d_cost < 0 or random.random() < math.exp(-d_cost / max(T, 1e-10)):
            # Accept swap
            lq1 = current_reverse[pq1]
            lq2 = current_reverse[pq2]
            current_mapping[lq1] = pq2
            current_mapping[lq2] = pq1
            current_reverse[pq1] = lq2
            current_reverse[pq2] = lq1
            current_cost += d_cost

            if current_cost < best_sa_cost:
                best_sa_cost = current_cost
                best_sa_mapping = current_mapping[:]
                best_sa_reverse = current_reverse[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        T *= cooling_rate

        # Restart if stuck
        if no_improve_count > 2000:
            # Perturbation: random swaps
            for _ in range(max(3, n_logical // 5)):
                rpq1 = random.choice(physical_qubits)
                rpq2 = random.choice(physical_qubits)
                if rpq1 != rpq2:
                    rlq1 = current_reverse[rpq1]
                    rlq2 = current_reverse[rpq2]
                    current_mapping[rlq1] = rpq2
                    current_mapping[rlq2] = rpq1
                    current_reverse[rpq1] = rlq2
                    current_reverse[rpq2] = rlq1
            current_cost = cost_of_mapping(current_mapping)
            T = max(T, T_init * 0.3)
            no_improve_count = 0

            # Re-simulate routing for updated cohort weights
            if iteration % 10000 == 0:
                cohort_swaps = simulate_routing_swaps(current_mapping)
                for ci in range(len(cohorts)):
                    base_w = cohort_interactions[ci][1]
                    swap_w = cohort_swaps[ci] + 1.0
                    time_bias = 1.0 / (1.0 + 0.1 * ci)
                    cohort_selection_weights[ci] = base_w * swap_w * time_bias
                total_csw = sum(cohort_selection_weights)
                if total_csw > 0:
                    cohort_selection_probs = [w / total_csw for w in cohort_selection_weights]

        if T < T_min:
            break

    # --- Step 6: Set final mapping ---
    self.mapping_dict = best_sa_mapping
    self.reverse_mapping_dict = best_sa_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)