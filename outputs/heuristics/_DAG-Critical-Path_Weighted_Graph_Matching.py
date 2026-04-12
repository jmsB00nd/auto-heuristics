def init_mapping(self):
    from collections import defaultdict, deque
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    num_q = self.num_qubits

    # Default trivial mapping
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    # ---------------------------------------------------------------
    # Step 1: Build a lightweight DAG from access & write_dict
    # ---------------------------------------------------------------
    schedule = sorted(self.access.keys())
    successors = defaultdict(set)
    predecessors = defaultdict(set)

    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]

        # RAW: read depends on latest writer
        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)

        # WAW and WAR
        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

    # ---------------------------------------------------------------
    # Step 2: Identify 2-qubit gates and compute critical-path length
    #         through each gate (longest path passing through it)
    # ---------------------------------------------------------------
    two_q_gates = [g for g in schedule if len(self.access[g]) == 2]

    if not two_q_gates:
        # No 2-qubit gates; trivial mapping suffices
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Forward pass: longest path from any source to each node
    dist_from_start = defaultdict(int)
    for node in schedule:
        for s in successors[node]:
            if dist_from_start[s] < dist_from_start[node] + 1:
                dist_from_start[s] = dist_from_start[node] + 1

    # Backward pass: longest path from each node to any sink
    dist_to_end = defaultdict(int)
    for node in reversed(schedule):
        for p in predecessors[node]:
            if dist_to_end[p] < dist_to_end[node] + 1:
                dist_to_end[p] = dist_to_end[node] + 1

    # Critical-path length through gate g = dist_from_start[g] + 1 + dist_to_end[g]
    # ---------------------------------------------------------------
    # Step 3: Build logical interaction graph with critical-path weights
    # ---------------------------------------------------------------
    interaction_weight = defaultdict(float)
    logical_qubits_used = set()

    for g in two_q_gates:
        q1, q2 = self.access[g]
        logical_qubits_used.add(q1)
        logical_qubits_used.add(q2)
        cp_len = dist_from_start[g] + 1 + dist_to_end[g]
        edge = (min(q1, q2), max(q1, q2))
        interaction_weight[edge] += cp_len

    # Build adjacency with weights for logical interaction graph
    logical_neighbors = defaultdict(dict)  # logical_neighbors[q][k] = weight
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # ---------------------------------------------------------------
    # Step 4: Build closeness matrix for hardware graph
    #         closeness(pa, pb) = 1 / distance(pa, pb) for pa != pb
    # ---------------------------------------------------------------
    dist_matrix = self.distance_matrix
    physical_nodes = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 5: Build cost matrix using iterative approximation
    #         For each logical qubit i and physical qubit j:
    #         C[i][j] = sum over neighbors k of i:
    #             interaction_weight(i,k) * max closeness(j, any physical)
    #         We approximate "best_candidate_for_k" by using the physical
    #         qubit closest to j (highest closeness) for each neighbor.
    # ---------------------------------------------------------------
    logical_list = sorted(logical_qubits_used)
    n_logical = len(logical_list)
    n_physical = len(physical_nodes)

    # For the cost matrix, we want to MAXIMIZE benefit, but
    # linear_sum_assignment minimizes. So we negate.
    cost = np.zeros((n_logical, n_physical), dtype=np.float64)

    for i_idx, lq in enumerate(logical_list):
        neighbors = logical_neighbors.get(lq, {})
        if not neighbors:
            continue
        for j_idx, pq in enumerate(physical_nodes):
            score = 0.0
            for k, w in neighbors.items():
                # Best closeness for neighbor k: closest physical neighbor of pq
                # Since we don't know where k will land yet, approximate with
                # the best (closest) neighbor of pq on hardware
                best_closeness = 0.0
                for pn in self.backend[pq]:
                    c = 1.0 / max(dist_matrix[pq][pn], 1)
                    if c > best_closeness:
                        best_closeness = c
                # Weight by interaction strength
                score += w * best_closeness
            cost[i_idx][j_idx] = -score  # negate for minimization

    # ---------------------------------------------------------------
    # Step 6: Solve assignment with Hungarian algorithm
    # ---------------------------------------------------------------
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the mapping
    mapping_dict = list(range(num_q))
    reverse_mapping_dict = [-1] * num_q

    # First, assign the matched logical qubits
    assigned_physical = set()
    for r, c in zip(row_ind, col_ind):
        lq = logical_list[r]
        pq = physical_nodes[c]
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
        assigned_physical.add(pq)

    # Assign remaining logical qubits (not in any 2-qubit gate) to
    # remaining physical qubits
    remaining_physical = [p for p in physical_nodes if p not in assigned_physical]
    remaining_logical = [q for q in range(num_q) if q not in logical_qubits_used]

    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)