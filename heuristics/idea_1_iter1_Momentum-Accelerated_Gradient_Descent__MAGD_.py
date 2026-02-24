# Idea: Momentum-Accelerated Gradient Descent (MAGD)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    # Momentum-Accelerated Gradient Descent (MAGD) - Robust Implementation
    # Addresses crashes by adding defensive guards for unmapped qubits (-1),
    # verifying gate qubit counts, and ensuring safe array access.
    # Maintains the core logic: rewarding consistent gradient descent across layers.

    p1, p2 = swap_gate

    # 1. Identify logical qubits involved (state BEFORE swap)
    # self.reverse_mapping_dict is [physical -> logical]
    # It might contain -1 for empty physical qubits
    q1 = self.reverse_mapping_dict[p1]
    q2 = self.reverse_mapping_dict[p2]

    # Cache references for performance
    temp_map = self.temp_mapping_dict
    dist_mat = self.distance_matrix
    access2q = self.access2q
    dag_dep = self.dag_dependencies_count
    ext_idx = self.extended_layer_index

    # --- Front Layer ---
    front_cost = 0.0
    d_front_imp = 0.0
    relevant_front_count = 0

    for g in self.front_layer:
        # Guard: Ensure gate involves exactly 2 qubits (handle 1-qubit/multi-qubit cases)
        qubits = access2q[g]
        if len(qubits) != 2:
            continue

        qa, qb = qubits

        # Guard: Ensure logical qubits are mapped to valid physical qubits
        pa = temp_map[qa]
        pb = temp_map[qb]

        if pa == -1 or pb == -1:
            continue

        d_new = dist_mat[pa][pb]

        # Guard: Safe access to criticality score
        crit = 0.0
        if 0 <= g < len(dag_dep):
            crit = dag_dep[g]

        front_cost += d_new * (1.0 + crit * 0.1)

        # Momentum Delta: Calculate only if this gate involves the swapped qubits
        if qa == q1 or qb == q1 or qa == q2 or qb == q2:
            # Determine OLD physical positions to compare gradients
            # If qa is q1, it WAS at p1. If qa is q2, it WAS at p2. Otherwise it stayed at pa.
            pa_old = p1 if qa == q1 else (p2 if qa == q2 else pa)
            pb_old = p1 if qb == q1 else (p2 if qb == q2 else pb)

            d_old = dist_mat[pa_old][pb_old]
            d_front_imp += (d_new - d_old) # Negative delta implies improvement
            relevant_front_count += 1

    # --- Extended Layer ---
    extended_cost = 0.0
    d_ext_imp = 0.0
    relevant_ext_count = 0

    for g in self.extended_layer:
        qubits = access2q[g]
        if len(qubits) != 2:
            continue

        qa, qb = qubits
        pa = temp_map[qa]
        pb = temp_map[qb]

        if pa == -1 or pb == -1:
            continue

        d_new = dist_mat[pa][pb]

        depth = ext_idx.get(g, 0)
        # Weight decay for lookahead
        w = 0.5 ** (depth / 5.0)

        extended_cost += d_new * w

        if qa == q1 or qb == q1 or qa == q2 or qb == q2:
            pa_old = p1 if qa == q1 else (p2 if qa == q2 else pa)
            pb_old = p1 if qb == q1 else (p2 if qb == q2 else pb)

            d_old = dist_mat[pa_old][pb_old]
            d_ext_imp += (d_new - d_old) * w
            relevant_ext_count += 1

    # --- Momentum / Consistency Logic ---

    avg_front = 0.0
    if relevant_front_count > 0:
        avg_front = d_front_imp / relevant_front_count

    avg_ext = 0.0
    if relevant_ext_count > 0:
        avg_ext = d_ext_imp / relevant_ext_count

    momentum_score = 0.0

    # Use epsilon threshold for float comparison
    # "Acceleration": Both layers improved (coherent gradient)
    if avg_front < -1e-9 and avg_ext < -1e-9:
        momentum_score = -2.0 * (abs(avg_front) + abs(avg_ext))
    # "Jitter": Gradients oppose each other (incoherent)
    elif (avg_front < -1e-9 and avg_ext > 1e-9) or (avg_ext < -1e-9 and avg_front > 1e-9):
        momentum_score = 1.5 * (abs(avg_front) + abs(avg_ext))

    # Final Normalization
    n_front = len(self.front_layer)
    n_ext = len(self.extended_layer)

    norm_front = front_cost / n_front if n_front > 0 else 0.0
    norm_ext = extended_cost / n_ext if n_ext > 0 else 0.0

    return norm_front + 0.5 * norm_ext + momentum_score