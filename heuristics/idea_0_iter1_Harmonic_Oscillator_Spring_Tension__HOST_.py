# Idea: Harmonic Oscillator Spring Tension (HOST)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on qugan_n71__72CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    """
    Implements the Harmonic Oscillator Spring Tension (HOST) heuristic.
    Optimized to eliminate O(N) list slicing overhead and reduce lookahead cost.
    Treats connections as springs (U = k * x^2) where stiffness k depends on 
    criticality and decays with temporal depth.
    """
    # Localize references for speed
    tm = self.temp_mapping_dict
    dm = self.distance_matrix
    a2q = self.access2q
    ddc = self.dag_dependencies_count
    eli = self.extended_layer_index

    cost = 0.0

    # Front Layer: Immediate gates act as stiff springs
    for gid in self.front_layer:
        qs = a2q[gid]
        p1, p2 = tm[qs[0]], tm[qs[1]]

        # Verify logical qubits are mapped
        if p1 != -1 and p2 != -1:
            d = dm[p1][p2]
            # Stiff spring constant proportional to criticality
            k = 1.0 + ddc[gid]
            cost += k * (d * d)

    # Extended Layer: Future gates act as relaxed springs
    # Optimization: Use counter break instead of slicing to avoid copying the list
    count = 0
    max_lookahead = 15  # Tighter limit to ensure convergence

    for gid in self.extended_layer:
        if count >= max_lookahead:
            break

        qs = a2q[gid]
        p1, p2 = tm[qs[0]], tm[qs[1]]

        if p1 != -1 and p2 != -1:
            d = dm[p1][p2]
            # Stiffness decays with lookahead depth (Time)
            # Use get() as future gates might not be indexed yet
            depth = eli.get(gid, 0)
            k = (1.0 + ddc[gid]) / (1.0 + depth)
            cost += k * (d * d)
            count += 1

    return cost