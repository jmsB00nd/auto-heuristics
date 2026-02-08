# Strategy: "Gradient Flow Potential"
# Intuition: Model the routing problem as a potential field where each gate creates an attractive force proportional to its criticality. The cost measures the "energy" required to reach equilibrium - SWAPs that align qubits along the gradient of maximum criticality flow should minimize total work, penalizing moves that increase potential energy in high-dependency regions.
# Stats: {'mean_swaps': 810.6777777777778, 'mean_depth': 1133.0777777777778, 'mean_runtime': 1.5284287982516818, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Compute the "potential field" contribution from each gate
    # weighted by how much downstream work depends on it
    
    p1, p2 = swap_gate
    swap_heat = self.decay_parameter[p1] * self.decay_parameter[p2]
    
    # Front layer: compute gradient magnitude (criticality * inverse distance)
    front_potential = 0.0
    front_gradient_alignment = 0.0
    
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        crit = self.dag_dependencies_count[g] + 1
        
        # Potential energy: criticality creates a "well" that must be resolved
        front_potential += crit * (dist ** 2)
        
        # Check if swap moves along the gradient (toward resolving high-crit gates)
        # by measuring if swapped qubits are on the path
        dist_p1_P1 = self.distance_matrix[p1][P1]
        dist_p1_P2 = self.distance_matrix[p1][P2]
        dist_p2_P1 = self.distance_matrix[p2][P1]
        dist_p2_P2 = self.distance_matrix[p2][P2]
        
        # Gradient alignment: positive if swap helps, negative if hurts
        min_before = min(dist_p1_P1 + dist_p2_P2, dist_p1_P2 + dist_p2_P1)
        involvement = 1.0 / (1.0 + min_before)
        front_gradient_alignment += crit * involvement * dist
    
    # Extended layer: diminishing potential with depth
    extended_potential = 0.0
    total_future_crit = 0
    
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        depth = self.extended_layer_index.get(g, 1) + 1
        crit = self.dag_dependencies_count[g] + 1
        
        # Potential decays with depth but scales with criticality
        decay_factor = 1.0 / (depth ** 1.5)
        extended_potential += crit * dist * decay_factor
        total_future_crit += crit * decay_factor
    
    # Normalize by layer sizes
    n_front = len(self.front_layer) if self.front_layer else 1
    n_extended = len(self.extended_layer) if self.extended_layer else 1
    
    # Compute "flow resistance" - how much this swap fights the natural gradient
    # High criticality gates should have low resistance paths
    flow_resistance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        crit = self.dag_dependencies_count[g] + 1
        
        # If either physical qubit in swap is involved, measure resistance
        if P1 in swap_gate or P2 in swap_gate:
            dist = self.distance_matrix[P1][P2]
            flow_resistance += crit * (dist - 1)  # -1 because dist=1 is optimal
    
    # Final cost: potential energy + flow resistance, modulated by heat
    H = (swap_heat ** 0.5) * (
        (front_potential + front_gradient_alignment) / n_front +
        0.5 * extended_potential / n_extended +
        0.3 * flow_resistance
    )
    
    return H