# Strategy: Topological Flow Resistance
# Intuition: Model each pending gate as a current source demanding flow between its qubits. The "resistance" of a SWAP is proportional to how much it increases total circuit resistance (distance) weighted by the power (dependency count squared) each gate must deliver. Lower resistance configurations allow more efficient "current flow" through the quantum circuit.
# Stats: {'mean_swaps': 508.1363636363636, 'mean_depth': 917.0454545454545, 'mean_runtime': 1.890182159163735, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Treat routing as minimizing electrical resistance in a circuit
    # Each gate creates a "current demand" proportional to its criticality^2 (power law)
    # Distance acts as resistance; we want minimal total power dissipation
    
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    
    # Compute "conductance" improvement for front layer (highest priority)
    front_power_dissipation = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[P1][P2]
        deps = self.dag_dependencies_count[g]
        
        # Power = I^2 * R, where I ~ sqrt(deps+1), R ~ dist
        # So power ~ (deps+1) * dist
        # But we use cubic root of deps for diminishing returns on very high criticality
        current = (deps + 1) ** 0.333
        resistance = dist
        front_power_dissipation += (current ** 2) * resistance
    
    # Extended layer contributes with exponential depth decay (signal attenuation)
    extended_power_dissipation = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[P1][P2]
        deps = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0) + 1
        
        # Attenuation factor: signal weakens exponentially with depth
        attenuation = 0.7 ** depth
        current = (deps + 1) ** 0.333
        resistance = dist
        extended_power_dissipation += attenuation * (current ** 2) * resistance
    
    # Normalize by layer sizes
    front_term = front_power_dissipation / front_layer_size if front_layer_size else 0
    extended_term = extended_power_dissipation / extended_layer_size if extended_layer_size else 0
    
    # SWAP "switching cost" based on qubit heat (like thermal noise in electronics)
    thermal_noise = (self.decay_parameter[swap_gate[0]] + self.decay_parameter[swap_gate[1]]) / 2.0
    
    # Combine: total power dissipation with thermal penalty
    # Front layer weighted 3x more than extended (immediate priorities)
    H = thermal_noise * (3.0 * front_term + extended_term)
    
    return H