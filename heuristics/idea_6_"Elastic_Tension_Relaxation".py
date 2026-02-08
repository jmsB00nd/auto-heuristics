# Strategy: "Elastic Tension Relaxation"
# Intuition: Model each pending gate as a spring connecting its logical qubits. The "elastic potential energy" of the system is the sum of squared distances (Hooke's law). A good SWAP should reduce total system tension, with criticality acting as spring stiffness (more critical gates = stronger springs that dominate the optimization).
# Stats: {'mean_swaps': 573.4222222222222, 'mean_depth': 939.1, 'mean_runtime': 1.1204782432980007, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    """
    Elastic Tension Relaxation: Minimize total "spring potential energy" 
    where each gate is a spring with stiffness proportional to criticality.
    Energy = sum(k * d^2) where k = criticality, d = distance.
    """
    total_tension = 0.0
    
    # Process front layer with highest priority (stiffness multiplier)
    front_layer_stiffness = 3.0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1][p2]
        
        # Criticality as spring constant
        criticality = self.dag_dependencies_count[gate_id] + 1
        
        # Quadratic potential energy (Hooke's law: E = k * x^2)
        tension = front_layer_stiffness * criticality * (dist ** 2)
        total_tension += tension
    
    # Process extended layer with depth-decayed stiffness
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1][p2]
        depth = self.extended_layer_index.get(gate_id, 1)
        
        criticality = self.dag_dependencies_count[gate_id] + 1
        
        # Stiffness decays exponentially with depth (distant future matters less)
        depth_decay = 1.0 / (1.0 + depth)
        
        # Quadratic tension with decay
        tension = depth_decay * criticality * (dist ** 2)
        total_tension += tension
    
    # Add "thermal noise" penalty from decay parameter
    # Swapping hot qubits adds friction to the system
    phys_q1, phys_q2 = swap_gate
    thermal_friction = 0.5 * (self.decay_parameter[phys_q1] + self.decay_parameter[phys_q2])
    
    # Compute "relaxation gradient" - how much tension is released
    # by checking if swap brings any front-layer pairs closer
    relaxation_bonus = 0.0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        
        # Bonus for achieving adjacency (distance = 1 means executable)
        if self.distance_matrix[p1][p2] == 1:
            relaxation_bonus += self.dag_dependencies_count[gate_id] + 1
    
    # Final cost: tension + friction - relaxation bonus
    cost = total_tension + thermal_friction - (2.0 * relaxation_bonus)
    
    return cost