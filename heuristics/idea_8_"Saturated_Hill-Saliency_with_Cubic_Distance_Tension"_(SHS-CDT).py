# Strategy: "Saturated Hill-Saliency with Cubic Distance Tension" (SHS-CDT)
# Intuition: This heuristic employs a Hill-type saturation curve (borrowed from enzyme kinetics) to weight gate criticality, which prevents a single massive dependency chain from completely over-powering the spatial constraints of the hardware. By pairing this with a cubic distance penalty ($d^3$), the router becomes "aggressive" toward resolving long-range qubit separations, ensuring that distant logical pairs are prioritized for closure before fine-tuning nearby ones.
# Stats: {'mean_swaps': 1254.2727272727273, 'mean_depth': 1334.6363636363637, 'mean_runtime': 3.902772058140148, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    p1, p2 = swap_gate
    
    # 1. Front Layer Contribution (Immediate Priority)
    front_score = 0.0
    for gate_id in self.front_layer:
        logical_qubits = self.access2q[gate_id]
        if len(logical_qubits) == 2:
            l1, l2 = logical_qubits
            p_l1 = self.temp_mapping_dict[l1]
            p_l2 = self.temp_mapping_dict[l2]
            
            dist = self.distance_matrix[p_l1][p_l2]
            
            # Hill-function saturation for saliency (K=10)
            # This squashes the criticality score into a [0, 1] range
            saliency = self.dag_dependencies_count[gate_id]
            weight = saliency / (saliency + 10.0) if saliency > 0 else 0.05
            
            # Cubic tension to penalize long-range outliers heavily
            front_score += weight * (dist ** 3)
            
    # 2. Extended Layer Contribution (Lookahead strategy)
    extended_score = 0.0
    for gate_id in self.extended_layer:
        logical_qubits = self.access2q[gate_id]
        if len(logical_qubits) == 2:
            l1, l2 = logical_qubits
            p_l1 = self.temp_mapping_dict[l1]
            p_l2 = self.temp_mapping_dict[l2]
            
            dist = self.distance_matrix[p_l1][p_l2]
            depth = self.extended_layer_index[gate_id]
            
            saliency = self.dag_dependencies_count[gate_id]
            weight = saliency / (saliency + 10.0) if saliency > 0 else 0.05
            
            # Quadratic depth decay to prevent future noise from drowning out the front layer
            # Cubic distance maintained for consistency in geometric attraction
            extended_score += (weight * (dist ** 3)) / ((depth + 1) ** 2)
            
    total_raw_score = front_score + extended_score
    
    # 3. Heat-Aware Anti-Chatter Multiplier
    # Incorporates physical qubit health (decay) as a multiplicative penalty
    # to avoid oscillating swaps on noisy or over-used hardware nodes.
    avg_heat = (self.decay_parameter[p1] + self.decay_parameter[p2]) / 2.0
    heat_multiplier = 1.0 + (0.05 * avg_heat)
    
    return total_raw_score * heat_multiplier