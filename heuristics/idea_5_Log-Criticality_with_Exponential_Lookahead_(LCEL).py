# Strategy: Log-Criticality with Exponential Lookahead (LCEL)
# Intuition: This heuristic applies a logarithmic transform to gate criticality to prevent deep dependency chains from over-dominating routing decisions, ensuring parallel branches are not starved. It uses an exponential decay for lookahead gates to provide a focused guidance window, weighted by the physical qubit usage heat (decay_parameter) to promote load balancing across the topology.
# Stats: {'mean_swaps': 535.3181818181819, 'mean_depth': 940.4545454545455, 'mean_runtime': 1.5161106369712136, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    f_size = len(self.front_layer)
    e_size = len(self.extended_layer)
    
    # 1. Front Layer: Quadratic Tension with Log-Criticality
    # We use math.log(crit + 2) to dampen the influence of massive downstream 
    # closures, ensuring a more balanced priority across parallel gate layers.
    f_cost = 0
    for g in self.front_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        # Log-scaling criticality weight (always >= log(2) > 0)
        crit_weight = math.log(self.dag_dependencies_count[g] + 2)
        
        # Quadratic distance creates high tension for immediate gates
        f_cost += crit_weight * (dist ** 2)

    # 2. Extended Layer: Exponential Lookahead Decay
    # Unlike harmonic decay, exponential decay (0.5^depth) creates a sharper 
    # focus on the immediate successors of the front layer.
    e_cost = 0
    for g in self.extended_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        depth = self.extended_layer_index.get(g, 0)
        crit_weight = math.log(self.dag_dependencies_count[g] + 2)
        
        # Exponentially decay the pull of future interactions
        e_cost += (crit_weight * dist) * (0.5 ** depth)

    # 3. Normalization and Physical Resistance
    # max_decay incorporates the usage 'heat' of the physical qubits involved 
    # in the candidate swap, discouraging congestion on overused hardware nodes.
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    h_front = (f_cost / f_size) if f_size > 0 else 0
    h_ext = (e_cost / e_size) if e_size > 0 else 0
    
    # Return total heuristic cost (minimization)
    return max_decay * (h_front + h_ext)