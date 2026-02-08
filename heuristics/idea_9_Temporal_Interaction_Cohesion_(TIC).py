# Strategy: Temporal Interaction Cohesion (TIC)
# Intuition: Circuit routing often suffers from "oscillation" where qubits are moved to satisfy an immediate gate but then immediately moved away for a future one. TIC mitigates this by calculating a pair-wise "Cohesion Score" that aggregates the criticality of all interactions between the same two logical qubits across the entire lookahead window. By weighting interaction distances by this cumulative cohesion, the router prioritizes keeping "frequently interacting pairs" close together, effectively pre-positioning qubits for upcoming interactions and reducing the total number of required SWAPs.
# Stats: {'mean_swaps': 571.9545454545455, 'mean_depth': 958.0, 'mean_runtime': 11.999133987860246, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # 1. Map Temporal Cohesion for logical qubit pairs
    # This identifies pairs of qubits that interact frequently or critically in the visible window.
    # We use a pair-centric urgency instead of a gate-centric one.
    cohesion = {}
    
    # Front Layer (depth 0)
    for g in self.front_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        pair = tuple(sorted(qs))
        crit = self.dag_dependencies_count[g] + 1.0
        # Sub-linear criticality power (0.8) balances deep chains and parallel breadth
        cohesion[pair] = cohesion.get(pair, 0.0) + (crit ** 0.8)
            
    # Extended Layer (depth 1+)
    for g in self.extended_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        pair = tuple(sorted(qs))
        crit = self.dag_dependencies_count[g] + 1.0
        depth = self.extended_layer_index.get(g, 0) + 1.0
        # Harmonic depth decay for future interactions' contribution to cohesion
        cohesion[pair] = cohesion.get(pair, 0.0) + (crit ** 0.8) / (1.0 + depth)

    # 2. Score candidate mapping
    p1, p2 = swap_gate[0], swap_gate[1]
    # Penalize based on physical qubit "heat" (usage frequency) to avoid congestion
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # --- Front Layer Score ---
    f_score, f_count = 0.0, 0
    for g in self.front_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        # Retrieve physical locations AFTER the candidate SWAP
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        pair = tuple(sorted(qs))
        
        # Use a 1.25 power for front-layer distance to favor aggressive gap closing
        # and punish logical drifting of high-cohesion pairs.
        f_score += cohesion[pair] * (dist ** 1.25)
        f_count += 1
        
    # --- Extended Layer Score ---
    e_score, e_count = 0.0, 0
    sigma_sq_2 = 8.0 # Gaussian width parameter for temporal focus
    for g in self.extended_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        depth = self.extended_layer_index.get(g, 0)
        pair = tuple(sorted(qs))
        
        # Gaussian gating: e^(-depth^2 / 2sigma^2)
        # Filters future noise while maintaining a strong pull for immediate successors.
        decay = math.exp(-(depth**2) / sigma_sq_2)
        e_score += cohesion[pair] * dist * decay
        e_count += 1

    # Normalization by layer density
    h_f = (f_score / f_count) if f_count > 0 else 0.0
    h_e = (e_score / e_count) if e_count > 0 else 0.0
    
    # 3. Final Aggregation
    # Balanced weighting of immediate (1.0) and lookahead (0.75) needs.
    # The cohesion factor naturally scales these based on future pair-entanglement.
    return max_decay * (h_f + 0.75 * h_e)