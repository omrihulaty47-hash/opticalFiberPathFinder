import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def analyze_dynamic_m_points(num_points_M=8, link_distance_m=15.0, num_anchors=6, circle_radius=70.0, iterations=100):
    # Constants
    SPEED_OF_SOUND = 343.0  # m/s
    TIME_ERROR_MS = 5.0
    error_std_meters = (TIME_ERROR_MS / 1000.0) * SPEED_OF_SOUND  # ~1.715m
    
    circle_center = np.array([100.0, 100.0])
    
    # 1. Generate TRUE positions for an M-point chain (S-curve layout)
    true_points = np.zeros((num_points_M, 2))
    true_points[0] = circle_center + np.array([-25.0, -25.0]) # Start point
    
    for i in range(1, num_points_M):
        # alternate directions to create a snake/chain layout
        angle = (i * np.pi / 3) if (i % 2 == 0) else (-i * np.pi / 4)
        step = np.array([link_distance_m * np.cos(angle), link_distance_m * np.sin(angle)])
        true_points[i] = true_points[i-1] + step
        
    # 2. Setup Microphone Anchors in a circle around the center zone
    anchors = []
    for i in range(num_anchors):
        angle = 2 * np.pi * i / num_anchors
        x_a = circle_center[0] + circle_radius * np.cos(angle)
        y_a = circle_center[1] + circle_radius * np.sin(angle)
        anchors.append([x_a, y_a])
    anchors = np.array(anchors)
    
    # 3. Compute baseline anchor-to-subject distances
    true_anchor_distances = np.zeros((num_points_M, num_anchors))
    for i in range(num_points_M):
        true_anchor_distances[i] = np.linalg.norm(anchors - true_points[i], axis=1)
        
    # Track matrix errors: rows = points (1 to M), columns = iteration trials
    all_errors = np.zeros((num_points_M, iterations))
    max_constraint_violations = []
    
    # 4. Dynamic Objective Loss Function for M points (M * 2 variables)
    def objective_function(states, noisy_matrix):
        current_guess = states.reshape(num_points_M, 2)
        total_loss = 0
        for i in range(num_points_M):
            calc_dist = np.linalg.norm(anchors - current_guess[i], axis=1)
            total_loss += np.sum((calc_dist - noisy_matrix[i])**2)
        return total_loss
    
    # 5. Dynamically build M-1 Sequential Constraints
    constraints = []
    for i in range(1, num_points_M):
        def make_constraint_factory(idx):
            return lambda states: np.linalg.norm(states.reshape(num_points_M, 2)[idx] - states.reshape(num_points_M, 2)[idx-1]) - link_distance_m
        
        constraints.append({
            'type': 'eq',
            'fun': make_constraint_factory(i)
        })
        
    # Initial guess baseline
    initial_guess = (true_points + np.random.normal(0, 2.0, size=(num_points_M, 2))).flatten()
    
    print(f"Running Monte Carlo loop for M = {num_points_M} points ({num_points_M*2} independent coordinate variables)...")
    
    # Run Simulation Loop
    last_estimation = None
    for trial in range(iterations):
        noisy_matrix = true_anchor_distances + np.random.normal(0, error_std_meters, size=(num_points_M, num_anchors))
        
        result = minimize(objective_function, initial_guess, args=(noisy_matrix,), 
                          method='SLSQP', constraints=constraints, options={'maxiter': 150})
        
        est_points = result.x.reshape(num_points_M, 2)
        last_estimation = est_points # Keep the final run coordinates to visualize shape mapping
        
        # Log error trends per node
        for i in range(num_points_M):
            all_errors[i, trial] = np.linalg.norm(est_points[i] - true_points[i])
            
        # Log distance validation checks
        violations = [np.abs(np.linalg.norm(est_points[k] - est_points[k-1]) - link_distance_m) for k in range(1, num_points_M)]
        max_constraint_violations.append(np.max(violations))

    # --- Metrics Extraction ---
    mean_errors_per_point = np.mean(all_errors, axis=1)
    p95_errors_per_point = np.percentile(all_errors, 95, axis=1)
    
    # --- Visualization Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left Plot: Geometry Mapping Layout
    ax1.scatter(anchors[:, 0], anchors[:, 1], color='red', marker='^', s=100, zorder=3, label='Microphones')
    ax1.plot(true_points[:, 0], true_points[:, 1], 'go-', linewidth=2.5, markersize=8, label='True Chain Path')
    ax1.plot(last_estimation[:, 0], last_estimation[:, 1], 'mx--', linewidth=2, markersize=8, label='Sample Estimated Path')
    
    # Label first and last node indices
    ax1.text(true_points[0,0]-5, true_points[0,1]-5, "Node 1", weight='bold', color='darkgreen')
    ax1.text(true_points[-1,0]+3, true_points[-1,1]+3, f"Node {num_points_M}", weight='bold', color='darkgreen')
    
    ax1.set_title('Physical Layout & Tracking Shape Fit', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    
    # Right Plot: Error Trend Across the Chain Nodes
    node_indices = np.arange(1, num_points_M + 1)
    ax2.plot(node_indices, mean_errors_per_point, 'b-o', linewidth=2.5, label='Mean Absolute Error')
    ax2.fill_between(node_indices, mean_errors_per_point, p95_errors_per_point, color='blue', alpha=0.15, label='95% Probability Envelope')
    
    ax2.set_title('Tracking Error vs. Position Index in Chain', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Node Index along the Chain (1 to M)')
    ax2.set_ylabel('Positioning Error (Meters)')
    ax2.set_xticks(node_indices)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.suptitle(f'Dynamic Simulation: M = {num_points_M} Connected Nodes\nLink Distance Constraint = {link_distance_m}m', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Verification prints
    print(f"--- Global Performance Summary ---")
    print(f"Average Tracking Error across all links: {np.mean(mean_errors_per_point):.3f} meters")
    print(f"Maximum Chain Constraint Gap Slip: {np.max(max_constraint_violations):.2e} meters")

# --- Try changing parameters here! ---
# You can freely scale num_points_M to whatever integer length you want to test.
analyze_dynamic_m_points(num_points_M=20, link_distance_m=3.0, num_anchors=10, circle_radius=1000.0)