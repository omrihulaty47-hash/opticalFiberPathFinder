import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def simulate_constrained_two_points(num_anchors=6, circle_radius=60.0, known_distance_m=30.0):
    # Constants
    SPEED_OF_SOUND = 343.0  # m/s
    TIME_ERROR_MS = 5.0
    error_std_meters = (TIME_ERROR_MS / 1000.0) * SPEED_OF_SOUND  # ~1.715m
    
    circle_center = np.array([100.0, 100.0])
    
    # 1. Define TRUE locations for Subject 1 and Subject 2 (Exactly 'known_distance_m' apart)
    true_p1 = circle_center + np.array([-known_distance_m / 2, 10.0])
    true_p2 = circle_center + np.array([known_distance_m / 2, 10.0])
    
    # 2. Setup Anchors in a circle around the central zone
    anchors = []
    for i in range(num_anchors):
        angle = 2 * np.pi * i / num_anchors
        x_a = circle_center[0] + circle_radius * np.cos(angle)
        y_a = circle_center[1] + circle_radius * np.sin(angle)
        anchors.append([x_a, y_a])
    anchors = np.array(anchors)
    
    # 3. Simulate independent noisy acoustic distance readings to BOTH subjects
    true_dist_p1 = np.linalg.norm(anchors - true_p1, axis=1)
    true_dist_p2 = np.linalg.norm(anchors - true_p2, axis=1)
    
    noisy_dist_p1 = true_dist_p1 + np.random.normal(0, error_std_meters, size=num_anchors)
    noisy_dist_p2 = true_dist_p2 + np.random.normal(0, error_std_meters, size=num_anchors)
    
    # 4. Joint Loss Function (Minimizes acoustic variance for both targets together)
    def objective_function(states):
        # Unpack the 4 optimized elements
        x1, y1, x2, y2 = states
        p1_guess = np.array([x1, y1])
        p2_guess = np.array([x2, y2])
        
        # Calculate residuals for both sets of data
        residuals_p1 = np.linalg.norm(anchors - p1_guess, axis=1) - noisy_dist_p1
        residuals_p2 = np.linalg.norm(anchors - p2_guess, axis=1) - noisy_dist_p2
        
        return np.sum(residuals_p1**2) + np.sum(residuals_p2**2)
    
    # 5. THE CRITICAL RESTRICTION: Equality Constraint
    # This function must equal 0 for the optimizer's solution to be valid.
    def distance_constraint(states):
        x1, y1, x2, y2 = states
        calculated_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return calculated_distance - known_distance_m

    constraints = ({'type': 'eq', 'fun': distance_constraint})
    
    # 6. Execute Constrained Optimization using SLSQP
    # Initial guess places both targets roughly near the array center, slightly separated
    initial_guess = [90.0, 100.0, 110.0, 100.0] 
    
    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)
    
    # Unpack final outputs
    est_x1, est_y1, est_x2, est_y2 = result.x
    est_p1 = np.array([est_x1, est_y1])
    est_p2 = np.array([est_x2, est_y2])
    
    # Calculate performance metrics
    err_p1 = np.linalg.norm(est_p1 - true_p1)
    err_p2 = np.linalg.norm(est_p2 - true_p2)
    final_calculated_separation = np.linalg.norm(est_p1 - est_p2)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 10))
    
    # Plot anchors
    plt.scatter(anchors[:,0], anchors[:,1], color='red', marker='^', s=100, zorder=4, label='Microphones')
    
    # Plot true positions & their baseline bond
    plt.scatter([true_p1[0], true_p2[0]], [true_p1[1], true_p2[1]], color='green', s=120, zorder=5, label='True Targets')
    plt.plot([true_p1[0], true_p2[0]], [true_p1[1], true_p2[1]], 'g-', linewidth=2, label=f'True Bond ({known_distance_m}m)')
    
    # Plot estimated locations & their calculated bond
    plt.scatter([est_p1[0], est_p2[0]], [est_p1[1], est_p2[1]], color='magenta', marker='x', s=120, linewidths=3, zorder=6, label='Estimated Positions')
    plt.plot([est_p1[0], est_p2[0]], [est_p1[1], est_p2[1]], 'm--', linewidth=2, label=f'Enforced Bond ({final_calculated_separation:.3f}m)')
    
    # Format environment layout
    plt.title(f'Constrained Dual-Target Triangulation\nEnforced Separation Constraint (m) = {known_distance_m}m', fontsize=12, fontweight='bold')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.xlim(circle_center[0] - circle_radius - 20, circle_center[0] + circle_radius + 20)
    plt.ylim(circle_center[1] - circle_radius - 20, circle_center[1] + circle_radius + 20)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    
    # Print numerical truth tests
    print("--- Constrained Solver Outputs ---")
    print(f"Target 1 Coordinate Error: {err_p1:.3f} meters")
    print(f"Target 2 Coordinate Error: {err_p2:.3f} meters")
    print(f"Resulting separation between estimations: {final_calculated_separation:.6f} meters")

# Execute the dual tracking simulation where the points are locked exactly 30 meters apart
simulate_constrained_two_points(num_anchors=6, circle_radius=1000.0, known_distance_m=3.0)