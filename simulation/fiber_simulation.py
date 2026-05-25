import numpy as np
import matplotlib.pyplot as plt
import time

def generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0, max_deviation=20.0):
    """
    Generates a path with an ultra-low derivative by tightly restricting 
    the maximum turn rate per step. Runs in milliseconds.
    """
    x = np.zeros(num_points_M)
    y = np.zeros(num_points_M)
    angles = np.zeros(num_points_M)
    
    # CRITICAL CHANGE: Max turn angle restricted to 0.75 degrees per 3-meter step
    # This keeps the heading changes incredibly tiny.
    max_turn_per_step = np.radians(0.75)
    random_turn_angles = np.random.uniform(-max_turn_per_step, max_turn_per_step, size=num_points_M)
    
    current_angle = 0.0
    for i in range(1, num_points_M):
        # Apply the tiny random adjustment
        proposed_angle = current_angle + random_turn_angles[i]
        
        # Proactive centering force: if we drift away from X=0, 
        # apply a subtle counter-steering bias proportional to our distance
        # This keeps the trajectory stable and near the center.
        proportional_bias = (x[i-1] / max_deviation) * np.radians(0.4)
        proposed_angle -= proportional_bias
            
        # Calculate steps
        dx = link_distance_m * np.sin(proposed_angle)
        dy = link_distance_m * np.cos(proposed_angle)
        
        proposed_x = x[i-1] + dx
        
        # Safety hard clamp (virtually never hit now because of the ultra-low derivative)
        if proposed_x > max_deviation:
            proposed_x = max_deviation
            proposed_angle = 0.0
        elif proposed_x < -max_deviation:
            proposed_x = -max_deviation
            proposed_angle = 0.0
            
        x[i] = proposed_x
        y[i] = y[i-1] + dy
        current_angle = proposed_angle
        angles[i] = current_angle
        
    distances = np.arange(0, num_points_M) * link_distance_m
    return x, y, distances


if __name__ == "__main__":
    # --- Execution and Speed Benchmark ---
    start_time = time.time()
    x, y, distances = generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0)
    execution_time = (time.time() - start_time) * 1000
    print(f"Path generated in: {execution_time:.2f} milliseconds")

    # --- Visualization ---

    # Full layout plot
    plt.axvline(x=-20, color='red', linestyle='--', alpha=0.5, label='20m Boundary')
    plt.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle=':', label='Ideal Center Line (X=0)')
    plt.plot(x, y, color='blue', alpha=0.7, label='Ultra-Low Derivative Path')
    plt.scatter(x, y, color='black', s=2, zorder=3)
    plt.title('Full View: Ultra-Smooth Path (Total 3000m)')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.xlim(-25, 25)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Verify Derivative and Spacing Statistics ---
    calculated_intervals = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    derivatives = np.diff(x) / np.diff(y)  # dx/dy

    print("\n--- Structural Analysis ---")
    print(f"Minimum segment spacing: {np.min(calculated_intervals):.6f} meters")
    print(f"Maximum segment spacing: {np.max(calculated_intervals):.6f} meters")
    print(f"Average absolute derivative |dx/dy|: {np.mean(np.abs(derivatives)):.6f} (Extremely close to 0!)")
    print(f"Peak maximum derivative |dx/dy|: {np.max(np.abs(derivatives)):.6f}")