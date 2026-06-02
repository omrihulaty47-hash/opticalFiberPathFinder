import numpy as np
import matplotlib.pyplot as plt
import time

def generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0, max_deviation=20.0):
    """
    Generates a path with an ultra-low derivative by tightly restricting 
    the maximum turn rate per step. Runs in milliseconds.
    
    Returns:
        x, y, distances, uncertain_start, uncertain_end
    """
    x = np.zeros(num_points_M)
    y = np.zeros(num_points_M)
    angles = np.zeros(num_points_M)
    
    # Max turn angle restricted to 0.75 degrees per 3-meter step
    max_turn_per_step = np.radians(0.75)
    random_turn_angles = np.random.uniform(-max_turn_per_step, max_turn_per_step, size=num_points_M)
    
    current_angle = 0.0
    for i in range(1, num_points_M):
        proposed_angle = current_angle + random_turn_angles[i]
        
        # Proactive centering force
        proportional_bias = (x[i-1] / max_deviation) * np.radians(0.4)
        proposed_angle -= proportional_bias
            
        # Calculate steps
        dx = link_distance_m * np.sin(proposed_angle)
        dy = link_distance_m * np.cos(proposed_angle)
        
        proposed_x = x[i-1] + dx
        
        # Safety hard clamp
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
    
    # --- NEW CODE: Calculate Uncertain Start and End Coordinates ---
    # Generates a circular Gaussian blur around (x[0], y[0]) and (x[-1], y[-1])
    start_noise_x, start_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    end_noise_x, end_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    
    uncertain_start = (x[0] + start_noise_x, y[0] + start_noise_y)
    uncertain_end = (x[-1] + end_noise_x, y[-1] + end_noise_y)
    # -----------------------------------------------------------------
    
    return x, y, distances, uncertain_start, uncertain_end


if __name__ == "__main__":
    # --- Execution and Speed Benchmark ---
    start_time = time.time()
    # Unpack the new 5-element return tuple
    x, y, distances, start_pt, end_pt = generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0)
    execution_time = (time.time() - start_time) * 1000
    
    print(f"Path generated in: {execution_time:.2f} milliseconds")
    print(f"True Start: (0.0, 0.0) -> Uncertain Start: ({start_pt[0]:.2f}, {start_pt[1]:.2f})")
    print(f"True End: ({x[-1]:.2f}, {y[-1]:.2f}) -> Uncertain End: ({end_pt[0]:.2f}, {end_pt[1]:.2f})")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Close-up of Start Point Uncertainty
    ax1.plot(x[:10], y[:10], color='blue', alpha=0.5, label='Path Origin')
    ax1.scatter(x[0], y[0], color='black', s=100, zorder=5, label='True Start (0,0)')
    ax1.scatter(start_pt[0], start_pt[1], color='magenta', s=100, zorder=6, label='Uncertain Start')
    # Draw a 1-meter radius circle representing 1 standard deviation
    circle_start = plt.Circle((x[0], y[0]), 1.0, color='magenta', fill=False, linestyle='--', alpha=0.5, label='1m Std Dev Circle')
    ax1.add_patch(circle_start)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title("Start Point Uncertainty Close-up")
    ax1.legend()

    # Plot 2: Close-up of End Point Uncertainty
    ax2.plot(x[-10:], y[-10:], color='blue', alpha=0.5, label='Path Termination')
    ax2.scatter(x[-1], y[-1], color='black', s=100, zorder=5, label='True End')
    ax2.scatter(end_pt[0], end_pt[1], color='cyan', s=100, zorder=6, label='Uncertain End')
    # Draw a 1-meter radius circle representing 1 standard deviation
    circle_end = plt.Circle((x[-1], y[-1]), 1.0, color='cyan', fill=False, linestyle='--', alpha=0.5, label='1m Std Dev Circle')
    ax2.add_patch(circle_end)
    ax2.set_xlim(x[-1] - 3, x[-1] + 3)
    ax2.set_ylim(y[-1] - 3, y[-1] + 3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_title("End Point Uncertainty Close-up")
    ax2.legend()

    plt.tight_layout()
    plt.show()