import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np

def transform_data_to_fiber_format(fiber_data, link_distance_m=1.0):
    """
    Transforms raw data into a continuous South-to-North tracking path 
    with a precise node spacing of exactly 1 meter.
    
    Returns:
        x, y, distances, uncertain_start, uncertain_end
    """
    data = np.atleast_2d(fiber_data)
    angles_deg = data[:, 0]
    segment_distances = data[:, 1]
    
    # 1. Calculate raw relative coordinate jumps
    angles_rad = np.radians(angles_deg)
    dx_raw = segment_distances * np.sin(angles_rad)
    dy_raw = segment_distances * np.cos(angles_rad)
    
    # 2. Build the initial raw waypoint backbone joints
    x_raw = np.concatenate(([0.0], np.cumsum(dx_raw)))
    y_raw = np.concatenate(([0.0], np.cumsum(dy_raw)))
    
    # 3. Rotate the backbone joints so the trend aligns straight up the Y-axis
    net_x = x_raw[-1] - x_raw[0]
    net_y = y_raw[-1] - y_raw[0]
    
    current_heading = np.arctan2(net_y, net_x)
    desired_heading = np.pi / 2.0  # 90 degrees (Straight up / North)
    rotation_angle = desired_heading - current_heading
    
    cos_rot = np.cos(rotation_angle)
    sin_rot = np.sin(rotation_angle)
    
    x_rotated_joints = x_raw * cos_rot - y_raw * sin_rot
    y_rotated_joints = x_raw * sin_rot + y_raw * cos_rot
    
    # 4. Interpolate the rotated path to place a node at every single meter
    # Track the cumulative distance along the structural joint segments
    joint_distances = np.concatenate(([0.0], np.cumsum(segment_distances)))
    total_fiber_length = joint_distances[-1]
    
    # Generate the high-density distance array (0, 1, 2, 3... total_length)
    distances = np.arange(0, total_fiber_length, link_distance_m)
    
    # Map the coordinates continuously at 1-meter intervals
    x = np.interp(distances, joint_distances, x_rotated_joints)
    y = np.interp(distances, joint_distances, y_rotated_joints)
    
    # 5. Generate circular Gaussian blur around start and end positions
    start_noise_x, start_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    end_noise_x, end_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    
    uncertain_start = (x[0] + start_noise_x, y[0] + start_noise_y)
    uncertain_end = (x[-1] + end_noise_x, y[-1] + end_noise_y)
    
    return x, y, distances, uncertain_start, uncertain_end

def generate_ultra_smooth_path(num_points_M=3000, link_distance_m=1.0, max_deviation=50.0):
    """
    Generates a path with an ultra-low derivative by tightly restricting 
    the maximum turn rate per step.
    
    Returns:
        x, y, distances, uncertain_start, uncertain_end
    """
    x = np.zeros(num_points_M)
    y = np.zeros(num_points_M)
    angles = np.zeros(num_points_M)
    
    # Max turn angle restricted to 0.75 degrees per step
    max_turn_per_step = np.radians(0.75)
    random_turn_angles = np.random.uniform(-max_turn_per_step, max_turn_per_step, size=num_points_M)
    
    current_angle = 0.0
    for i in range(1, num_points_M):
        proposed_angle = current_angle + random_turn_angles[i]
        
        # FIXED: Cubic proactive centering force. 
        # Weak near x=0, but scales up aggressively near the 50m boundary
        proportional_bias = ((x[i-1] / max_deviation) ** 3) * np.radians(0.5)
        proposed_angle -= proportional_bias
            
        # Calculate steps (0 degrees travels straight up the Y-axis)
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
    
    # Calculate Uncertain Start and End Coordinates with Gaussian blur
    start_noise_x, start_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    end_noise_x, end_noise_y = np.random.normal(loc=0.0, scale=1.0, size=2)
    
    uncertain_start = (x[0] + start_noise_x, y[0] + start_noise_y)
    uncertain_end = (x[-1] + end_noise_x, y[-1] + end_noise_y)
    
    return x, y, distances, uncertain_start, uncertain_end

def plot_fiber_trajectory(x, y, max_deviation=50.0):
    """
    Plots the complete graph of the fiber trajectory along its entire path.
    """
    plt.figure(figsize=(6, 8))
    
    # Boundary walls and center lines
    plt.axvline(x=-max_deviation, color='red', linestyle='--', alpha=0.5, label=f'{int(max_deviation)}m Boundary')
    plt.axvline(x=max_deviation, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle=':', label='Ideal Center Line (X=0)')
    
    # Plotting the trajectory
    plt.plot(x, y, color='blue', alpha=0.7, label='Ultra-Smooth Fiber Path')
    plt.scatter(x, y, color='black', s=1, zorder=3)
    
    plt.title('Full View: Ultra-Smooth Fiber Path (Total 3000m)')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.xlim(-max_deviation - 5, max_deviation + 5)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Execution and Speed Benchmark ---
    start_time = time.time()

    raw_data = [
        [36, 214], [42, 106], [23, 235], [35, 121], [42, 111],
        [37, 167], [48, 211], [57, 52],  [40, 25],  [40, 25],
        [49, 38],  [48, 167], [48, 195], [46, 212]
    ]

    # Unpack the new 5-element return tuple
    x, y, distances, start_pt, end_pt = generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0)
    # x, y, distances, start_pt, end_pt = transform_data_to_fiber_format(raw_data)
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

    plot_fiber_trajectory(x,y, 50)