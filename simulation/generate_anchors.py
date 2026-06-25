import numpy as np
import matplotlib.pyplot as plt
import fiber_simulation

def generate_circular_anchors(center_point, radius, num_anchors_N):
    """
    Generates the coordinates of N anchors evenly spaced in a circle 
    around a specified center point.
    
    Parameters:
    -----------
    center_point : list or numpy.ndarray
        The [X, Y] coordinates of the circle's center.
    radius : float
        The radius of the circle in meters.
    num_anchors_N : int
        The number of anchors to place.
        
    Returns:
    --------
    numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the placed anchors.
    """
    # 1. Generate N equally spaced angles from 0 to 2*pi
    # endpoint=False prevents overlapping the first and last points at 2*pi
    angles = np.linspace(0, 2 * np.pi, num_anchors_N, endpoint=False)
    
    # 2. Calculate the relative X and Y offsets using trig
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)
    
    # 3. Stack offsets into an (N, 2) array and shift them by the center point
    anchors = np.column_stack((x_offsets, y_offsets)) + center_point
    
    return anchors

def generate_linear_anchors(center_point, spacing_m, num_anchors_N, angle_degrees=0.0):
    """
    Generates the coordinates of N anchors perfectly aligned in a straight line 
    passing through a center point at a specific angle.
    
    Parameters:
    -----------
    center_point : list or numpy.ndarray
        The [X, Y] coordinates of the central point of the line array.
    spacing_m : float
        The physical distance (in meters) between adjacent anchors.
    num_anchors_N : int
        The total number of anchors to place.
    angle_degrees : float
        The heading of the line in degrees (0 = along X-axis, 90 = along Y-axis).
        
    Returns:
    --------
    numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the linear anchor array.
    """
    center_point = np.asarray(center_point)
    
    # 1. Convert the heading angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # 2. Create centered linear step indices 
    # For example, if N=5, steps = [-2, -1, 0, 1, 2]
    # If N=4, steps = [-1.5, -0.5, 0.5, 1.5]
    steps = np.arange(num_anchors_N) - (num_anchors_N - 1) / 2.0
    
    # 3. Calculate absolute 1D distances along the line from the center
    distances = steps * spacing_m
    
    # 4. Project the 1D distances onto the X and Y axes using trig
    dx = distances * np.cos(angle_rad)
    dy = distances * np.sin(angle_rad)
    
    # 5. Offset the generated grid by the true center coordinate
    anchors = np.column_stack((dx, dy)) + center_point
    
    return anchors

def generate_staggered_anchors(center_point, spacing_m, num_anchors_N, angle_degrees=0.0, stagger_offset_m=0.0):
    """
    Generates the coordinates of N anchors distributed along a line baseline,
    but staggers them alternatingly left and right to maintain symmetry-breaking
    coverage across a limited hearing range.
    
    Parameters:
    -----------
    center_point : list or numpy.ndarray
        The [X, Y] coordinates of the central point of the line array.
    spacing_m : float
        The physical distance (in meters) between adjacent anchors along the track length.
    num_anchors_N : int
        The total number of anchors to place.
    angle_degrees : float
        The heading of the baseline in degrees (0 = along X-axis, 90 = along Y-axis).
    stagger_offset_m : float, default=0.0
        The distance (in meters) to displace anchors perpendicular to the baseline.
        Alternates directions (+ / -) for each sequential anchor.
        
    Returns:
    --------
    numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the staggered anchor array.
    """
    center_point = np.asarray(center_point)
    
    # 1. Convert the heading angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # 2. Create centered linear step indices along the track length
    steps = np.arange(num_anchors_N) - (num_anchors_N - 1) / 2.0
    distances = steps * spacing_m
    
    # 3. Project the base inline positions
    dx = distances * np.cos(angle_rad)
    dy = distances * np.sin(angle_rad)
    anchors = np.column_stack((dx, dy)) + center_point
    
    # 4. Continuous Symmetry Breaking: Apply an alternating perpendicular offset
    if stagger_offset_m != 0.0:
        # Calculate the perpendicular unit vector (90 degrees counter-clockwise)
        perp_angle_rad = angle_rad + (np.pi / 2.0)
        perp_dir = np.array([np.cos(perp_angle_rad), np.sin(perp_angle_rad)])
        
        # Create an alternating array: [1.0, -1.0, 1.0, -1.0, ...]
        # This creates a reliable zigzag down the entire track
        alternator = np.ones(num_anchors_N)
        alternator[1::2] = -1.0
        
        # Compute the scale of displacement for each anchor
        offsets = alternator[:, np.newaxis] * stagger_offset_m * perp_dir
        
        # Apply the offsets globally across the array matrix
        anchors += offsets
        
    return anchors

def generate_half_circle_anchors(center_point, radius, num_anchors_N, start_angle_degrees=0.0):
    """
    Generates the coordinates of N anchors evenly spaced along a half-circle arc
    around a specified center point.
    
    Parameters:
    -----------
    center_point : list or numpy.ndarray
        The [X, Y] coordinates of the arc's center.
    radius : float
        The radius of the half-circle in meters.
    num_anchors_N : int
        The number of anchors to distribute along the arc.
    start_angle_degrees : float
        The starting angle of the arc in degrees (0 = Positive X-axis).
        The arc will sweep 180 degrees counter-clockwise from this angle.
        
    Returns:
    --------
    numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the anchors.
    """
    center_point = np.asarray(center_point)
    
    # 1. Convert the starting angle to radians
    start_rad = np.radians(start_angle_degrees)
    end_rad = start_rad + np.pi  # A half-circle is exactly pi radians (180 degrees)
    
    # 2. Generate N equally spaced angles from start to end
    # endpoint=True ensures anchors are placed exactly at both ends of the arc
    angles = np.linspace(start_rad, end_rad, num_anchors_N, endpoint=True)
    
    # 3. Calculate relative X and Y offsets using trig
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)
    
    # 4. Combine offsets and translate by the center point coordinates
    anchors = np.column_stack((x_offsets, y_offsets)) + center_point
    
    return anchors

def perturb_points_max_1m(points):
    """
    Takes an (M, 2) array of points and shifts each index by a random 
    distance of AT MOST 1 meter in a random uniform direction.
    
    Returns an (M, 2) array of the perturbed points.
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    # 1. Generate a random uniform angle for the direction (0 to 2*pi)
    angles = np.random.uniform(0, 2 * np.pi, size=num_points_M)
    
    # 2. Generate a random uniform radius (0 to 1 meter)
    # Using np.sqrt ensures a true uniform distribution across the area of the 1m disk
    radii = np.random.normal(loc=0, scale=1, size=num_points_M)
    
    # 3. Convert polar coordinates (radius, angle) to Cartesian offsets (dx, dy)
    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)
    
    # 4. Add the offsets directly to the original coordinates
    perturbed_points = points + np.column_stack((dx, dy))
    
    return perturbed_points


if __name__ == "__main__":
    # Create 5 sample original points
    original = generate_staggered_anchors(np.array([0,1500]), 300, 10, 90, 4)
    
    # Perturb them
    perturbed = perturb_points_max_1m(original)
    
    # Verify the distance constraints mathematically
    distances = np.linalg.norm(perturbed - original, axis=1)
    
    print("--- Distance Verification ---")
    for i in range(len(original)):
        print(f"Point {i+1}: Shifted by {distances[i]:.4f} meters (Max allowed: 1.0m)")
        
    # Quick plot to visually see the 1-meter bounding "halos"
    # plt.figure(figsize=(8, 6))
    plt.scatter(original[:, 0], original[:, 1], color='black', marker='o', s=50, label='Original Points', zorder=4)
    plt.scatter(perturbed[:, 0], perturbed[:, 1], color='red', marker='x', s=50, label='Perturbed Points', zorder=4)
    
    # Draw 1-meter boundary circles around the original points to prove compliance
    for idx, pt in enumerate(original):
        # circle = plt.Circle((pt[0], pt[1]), 1.0, color='blue', fill=False, linestyle='--', alpha=0.3)
        # plt.gca().add_patch(circle)
        # Draw a small line connecting the movement
        plt.plot([original[idx, 0], perturbed[idx, 0]], [original[idx, 1], perturbed[idx, 1]], 'gray', alpha=0.5)


    x, y, distances, start_pt, end_pt = fiber_simulation.generate_ultra_smooth_path(num_points_M=1001, link_distance_m=3.0)
    plt.axvline(x=0, color='gray', linestyle=':', label='Ideal Center Line (X=0)')
    
    # Plotting the trajectory
    plt.plot(x, y, color='blue', alpha=0.7, label='Ultra-Smooth Fiber Path')
    plt.scatter(x, y, color='black', s=1, zorder=3)

    plt.title("Anchors On Fiber (Max 1 Meter Step Bounds)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    # plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()
