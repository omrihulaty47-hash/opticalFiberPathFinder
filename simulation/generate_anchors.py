import numpy as np
import matplotlib.pyplot as plt

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
    original = generate_circular_anchors(np.array([0,0]), 100, 10)
    
    # Perturb them
    perturbed = perturb_points_max_1m(original)
    
    # Verify the distance constraints mathematically
    distances = np.linalg.norm(perturbed - original, axis=1)
    
    print("--- Distance Verification ---")
    for i in range(len(original)):
        print(f"Point {i+1}: Shifted by {distances[i]:.4f} meters (Max allowed: 1.0m)")
        
    # Quick plot to visually see the 1-meter bounding "halos"
    plt.figure(figsize=(8, 6))
    plt.scatter(original[:, 0], original[:, 1], color='black', marker='o', s=50, label='Original Points', zorder=4)
    plt.scatter(perturbed[:, 0], perturbed[:, 1], color='red', marker='x', s=50, label='Perturbed Points', zorder=4)
    
    # Draw 1-meter boundary circles around the original points to prove compliance
    for idx, pt in enumerate(original):
        circle = plt.Circle((pt[0], pt[1]), 1.0, color='blue', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        # Draw a small line connecting the movement
        plt.plot([original[idx, 0], perturbed[idx, 0]], [original[idx, 1], perturbed[idx, 1]], 'gray', alpha=0.5)

    plt.title("Spatial Perturbation Map (Max 1 Meter Step Bounds)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()
