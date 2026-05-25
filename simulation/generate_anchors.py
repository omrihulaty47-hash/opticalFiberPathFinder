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

