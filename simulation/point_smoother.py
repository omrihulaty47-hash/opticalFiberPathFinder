import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def smooth_path_by_sections(points, num_sections=5, smoothing_factor=None):
    """
    Smooths an (M, 2) path by dividing it into sections based on point indices.
    Works for any path geometry, including loops, curves, and vertical lines.
    
    Parameters:
    -----------
    points : numpy.ndarray
        An (M, 2) array of raw [X, Y] coordinates.
    num_sections : int
        The number of physical sections to divide the path into.
    smoothing_factor : float or None
        Controls the smoothness. Higher = smoother curve.
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    # 1. Create a strictly increasing independent parameter 't' (the step index)
    t = np.arange(num_points_M)
    
    # Extract independent X and Y channels
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    
    # 2. Divide 't' into uniform sectional knots
    quantiles = np.linspace(0, 1, num_sections + 1)[1:-1]
    knots = np.quantile(t, quantiles)
    
    if smoothing_factor is None:
        smoothing_factor = num_points_M * 0.5
        
    # 3. Fit separate splines for X and Y relative to the step index 't'
    # This completely bypasses the non-increasing X array restriction
    spline_x = UnivariateSpline(t, x_raw, k=3, s=smoothing_factor)
    spline_y = UnivariateSpline(t, y_raw, k=3, s=smoothing_factor)
    
    spline_x.set_knots(knots)
    spline_y.set_knots(knots)
    
    # 4. Generate a high-resolution smooth index map for rendering
    t_smooth = np.linspace(0, num_points_M - 1, 500)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)
    
    # Bundle smooth output coordinates together
    smooth_path = np.column_stack((x_smooth, y_smooth))
    
    return smooth_path, spline_x, spline_y
