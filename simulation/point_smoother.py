"""
Parametric Sectional Path Smoothing Module
==========================================
This module provides spatial smoothing algorithms for 2D coordinate paths.
By breaking an overall trajectory down into parametric intervals, these algorithms 
fit low-degree local polynomials to filter out high-frequency tracking noise 
while maintaining structural continuity via boundary window blending.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline  # Available for alternative spline workflows

def smooth_path_by_polyfit_sections(points, num_sections=4, poly_degree=2, blend_percentage=0.2):
    """
    Smooths an (M, 2) path using independent, rigid section-based polynomial fitting.

    Parameters:
    -----------
    points : array-like
        An (M, 2) array containing raw [X, Y] path coordinates.
    num_sections : int, default=4
        The number of disjoint spatial segments to partition the path into.
    poly_degree : int, default=2
        The mathematical degree of the fitting polynomial (e.g., 2 for quadratic).
    blend_percentage : float, default=0.2
        The percentage of the section width used to smoothly interpolate coordinates 
        at the segment boundaries. Must be between 0.0 and 0.5.

    Returns:
    --------
    tuple (x_smooth, y_smooth)
        Two 1D arrays of length M representing the filtered path coordinates.
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    # 1. Initialize an independent parametric tracking array 't'
    t = np.arange(num_points_M)
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    
    # Determine the step size of each sub-section
    section_size = num_points_M / num_sections
    
    poly_coeffs_x = []
    poly_coeffs_y = []
    
    # 2. Fitting Phase: Process each independent section
    for s in range(num_sections):
        start_idx = int(s * section_size)
        end_idx = int(min((s + 1) * section_size, num_points_M))
        
        # Isolate the segment tracking data
        t_sec = t[start_idx:end_idx]
        x_sec = x_raw[start_idx:end_idx]
        y_sec = y_raw[start_idx:end_idx]
        
        # Calculate Least-Squares polynomial coefficients
        cx = np.polyfit(t_sec, x_sec, poly_degree)
        cy = np.polyfit(t_sec, y_sec, poly_degree)

        # # Boundary Constraint: Pin the initial step to the exact starting coordinate
        # if s == 0:
        #     cx[-1] += x_raw[0] - np.polyval(cx, t[0])
        #     cy[-1] += y_raw[0] - np.polyval(cy, t[0])

        # # Boundary Constraint: Pin the final step to the exact ending coordinate
        # if s == num_sections - 1:
        #     cx[-1] += x_raw[-1] - np.polyval(cx, t[-1])
        #     cy[-1] += y_raw[-1] - np.polyval(cy, t[-1])

        poly_coeffs_x.append(cx)
        poly_coeffs_y.append(cy)

    # Pre-allocate output arrays matching the original array sizes
    x_smooth = np.zeros_like(x_raw)
    y_smooth = np.zeros_like(y_raw)
    
    # 3. Evaluation and Blending Phase
    for idx, t_val in enumerate(t):
        sec_float = t_val / section_size
        current_sec = int(np.floor(sec_float))
        current_sec = min(current_sec, num_sections - 1)
        
        # Local position scale inside the current section (0.0 to 1.0)
        local_t = sec_float - current_sec
        
        # Compute the baseline value for the current section
        val_x_curr = np.polyval(poly_coeffs_x[current_sec], t_val)
        val_y_curr = np.polyval(poly_coeffs_y[current_sec], t_val)
        
        # Case A: Forward Transition Region -> Blend with the NEXT section
        if current_sec < num_sections - 1 and local_t > (1.0 - blend_percentage):
            w = (local_t - (1.0 - blend_percentage)) / blend_percentage
            w = w * w * (3 - 2 * w)  # Smooth cubic smoothstep curve
            
            val_x_next = np.polyval(poly_coeffs_x[current_sec + 1], t_val)
            val_y_next = np.polyval(poly_coeffs_y[current_sec + 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_curr + w * val_x_next
            y_smooth[idx] = (1 - w) * val_y_curr + w * val_y_next
            
        # Case B: Backward Transition Region -> Blend with the PREVIOUS section
        elif current_sec > 0 and local_t < blend_percentage:
            w = local_t / blend_percentage
            w = w * w * (3 - 2 * w)  # Smooth cubic smoothstep curve
            
            val_x_prev = np.polyval(poly_coeffs_x[current_sec - 1], t_val)
            val_y_prev = np.polyval(poly_coeffs_y[current_sec - 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_prev + w * val_x_curr
            y_smooth[idx] = (1 - w) * val_y_prev + w * val_y_curr
            
        # Case C: Center Stable Region -> 100% current polynomial
        else:
            x_smooth[idx] = val_x_curr
            y_smooth[idx] = val_y_curr
            
    return x_smooth, y_smooth


def smooth_path_by_segments_with_overlap(points, num_sections=4, poly_degree=2, extra_points=8, blend_percentage=0.2):
    """
    Smooths an (M, 2) path by dividing it into sections with overlapping context buffers.
    This limits edge distortion before applying boundary blending.

    Parameters:
    -----------
    points : array-like
        An (M, 2) array containing raw [X, Y] path coordinates.
    num_sections : int, default=4
        The number of core sections to partition the path into.
    poly_degree : int, default=2
        The mathematical degree of the fitting polynomial.
    extra_points : int, default=8
        The lookback and lookahead node margin used to add context during fitting.
    blend_percentage : float, default=0.2
        The boundary width scale used for smoothstep blending.

    Returns:
    --------
    tuple (x_smooth, y_smooth)
        Two 1D arrays of length M representing the filtered path coordinates.
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    t = np.arange(num_points_M)
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    
    section_size = num_points_M / num_sections
    
    poly_coeffs_x = []
    poly_coeffs_y = []
    
    # 1. Overlapping Fit Phase
    for s in range(num_sections):
        start_idx = int(s * section_size)
        end_idx = int(min((s + 1) * section_size, num_points_M))
        
        # Expand the fitting windows out symmetrically beyond the rigid boundaries
        fit_start = max(0, start_idx - extra_points)
        fit_end = min(num_points_M, end_idx + extra_points)
        
        t_fit = t[fit_start:fit_end]
        x_fit = x_raw[fit_start:fit_end]
        y_fit = y_raw[fit_start:fit_end]
        
        # Fit the model utilizing extended window contextual points
        cx = np.polyfit(t_fit, x_fit, poly_degree)
        cy = np.polyfit(t_fit, y_fit, poly_degree)

        # Force physical endpoint constraints
        # if s == 0:
        #     cx[-1] += x_raw[0] - np.polyval(cx, t[0])
        #     cy[-1] += y_raw[0] - np.polyval(cy, t[0])
        # if s == num_sections - 1:
        #     cx[-1] += x_raw[-1] - np.polyval(cx, t[-1])
        #     cy[-1] += y_raw[-1] - np.polyval(cy, t[-1])

        poly_coeffs_x.append(cx)
        poly_coeffs_y.append(cy)
        
    x_smooth = np.zeros_like(x_raw)
    y_smooth = np.zeros_like(y_raw)
    
    # 2. Evaluation Phase (Interpolates across adjacent models using smoothstep)
    for idx, t_val in enumerate(t):
        sec_float = t_val / section_size
        current_sec = int(np.floor(sec_float))
        current_sec = min(current_sec, num_sections - 1)
        
        local_t = sec_float - current_sec
        
        val_x_curr = np.polyval(poly_coeffs_x[current_sec], t_val)
        val_y_curr = np.polyval(poly_coeffs_y[current_sec], t_val)
        
        # Blend forward transition
        if current_sec < num_sections - 1 and local_t > (1.0 - blend_percentage):
            w = (local_t - (1.0 - blend_percentage)) / blend_percentage
            w = w * w * (3 - 2 * w)
            
            val_x_next = np.polyval(poly_coeffs_x[current_sec + 1], t_val)
            val_y_next = np.polyval(poly_coeffs_y[current_sec + 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_curr + w * val_x_next
            y_smooth[idx] = (1 - w) * val_y_curr + w * val_y_next
            
        # Blend backward transition
        elif current_sec > 0 and local_t < blend_percentage:
            w = local_t / blend_percentage
            w = w * w * (3 - 2 * w)
            
            val_x_prev = np.polyval(poly_coeffs_x[current_sec - 1], t_val)
            val_y_prev = np.polyval(poly_coeffs_y[current_sec - 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_prev + w * val_x_curr
            y_smooth[idx] = (1 - w) * val_y_prev + w * val_y_curr
            
        else:
            x_smooth[idx] = val_x_curr
            y_smooth[idx] = val_y_curr
            
    return x_smooth, y_smooth