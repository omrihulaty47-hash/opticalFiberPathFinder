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


def _linearize_edges(x_smooth, y_smooth, x_raw, y_raw, edge_linear_points):
    """
    Overwrites the first `edge_linear_points` and last `edge_linear_points`
    samples of (x_smooth, y_smooth) with a straight line.

    - The first N points become a straight line from the *actual* first raw
      point (x_raw[0], y_raw[0]) to the already-computed smoothed value at
      index N (i.e. the (N+1)-th point, which is left untouched).
    - The last N points become a straight line from the *actual* last raw
      point (x_raw[-1], y_raw[-1]) to the already-computed smoothed value at
      index -(N+1) (the (N+1)-th point counting from the end, also left
      untouched).

    This pins the path's extremities to their known/observed endpoints
    instead of letting the polynomial fit (which can wobble near the edges
    where it has less context) determine their position.

    Mutates x_smooth / y_smooth in place and also returns them.
    """
    n = int(edge_linear_points)
    num_points_M = len(x_smooth)

    if n <= 0:
        return x_smooth, y_smooth

    if 2 * n >= num_points_M:
        # Not enough points to carve out two non-overlapping N-point edges;
        # leave the path as computed rather than risk index errors / a
        # degenerate (overlapping) linear region.
        return x_smooth, y_smooth

    # ── Leading edge: straight line from the first raw point to the
    #    already-computed smoothed point at index N ──
    x_target_start, y_target_start = x_smooth[n], y_smooth[n]
    x_anchor_start, y_anchor_start = x_raw[0], y_raw[0]
    for i in range(n):
        w = i / n  # 0 at the anchor, approaching (but not reaching) 1 at i=n
        x_smooth[i] = (1 - w) * x_anchor_start + w * x_target_start
        y_smooth[i] = (1 - w) * y_anchor_start + w * y_target_start

    # ── Trailing edge: straight line from the last raw point to the
    #    already-computed smoothed point at index -(n+1) ──
    x_target_end, y_target_end = x_smooth[-(n + 1)], y_smooth[-(n + 1)]
    x_anchor_end, y_anchor_end = x_raw[-1], y_raw[-1]
    for i in range(n):
        idx = num_points_M - 1 - i
        w = i / n  # 0 at the anchor (last point), approaching 1 toward the target
        x_smooth[idx] = (1 - w) * x_anchor_end + w * x_target_end
        y_smooth[idx] = (1 - w) * y_anchor_end + w * y_target_end

    return x_smooth, y_smooth


def smooth_path_by_polyfit_sections(points, num_sections=4, poly_degree=2, blend_percentage=0.2,
                                     edge_linear_points=0):
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
    edge_linear_points : int, default=0
        If > 0, the first N and last N output points (N = edge_linear_points)
        are overwritten with a straight line running from the actual first/last
        raw point to the already-computed smoothed point at the (N+1)-th
        position from that edge. Set to 0 (default) to disable and keep the
        original polynomial-fit behavior at the edges.

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

    x_smooth, y_smooth = _linearize_edges(x_smooth, y_smooth, x_raw, y_raw, edge_linear_points)

    return x_smooth, y_smooth


def smooth_path_by_segments_with_overlap(points, num_sections=4, poly_degree=2, extra_points=8, blend_percentage=0.2,
                                          edge_linear_points=0):
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
    edge_linear_points : int, default=0
        If > 0, the first N and last N output points (N = edge_linear_points)
        are overwritten with a straight line running from the actual first/last
        raw point to the already-computed smoothed point at the (N+1)-th
        position from that edge. Set to 0 (default) to disable and keep the
        original polynomial-fit behavior at the edges.

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

    x_smooth, y_smooth = _linearize_edges(x_smooth, y_smooth, x_raw, y_raw, edge_linear_points)

    return x_smooth, y_smooth