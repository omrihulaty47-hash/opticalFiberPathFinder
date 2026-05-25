import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def smooth_path_by_polyfit_sections(points, num_sections=4, poly_degree=2, blend_percentage=0.2):
    """
    Smooths an (M, 2) path using np.polyfit across sections.
    Returns an output array of the EXACT same shape (M, 2) as the input.
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    # 1. Map our independent parametric tracker 't' 
    t = np.arange(num_points_M)
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    
    # Calculate section size based on the input length
    section_size = num_points_M / num_sections
    
    # 2. Fit polyfit coefficients for each separate section
    poly_coeffs_x = []
    poly_coeffs_y = []
    
    for s in range(num_sections):
        start_idx = int(s * section_size)
        end_idx = int(min((s + 1) * section_size, num_points_M))
        
        t_sec = t[start_idx:end_idx]
        x_sec = x_raw[start_idx:end_idx]
        y_sec = y_raw[start_idx:end_idx]
        
        poly_coeffs_x.append(np.polyfit(t_sec, x_sec, poly_degree))
        poly_coeffs_y.append(np.polyfit(t_sec, y_sec, poly_degree))
        
    # CRITICAL CHANGE: Evaluate at the EXACT original 't' indices to match input size
    x_smooth = np.zeros_like(x_raw)
    y_smooth = np.zeros_like(y_raw)
    
    # 3. Evaluate the polynomials with a smooth blend at the boundary edges
    for idx, t_val in enumerate(t):
        sec_float = t_val / section_size
        current_sec = int(np.floor(sec_float))
        current_sec = min(current_sec, num_sections - 1)
        
        local_t = sec_float - current_sec
        
        val_x_curr = np.polyval(poly_coeffs_x[current_sec], t_val)
        val_y_curr = np.polyval(poly_coeffs_y[current_sec], t_val)
        
        # Blend with NEXT section
        if current_sec < num_sections - 1 and local_t > (1.0 - blend_percentage):
            w = (local_t - (1.0 - blend_percentage)) / blend_percentage
            w = w * w * (3 - 2 * w) 
            
            val_x_next = np.polyval(poly_coeffs_x[current_sec + 1], t_val)
            val_y_next = np.polyval(poly_coeffs_y[current_sec + 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_curr + w * val_x_next
            y_smooth[idx] = (1 - w) * val_y_curr + w * val_y_next
            
        # Blend with PREVIOUS section
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

def smooth_path_by_segments_with_overlap(points, num_sections=4, poly_degree=2, extra_points=8, blend_percentage=0.2):
    """
    Smooths an (M, 2) path by dividing it into fixed segments.
    Each segment takes 'extra_points' backwards and forwards to fit the np.polyfit models,
    but only maps the output to the original section's indices with boundary blending.
    
    Returns an array of the exact same shape (M, 2).
    """
    points = np.asarray(points)
    num_points_M = points.shape[0]
    
    t = np.arange(num_points_M)
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    
    # Define rigid segment boundaries
    section_size = num_points_M / num_sections
    
    poly_coeffs_x = []
    poly_coeffs_y = []
    
    # 1. Fit Phase: Loop through the fixed segments
    for s in range(num_sections):
        start_idx = int(s * section_size)
        end_idx = int(min((s + 1) * section_size, num_points_M))
        
        # SYMMETRICAL EXPANSION: Reach backwards and forwards for context
        fit_start = max(0, start_idx - extra_points)
        fit_end = min(num_points_M, end_idx + extra_points)
        
        t_fit = t[fit_start:fit_end]
        x_fit = x_raw[fit_start:fit_end]
        y_fit = y_raw[fit_start:fit_end]
        
        # Fit the polynomial utilizing the extra points in both directions
        poly_coeffs_x.append(np.polyfit(t_fit, x_fit, poly_degree))
        poly_coeffs_y.append(np.polyfit(t_fit, y_fit, poly_degree))
        
    # Pre-allocate output arrays matching the exact original input shape
    x_smooth = np.zeros_like(x_raw)
    y_smooth = np.zeros_like(y_raw)
    
    # 2. Evaluation Phase: Loop through every original point index
    for idx, t_val in enumerate(t):
        # Identify which core segment the point belongs to
        sec_float = t_val / section_size
        current_sec = int(np.floor(sec_float))
        current_sec = min(current_sec, num_sections - 1)
        
        local_t = sec_float - current_sec
        
        # Evaluate using the current section's polynomial equations
        val_x_curr = np.polyval(poly_coeffs_x[current_sec], t_val)
        val_y_curr = np.polyval(poly_coeffs_y[current_sec], t_val)
        
        # Blend with the NEXT section if near the forward boundary
        if current_sec < num_sections - 1 and local_t > (1.0 - blend_percentage):
            w = (local_t - (1.0 - blend_percentage)) / blend_percentage
            w = w * w * (3 - 2 * w)  # Smooth cubic blend curve
            
            val_x_next = np.polyval(poly_coeffs_x[current_sec + 1], t_val)
            val_y_next = np.polyval(poly_coeffs_y[current_sec + 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_curr + w * val_x_next
            y_smooth[idx] = (1 - w) * val_y_curr + w * val_y_next
            
        # Blend with the PREVIOUS section if near the backward boundary
        elif current_sec > 0 and local_t < blend_percentage:
            w = local_t / blend_percentage
            w = w * w * (3 - 2 * w)
            
            val_x_prev = np.polyval(poly_coeffs_x[current_sec - 1], t_val)
            val_y_prev = np.polyval(poly_coeffs_y[current_sec - 1], t_val)
            
            x_smooth[idx] = (1 - w) * val_x_prev + w * val_x_curr
            y_smooth[idx] = (1 - w) * val_y_prev + w * val_y_curr
            
        else:
            # Standard segment middle-zone evaluation
            x_smooth[idx] = val_x_curr
            y_smooth[idx] = val_y_curr
            
    return x_smooth, y_smooth