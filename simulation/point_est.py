"""
Trilateration and Sensor Modeling Engine
========================================
This module contains the core mathematical functions for modeling noisy sensor 
distance profiles and executing inverse non-linear optimization algorithms 
to estimate 2D target coordinates.
"""

import numpy as np
from scipy.optimize import minimize

def generate_noisy_distances(true_point, anchors, time_error_ms=5.0, speed_of_sound=343.0, hearing_range=500):
    """
    Calculates exact geometric distances from a target point to N anchors,
    injects Gaussian measurement noise from timing errors, and drops out-of-range signals.
    
    Parameters:
    -----------
    true_point : array-like
        The actual [X, Y] coordinates of the target subject.
    anchors : numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the tracking sensors.
    time_error_ms : float, default=5.0
        The standard deviation of the receiver timing error in milliseconds.
    speed_of_sound : float, default=343.0
        The medium velocity in m/s (default speed of sound at ~20°C).
    hearing_range : float, default=500
        The maximum physical distance (meters) a signal can travel before dropping out.
        
    Returns:
    --------
    numpy.ndarray
        A 1D array of length N containing noisy distance measurements. 
        Out-of-range values are marked with 0.0.
    """
    # 1. Convert structural timing jitter (milliseconds) to spatial noise scales (meters)
    # Example: 5ms error = 0.005 seconds * 343 m/s = ~1.715 meters of standard deviation
    error_std_meters = (time_error_ms / 1000.0) * speed_of_sound
    
    # 2. Compute the true Euclidean distance vectors across row axes
    true_distances = np.linalg.norm(anchors - true_point, axis=1)
    
    # 3. Generate uncorrelated zero-mean Gaussian noise profiles for each sensor channel
    num_anchors = len(anchors)
    noise = np.random.normal(loc=0.0, scale=error_std_meters, size=num_anchors)
    
    # 4. Synthesize the noisy measurements
    dists = true_distances + noise
    
    # 5. Acoustic Horizon Constraint: Explicitly mute signals that exceed the maximum range.
    # We use 0.0 to match the mask logic inside the inverse estimation solver.
    dists[true_distances > hearing_range] = 0.0
    return dists


def estimate_single_point(anchors, noisy_distances, i, prev_point):
    """
    Estimates the 2D coordinates of an unknown target node by minimizing 
    distance residuals against active anchors.
    
    Parameters:
    -----------
    anchors : numpy.ndarray
        An (N, 2) array containing the known coordinates of all deployed sensors.
    noisy_distances : numpy.ndarray
        A 1D array of length N containing the range outputs from generate_noisy_distances.
    i : int
        The sequential step index along the tracking path (used for initial guess biasing).
    prev_point : numpy.ndarray
        The estimated [X, Y] coordinates of the previous sequential node.

    Returns:
    --------
    numpy.ndarray
        The optimized [X, Y] coordinate array for the current tracking step.
    """
    # 1. Active Channel Masking: Filter out signals that dropped out (marked as 0)
    # This keeps out-of-range anchors from feeding false telemetry into the optimizer.
    audible_mask = noisy_distances > 0
    
    filtered_anchors = anchors[audible_mask]
    filtered_distances = noisy_distances[audible_mask]
    
    # Kinematic Fallback: If the target loses contact with all anchors,
    # it cannot be trilaterated. Assume zero velocity and hold the previous position.
    if len(filtered_anchors) == 0:
        return prev_point

    # 2. Non-Linear Least-Squares Formulation
    def loss_function(guessed_point):
        """Calculates the sum of squared residuals between a guessed coordinate and real measurements."""
        # Calculate distance from our current guess to every visible anchor
        calculated_distances = np.linalg.norm(filtered_anchors - guessed_point, axis=1)
        
        # Residual = Mathematical Distance minus Measured Distance
        residuals = calculated_distances - filtered_distances
        
        # Minimize the objective scalar
        return np.sum(residuals**2)
    
    # 3. Initial Guess Allocation
    # Seeds the Quasi-Newton optimizer near the expected track centerline to accelerate convergence.
    initial_guess = prev_point
    
    # 4. Gradient Minimization Execution
    # Runs the BFGS (Broyden–Fletcher–Goldfarb–Shanno) Quasi-Newton method.
    result = minimize(loss_function, initial_guess, method='BFGS')
    
    # 5. Kinematic Step Restriction (Hard Clamp Constraint)
    # Prevents the optimization from making massive, non-physical spatial jumps due to noise.
    # If the solver jumps more than 1.0 meter from the previous position, cap the step length to 1.0.
    distance_to_prev = np.linalg.norm(result.x - prev_point)
    # if distance_to_prev > 5.0:
    #     direction = (result.x - prev_point) / distance_to_prev
    #     result.x = prev_point + direction

    return result.x