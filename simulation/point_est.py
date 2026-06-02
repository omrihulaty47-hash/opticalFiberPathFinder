import numpy as np
from scipy.optimize import minimize

def generate_noisy_distances(true_point, anchors, time_error_ms=5.0, speed_of_sound=343.0):
    """
    Calculates the exact distances from a true coordinate point to N anchors,
    then injects random Gaussian measurement noise modeled from acoustic timing errors.
    
    Parameters:
    -----------
    true_point : numpy.ndarray or list
        The real [X, Y] coordinates of the target subject.
    anchors : numpy.ndarray
        An (N, 2) array containing the [X, Y] coordinates of the anchors.
    time_error_ms : float
        The standard deviation of the sensor timing error in milliseconds.
    speed_of_sound : float
        The speed of sound in m/s (default 343.0 m/s at ~20°C).
        
    Returns:
    --------
    numpy.ndarray
        A 1D array of length N containing the noisy distance measurements.
    """
    # 1. Convert the timing error from milliseconds to meters
    # 5ms error = 0.005 seconds * 343 m/s = ~1.715 meters of standard deviation
    error_std_meters = (time_error_ms / 1000.0) * speed_of_sound
    
    # 2. Calculate the TRUE geometric distance to each anchor
    # axis=1 computes the norm across coordinates for each individual anchor row
    true_distances = np.linalg.norm(anchors - true_point, axis=1)
    
    # 3. Generate a unique random Gaussian noise sample for each anchor
    num_anchors = len(anchors)
    noise = np.random.normal(loc=0.0, scale=error_std_meters, size=num_anchors)
    
    # 4. Return the combined noisy distance profile
    return true_distances + noise

def estimate_single_point(anchors, noisy_distances, i, prev_point):
    """
    Estimates the coordinates of a single unknown point based on 
    measured distances from N known anchors.
    
    Parameters:
    -----------
    anchors : numpy.ndarray
        An (N, 2) array containing the (X, Y) coordinates of the anchors.
    noisy_distances : numpy.ndarray
        A 1D array of length N containing the measured distances from 
        each anchor to the unknown point.
        
    Returns:
    --------
    numpy.ndarray
        A 1D array [x, y] of the optimized coordinate estimation.
    """
    # 1. Define the cost function for a single coordinate guess
    def loss_function(guessed_point):
        
        # Calculate the Euclidean distance from the guess to all anchors
        calculated_distances = np.linalg.norm(anchors - guessed_point, axis=1)
        # Residuals represent the difference between reality and our guess
        residuals = calculated_distances - noisy_distances
        # Return the sum of squared errors to minimize
        return np.sum(residuals**2)
    
    # 2. Establish an initial guess at the center geometric centroid of the anchors
    initial_guess = np.array([0, i*3 + np.random.randint(-10, 10)])
    
    # 3. Optimize using Quasi-Newton gradient descent (BFGS)
    result = minimize(loss_function, initial_guess, method='BFGS')
    
    # Check if the optimized point jumped further than 3 meters
    distance_to_prev = np.linalg.norm(result.x - prev_point)

    if distance_to_prev > 3.0:
        # 1. Find the unit direction vector from prev_point to result.x
        direction = (result.x - prev_point) / distance_to_prev
        
        # 2. Overwrite result.x to be exactly 3 meters away along that same line
        result.x = prev_point + direction * 3.0

    return result.x