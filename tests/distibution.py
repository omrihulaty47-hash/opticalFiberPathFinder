import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats

def analyze_error_distribution(num_anchors=4, offset_distance=20.0, iterations=1000):
    # Constants
    SPEED_OF_SOUND = 343.0  # m/s
    TIME_ERROR_MS = 5.0
    error_std_meters = (TIME_ERROR_MS / 1000.0) * SPEED_OF_SOUND  # ~1.715m
    
    circle_center = np.array([100.0, 100.0])
    circle_radius = 1000.0
    
    # Place target point at the specified offset
    true_point = circle_center + np.array([offset_distance, 0.0])
    
    # Generate circular anchor setup
    anchors = []
    for i in range(num_anchors):
        angle = 2 * np.pi * i / num_anchors
        x_anchor = circle_center[0] + circle_radius * np.cos(angle)
        y_anchor = circle_center[1] + circle_radius * np.sin(angle)
        anchors.append([x_anchor, y_anchor])
    anchors = np.array(anchors)
    
    true_distances = np.linalg.norm(anchors - true_point, axis=1)
    
    errors = []
    
    # Perform 1,000 independent trials to map the distribution spread
    for _ in range(iterations):
        noise = np.random.normal(0, error_std_meters, size=num_anchors)
        noisy_distances = true_distances + noise
        
        def loss_function(guessed_point):
            calculated_distances = np.linalg.norm(anchors - guessed_point, axis=1)
            return np.sum((calculated_distances - noisy_distances)**2)
        
        result = minimize(loss_function, circle_center, method='BFGS')
        trial_error = np.linalg.norm(result.x - true_point)
        errors.append(trial_error)
        
    errors = np.array(errors)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    
    # Plot normalized histogram of errors
    count, bins, ignored = plt.hist(errors, bins=30, density=True, alpha=0.6, 
                                    color='skyblue', edgecolor='black', label='Simulated Errors')
    
    # Fit a Rayleigh distribution curve to the data
    # (Rayleigh distributions model the magnitude of 2D directional vectors)
    param = stats.rayleigh.fit(errors)
    pdf_fitted = stats.rayleigh.pdf(bins, loc=param[0], scale=param[1])
    plt.plot(bins, pdf_fitted, 'r-', linewidth=2.5, label='Fitted Error Distribution Curve')
    
    # Add statistical benchmark lines
    mean_err = np.mean(errors)
    p95_err = np.percentile(errors, 95)
    
    plt.axvline(mean_err, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Mean Error: {mean_err:.2f}m')
    plt.axvline(p95_err, color='purple', linestyle=':', linewidth=2, 
                label=f'95th Percentile: {p95_err:.2f}m')
    
    # Formatting
    plt.title(f'Error Distribution Profile\nSetup: {num_anchors} Anchors | Subject Offset: {offset_distance}m (Array Radius: 50m)', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Absolute Localization Error (Meters)')
    plt.ylabel('Probability Density')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.show()
    
    # Print data recap
    print(f"--- Distribution Stats ({num_anchors} Anchors, {offset_distance}m Offset) ---")
    print(f"Minimum Error:       {np.min(errors):.2f} meters")
    print(f"Average (Mean) Error:{mean_err:.2f} meters")
    print(f"Median Error:        {np.median(errors):.2f} meters")
    print(f"95% of trials fell under: {p95_err:.2f} meters")

# Run the distribution analyzer
# You can tweak these inputs to see how the curve stretches or squishes!
analyze_error_distribution(num_anchors=10, offset_distance=0)