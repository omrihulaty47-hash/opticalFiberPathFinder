import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_monte_carlo_test(num_anchors, offset_distance, iterations=100):
    SPEED_OF_SOUND = 343.0  # m/s
    TIME_ERROR_MS = 5.0
    error_std_meters = (TIME_ERROR_MS / 1000.0) * SPEED_OF_SOUND  # ~1.715m
    
    circle_center = np.array([100.0, 100.0])
    circle_radius = 1000.0
    
    # Place the subject offset to the right along the X-axis for consistency
    true_point = circle_center + np.array([offset_distance, 0.0])
    
    # Generate static anchor positions in a circle around the CENTER of the array
    anchors = []
    for i in range(num_anchors):
        angle = 2 * np.pi * i / num_anchors
        x_anchor = circle_center[0] + circle_radius * np.cos(angle)
        y_anchor = circle_center[1] + circle_radius * np.sin(angle)
        anchors.append([x_anchor, y_anchor])
    anchors = np.array(anchors)
    
    # Calculate true distances once
    true_distances = np.linalg.norm(anchors - true_point, axis=1)
    
    errors = []
    
    # Run the simulation 100 times to get an average behavior
    for _ in range(iterations):
        # Inject fresh normal noise every iteration
        noise = np.random.normal(0, error_std_meters, size=num_anchors)
        noisy_distances = true_distances + noise
        
        def loss_function(guessed_point):
            calculated_distances = np.linalg.norm(anchors - guessed_point, axis=1)
            return np.sum((calculated_distances - noisy_distances)**2)
        
        initial_guess = circle_center.copy()
        result = minimize(loss_function, initial_guess, method='BFGS')
        
        # Calculate distance error for this specific trial
        trial_error = np.linalg.norm(result.x - true_point)
        errors.append(trial_error)
        
    # Return the average error across all 100 runs
    return np.mean(errors)

# --- Define Parameters to Sweep ---
anchor_range = [3, 4, 5, 6, 8, 10, 20]              # Test different numbers of microphones
offset_range = [0, 100, 200, 300, 400, 450]          # Test target distance from the center (Radius is 50)
iterations_per_setup = 100

# Initialize an empty matrix to hold the results
results_matrix = np.zeros((len(anchor_range), len(offset_range)))

print("Running Monte Carlo Simulations (This may take a few seconds)...")
for i, n_anchors in enumerate(anchor_range):
    for j, offset in enumerate(offset_range):
        avg_error = run_monte_carlo_test(n_anchors, offset, iterations=iterations_per_setup)
        results_matrix[i, j] = avg_error
print("Done!\n")

# Convert data into a clean DataFrame for plotting
df = pd.DataFrame(results_matrix, index=anchor_range, columns=offset_range)

# --- Plotting the Accuracy Heatmap ---
plt.figure(figsize=(10, 6))
# Using standard matplotlib matrix display to avoid extra dependencies like seaborn
cax = plt.matshow(df.values, cmap='YlOrRd', fignum=1)
fig = plt.gcf()
fig.colorbar(cax, label='Mean Position Error (Meters)')

# Formatting Grid Labels
plt.title(f'Triangulation Error Heatmap\n(Average of {iterations_per_setup} Runs per Cell)', pad=20, fontweight='bold')
plt.ylabel('Number of Anchors')
plt.xlabel('Subject Offset from Center (Meters) [Array Radius = 50m]')

# Set tick locations
plt.gca().set_xticks(np.arange(len(offset_range)))
plt.gca().set_yticks(np.arange(len(anchor_range)))
# Apply labels
plt.gca().set_xticklabels(offset_range)
plt.gca().set_yticklabels(anchor_range)

# Move X-axis labels to the bottom where they belong normally
plt.gca().xaxis.set_ticks_position('bottom')

# Text annotations inside the boxes for absolute clarity
for i in range(len(anchor_range)):
    for j in range(len(offset_range)):
        plt.text(j, i, f"{df.iloc[i, j]:.2f}m", ha='center', va='center', 
                 color='black' if df.iloc[i, j] < 2.0 else 'white', fontweight='bold')

plt.show()

# --- Print the Text Summary ---
print("--- Mean Error Summary Table ---")
print(df.round(3))