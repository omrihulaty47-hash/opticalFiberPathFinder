"""
Fiber Optic Geometry Tracking and Localization Pipeline
=======================================================
This script simulates a unidirectional (forward-only) sequential estimation engine 
to reconstruct the 2D geometry of a linear asset. Tracking stability relies strictly 
on the historical spatial boundary of the previous node.
"""

import numpy as np
import matplotlib.pyplot as plt

# Custom project modules for localization physics, path generation, and filtering
import fiber_simulation
import point_est
import generate_anchors
import point_smoother

# ==============================================================================
# 1. SIMULATION ENVIRONMENT CONFIGURATION
# ==============================================================================

# N: Number of physical anchor nodes/transceivers deployed along the field
N = 100

number_of_points = 3001

hearing_range = 1000
# r: Oversampling factor. Number of independent temporal measurements taken 
# per coordinate node to average out zero-mean Gaussian noise.
r = 1

linearize_edges = True

# Generate ideal linear tracking anchor coordinates spanning from Y = 100m to Y = 1500m.
known_anchors = generate_anchors.generate_staggered_anchors(np.array([0, 1500]), 3000/N, N, 90, 4)

# Simulate deployment error (GPS surveying inaccuracies).
real_anchors = generate_anchors.perturb_points_max_1m(known_anchors)

# Generate the true underlying continuous asset path (ground-truth).
fiber_X, fiber_Y, _, start_point, end_point = \
        fiber_simulation.generate_ultra_smooth_path(number_of_points)

    # --- FORWARD PASS ---
est    = np.zeros((number_of_points, 2))
est[0] = start_point

for i in range(1, number_of_points):
    for _ in range(r):
        dists = point_est.generate_noisy_distances(
            [fiber_X[i], fiber_Y[i]], real_anchors,
            hearing_range=hearing_range,
        ) 
        # this is a parat of the algorithm - the real anchors parameter (which is unknown to the algorithm) 
        # is only used here to genrerate the distances the channel would detect in the real world.
        est[i] += point_est.estimate_single_point(
            known_anchors, dists, i, est[i - 1]
        )
    est[i] /= r

est[-1] = end_point

# --- SMOOTH ---
est_x, est_y = \
    point_smoother.smooth_path_by_segments_with_overlap(
        est, 20, 6, 50, 0, 50 if linearize_edges else 0
    )


# ==============================================================================
# 3. POST-PROCESS SMOOTHING
# ==============================================================================

# Apply an overlapping segment-based polynomial smoothing filter to the raw forward estimates


# ==============================================================================
# 4. DATA VISUALIZATION & GEOMETRIC METRIC EVALUATION
# ==============================================================================

# --- PLOT 1: Spatial Geometry Overview ---
plt.figure(figsize=(10, 6))
plt.plot(fiber_X, fiber_Y, label="True path", linewidth=2)
plt.scatter(est[:,0], est[:,1], label="Raw estimates", s=10, alpha=0.4)
# plt.scatter(real_anchors[:,0], real_anchors[:,1], label="anchors")
plt.plot(est_x, est_y, label="Smoothed estimate", linewidth=2, linestyle="--")
plt.title("Fiber Path Estimation")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.tight_layout()
plt.show()

# --- PLOT 2: Cross-Sectional Registration Error Analysis ---
# Sort arrays by Y to ensure the independent variable strictly increases for 1D interpolation
sort_idx = np.argsort(est_y)
est_y_sorted = est_y[sort_idx]
est_x_sorted = est_x[sort_idx]

# Resample the estimated X path coordinates onto the true ground-truth Y coordinate positions
est_x_at_fiber_y = np.interp(fiber_Y, est_y_sorted, est_x_sorted)

# Compute absolute horizontal offset errors at identical vertical cross-sections
dists_points = np.abs(fiber_X - est_x_at_fiber_y)
 
plt.figure(figsize=(10, 4))
plt.plot(fiber_Y, dists_points, label="Point-wise error", linewidth=1.5)
plt.axhline(dists_points.mean(), color="gray", linestyle=":", linewidth=1.2,
            label=f"Mean error: {dists_points.mean():.1f} m")
plt.title("Estimation Error Along Fiber")
plt.xlabel("Y position [m]")
plt.ylabel("Error [m]")
plt.legend()
plt.tight_layout()
plt.show()