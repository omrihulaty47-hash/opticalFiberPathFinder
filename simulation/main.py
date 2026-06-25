"""
Fiber Optic Geometry Tracking and Localization Pipeline
=======================================================
This script simulates a unidirectional (forward-only) sequential estimation engine 
to reconstruct the 2D geometry of a linear asset. Tracking stability relies strictly 
on the historical spatial boundary of the previous node.
"""

import numpy as np
import matplotlib.pyplot as plt

# Custom project modules for localization physics and path generation.
# Note: point_est / point_smoother are no longer imported here directly -
# they're implementation details of localization_algorithm now, which is
# the only thing that needs to know about them.
import fiber_simulation
import generate_anchors
import algorithm

# ==============================================================================
# 1. SIMULATION ENVIRONMENT CONFIGURATION
# ==============================================================================

# N: Number of physical anchor nodes/transceivers deployed along the field
N = 1000

number_of_points = 3001

hearing_range = 50
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

# --- SIMULATE THE CHANNEL: generate distance measurements ---
# `real_anchors` (the deployment error - unknown to the algorithm) is only
# ever used here, to generate the noisy distances the channel would detect
# in the real world. From this point on, the algorithm only sees `distances`
# and `known_anchors`.
distances = algorithm.generate_distances(
    fiber_X, fiber_Y, real_anchors, hearing_range, number_of_points, r=r,
)

# --- RUN THE ALGORITHM (forward pass + smoothing) ---
est, est_x, est_y = algorithm.estimate_path(
    known_anchors, distances, number_of_points, start_point,
    end_point=end_point, r=r, linearize_edges=linearize_edges,
)


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
# Resample the estimated path onto the true ground-truth Y positions and
# compute the absolute horizontal offset at each cross-section.
dists_points = algorithm.compute_position_error(fiber_X, fiber_Y, est_x, est_y)
 
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