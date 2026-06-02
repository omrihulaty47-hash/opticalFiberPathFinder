import numpy as np
import matplotlib.pyplot as plt
import fiber_simulation
import point_est
import generate_anchors
import point_smoother

N = 5
r = 1

known_anchors = generate_anchors.generate_linear_anchors(np.array([1500, 1500]), 3000/N, N, 90)
real_anchors = generate_anchors.perturb_points_max_1m(known_anchors)
fiber_X, fiber_Y, _, start_point, end_point = fiber_simulation.generate_ultra_smooth_path()

num_points = len(fiber_X)

# --- 1. FORWARD PASS ---
est_forward = np.zeros((num_points, 2))
est_forward[0] = start_point

for i in range(1, num_points):
    est_forward[i] = 0
    for j in range(r):
        dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], real_anchors)
        # Note: Passing the previous point 'est_forward[i-1]' for step constraints
        point = point_est.estimate_single_point(known_anchors, dists, i, est_forward[i-1])
        est_forward[i] += point
    est_forward[i] /= r

# --- 2. BACKWARD PASS ---
est_backward = np.zeros((num_points, 2))
est_backward[-1] = end_point

# Loop backwards from the second-to-last point down to 0
for i in range(num_points - 2, -1, -1):
    est_backward[i] = 0
    for j in range(r):
        dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], real_anchors)
        # Note: In backward pass, the "previous" physical step is at index 'i+1'
        point = point_est.estimate_single_point(known_anchors, dists, i, est_backward[i+1])
        est_backward[i] += point
    est_backward[i] /= r

# --- 3. DYNAMIC WEIGHTED FUSION ---
est = np.zeros((num_points, 2))

for i in range(num_points):
    # Calculate a weight from 0.0 to 1.0 based on position along the fiber
    # i=0 (Start): w_backward = 0.0 -> 100% Forward Pass
    # i=middle:    w_backward = 0.5 -> 50% Forward, 50% Backward
    # i=end:       w_backward = 1.0 -> 100% Backward Pass
    w_backward = i / (num_points - 1)
    w_forward = 1.0 - w_backward
    
    # Apply the distance-based advantage
    est[i] = (w_forward * est_forward[i]) + (w_backward * est_backward[i])

# est_x ,est_y = point_smoother.smooth_path_by_polyfit_sections(est, 10, 5 , 0)
est_x ,est_y = point_smoother.smooth_path_by_segments_with_overlap(est, 10, 10, 50, 0)


plt.figure(figsize=(10, 6))
plt.plot(fiber_X, fiber_Y, label="True path", linewidth=2)
plt.scatter(est[:,0], est[:,1], label="Raw estimates", s=10, alpha=0.4)
plt.plot(est_x, est_y, label="Smoothed estimate", linewidth=2, linestyle="--")
plt.title("Fiber Path Estimation")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.tight_layout()
plt.show()
 
# ── Plot 2: Error along path ───────────────────────────────────────────────────
# 1. Convert your estimated path into a continuous function of Y.
# We sort the arrays by est_y to ensure np.interp works correctly (it requires independent variables to be strictly increasing).
sort_idx = np.argsort(est_y)
est_y_sorted = est_y[sort_idx]
est_x_sorted = est_x[sort_idx]

# 2. Interp allows us to input the fiber's Y positions and get what the 
# Estimated X *would have been* at those exact same Y heights.
est_x_at_fiber_y = np.interp(fiber_Y, est_y_sorted, est_x_sorted)

# 3. Calculate the clean, point-to-point horizontal distance at identical Y levels
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

