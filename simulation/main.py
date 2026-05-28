import numpy as np
import matplotlib.pyplot as plt
import fiber_simulation
import point_est
import generate_anchors
import point_smoother

N = 10

known_anchors = generate_anchors.generate_circular_anchors(np.array([0,1500]), 2000, N)
real_anchors = generate_anchors.perturb_points_max_1m(known_anchors)
fiber_X, fiber_Y, _ = fiber_simulation.generate_ultra_smooth_path()

est = np.zeros((len(fiber_X),2))

for i in range(len(fiber_X)):
    dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], real_anchors)
    point = point_est.estimate_single_point(known_anchors, dists)
    est[i] = point

# est_x ,est_y = point_smoother.smooth_path_by_polyfit_sections(est, 10, 5 , 0)
est_x ,est_y = point_smoother.smooth_path_by_segments_with_overlap(est, 10, 8, 50, 0)


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
dists_points = np.array([
    np.linalg.norm([fiber_X[i] - est_x[i], fiber_Y[i] - est_y[i]])
    for i in range(len(fiber_Y))
])
 
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

