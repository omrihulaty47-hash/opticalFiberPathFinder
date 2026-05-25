import numpy as np
import matplotlib.pyplot as plt
import fiber_simulation
import point_est
import generate_anchors
import point_smoother

N = 10

anchors = generate_anchors.generate_circular_anchors(np.array([0,1500]), 2000, N)
fiber_X, fiber_Y, _ = fiber_simulation.generate_ultra_smooth_path()

est = np.zeros((len(fiber_X),2))

for i in range(len(fiber_X)):
    dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], anchors)
    point = point_est.estimate_single_point(anchors, dists)
    est[i] = point

p, est_x ,est_y = point_smoother.smooth_path_by_sections(est, 10)

# plt.plot(fiber_X, fiber_Y, label="true_line")
# plt.scatter(est[:,0], est[:,1], label="primitive est")
# plt.plot(est_x, est_y, label="fit est")
# plt.legend()
# plt.show()

dists_points = np.zeros(len(fiber_Y))
dists_line = np.zeros(len(fiber_Y))

for i in range(len(fiber_Y)):
    dists_points[i] = np.linalg.norm([fiber_X[i] - est[i,0], fiber_Y[i] - est[i,1]])
    dists_line[i] = abs(fiber_X[i] - p(fiber_Y[i]))


plt.plot(fiber_Y, dists_line, label="line")
plt.plot(fiber_Y, dists_points, label="points")
plt.legend()
plt.show()

