import numpy as np
import matplotlib.pyplot as plt

import fiber_simulation
import point_est
import generate_anchors
import point_smoother


def run_single_trial(anchors, num_points=1001):
    """
    Runs one full simulation trial:
    - Generates a fiber path
    - Estimates each point using noisy distances
    - Smooths the estimated path
    - Returns the average and max point-wise distances from the true path
    """
    fiber_X, fiber_Y, _ = fiber_simulation.generate_ultra_smooth_path(num_points_M=num_points)

    est = np.zeros((num_points, 2))
    for i in range(num_points):
        dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], anchors)
        est[i] = point_est.estimate_single_point(anchors, dists)

    est_x, est_y = point_smoother.smooth_path_by_segments_with_overlap(est, 10, 8, 50, 0)

    point_errors = np.sqrt((fiber_X - est_x) ** 2 + (fiber_Y - est_y) ** 2)

    return np.mean(point_errors[:-100]), np.max(point_errors[:-100])


def run_distribution(num_trials=50, num_anchors=10, num_points=1001):
    """
    Runs multiple trials and collects average and max distance statistics.
    """
    anchors = generate_anchors.generate_circular_anchors(np.array([0, 1500]), 2000, num_anchors)

    avg_dists = []
    max_dists = []

    print(f"Running {num_trials} trials...")
    for trial in range(num_trials):
        avg_d, max_d = run_single_trial(anchors, num_points=num_points)
        avg_dists.append(avg_d)
        max_dists.append(max_d)
        print(f"  Trial {trial + 1:>3}/{num_trials}  avg={avg_d:.3f}m  max={max_d:.3f}m")

    avg_dists = np.array(avg_dists)
    max_dists = np.array(max_dists)

    print("\n--- Summary ---")
    print(f"Avg distance:  mean={avg_dists.mean():.3f}m  std={avg_dists.std():.3f}m  "
          f"min={avg_dists.min():.3f}m  max={avg_dists.max():.3f}m")
    print(f"Max distance:  mean={max_dists.mean():.3f}m  std={max_dists.std():.3f}m  "
          f"min={max_dists.min():.3f}m  max={max_dists.max():.3f}m")

    return avg_dists, max_dists


def plot_distributions(avg_dists, max_dists):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution of Point Estimation Error Across Trials", fontsize=14)

    # --- Average distance distribution ---
    ax = axes[0]
    ax.hist(avg_dists, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(avg_dists.mean(), color='navy', linestyle='--', linewidth=1.5,
               label=f'Mean: {avg_dists.mean():.2f}m')
    ax.axvline(np.median(avg_dists), color='cornflowerblue', linestyle=':', linewidth=1.5,
               label=f'Median: {np.median(avg_dists):.2f}m')
    ax.set_title("Average Distance per Trial")
    ax.set_xlabel("Average distance (m)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # --- Max distance distribution ---
    ax = axes[1]
    ax.hist(max_dists, bins=15, color='tomato', edgecolor='white', alpha=0.85)
    ax.axvline(max_dists.mean(), color='darkred', linestyle='--', linewidth=1.5,
               label=f'Mean: {max_dists.mean():.2f}m')
    ax.axvline(np.median(max_dists), color='salmon', linestyle=':', linewidth=1.5,
               label=f'Median: {np.median(max_dists):.2f}m')
    ax.set_title("Max Distance per Trial")
    ax.set_xlabel("Max distance (m)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- Overlay plot ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.hist(avg_dists, bins=15, color='steelblue', edgecolor='white', alpha=0.7, label='Avg distance')
    ax2.hist(max_dists, bins=15, color='tomato', edgecolor='white', alpha=0.7, label='Max distance')
    ax2.axvline(avg_dists.mean(), color='navy', linestyle='--', linewidth=1.5)
    ax2.axvline(max_dists.mean(), color='darkred', linestyle='--', linewidth=1.5)
    ax2.set_title("Avg vs Max Distance Distribution (Overlaid)")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NUM_TRIALS = 50   # increase for smoother distributions
    NUM_ANCHORS = 10
    NUM_POINTS = 1001

    avg_dists, max_dists = run_distribution(
        num_trials=NUM_TRIALS,
        num_anchors=NUM_ANCHORS,
        num_points=NUM_POINTS
    )

    plot_distributions(avg_dists, max_dists)