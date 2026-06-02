import numpy as np
import matplotlib.pyplot as plt

import fiber_simulation
import point_est
import generate_anchors
import point_smoother


def run_single_trial(anchors, repetes, num_points=1001):
    """
    Runs one full simulation trial using two algorithms:
    1. Forward-Only estimation
    2. Bidirectional (Forward + Backward combined) estimation
    
    Returns metrics for both approaches after Y-aligned curve smoothing.
    """
    known_anchors = anchors
    real_anchors = generate_anchors.perturb_points_max_1m(known_anchors)
    fiber_X, fiber_Y, _, start_point, end_point = fiber_simulation.generate_ultra_smooth_path()

    # --- 1. FORWARD PASS ---
    est_forward = np.zeros((num_points, 2))
    est_forward[0] = start_point
    for i in range(1, num_points):
        for _ in range(repetes):
            dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], real_anchors)
            est_forward[i] += point_est.estimate_single_point(known_anchors, dists, i, est_forward[i-1])
        est_forward[i] /= repetes

    # --- 2. BACKWARD PASS ---
    est_backward = np.zeros((num_points, 2))
    est_backward[-1] = end_point
    for i in range(num_points - 2, -1, -1):
        for _ in range(repetes):
            dists = point_est.generate_noisy_distances([fiber_X[i], fiber_Y[i]], real_anchors)
            est_backward[i] += point_est.estimate_single_point(known_anchors, dists, i, est_backward[i+1])
        est_backward[i] /= repetes

    # --- 3. DYNAMIC WEIGHTED FUSION (BIDIRECTIONAL) ---
    est_bidirectional = np.zeros((num_points, 2))
    for i in range(num_points):
        w_backward = i / (num_points - 1)
        w_forward = 1.0 - w_backward
        est_bidirectional[i] = (w_forward * est_forward[i]) + (w_backward * est_backward[i])

    # --- 4. SMOOTH BOTH PATHS ---
    smooth_fwd_x, smooth_fwd_y = point_smoother.smooth_path_by_segments_with_overlap(est_forward, 10, 8, 50, 0)
    smooth_bi_x, smooth_bi_y = point_smoother.smooth_path_by_segments_with_overlap(est_bidirectional, 10, 8, 50, 0)

    # --- 5. EVALUATE FORWARD-ONLY ERRORS (Y-ALIGNED) ---
    sort_fwd = np.argsort(smooth_fwd_y)
    fwd_x_at_fiber_y = np.interp(fiber_Y, smooth_fwd_y[sort_fwd], smooth_fwd_x[sort_fwd])
    fwd_errors = np.abs(fiber_X - fwd_x_at_fiber_y)

    # --- 6. EVALUATE BIDIRECTIONAL ERRORS (Y-ALIGNED) ---
    sort_bi = np.argsort(smooth_bi_y)
    bi_x_at_fiber_y = np.interp(fiber_Y, smooth_bi_y[sort_bi], smooth_bi_x[sort_bi])
    bi_errors = np.abs(fiber_X - bi_x_at_fiber_y)

    # Return slicing excludes the last 100 points
    return (
        np.mean(fwd_errors[:-100]), np.max(fwd_errors[:-100]),
        np.mean(bi_errors[:-100]), np.max(bi_errors[:-100])
    )


def run_distribution(num_trials=50, num_anchors=10, repetes=1, num_points=1001):
    """
    Runs multiple trials and collects comparative statistics for both algorithms.
    """
    anchors = generate_anchors.generate_linear_anchors(np.array([1500, 1500]), 3000/num_anchors, num_anchors, 90)

    fwd_avg_dists, fwd_max_dists = [], []
    bi_avg_dists, bi_max_dists = [], []

    print(f"Running {num_trials} trials...")
    for trial in range(num_trials):
        fwd_avg_d, fwd_max_d, bi_avg_d, bi_max_d = run_single_trial(anchors, repetes, num_points=num_points)
        
        fwd_avg_dists.append(fwd_avg_d)
        fwd_max_dists.append(fwd_max_d)
        bi_avg_dists.append(bi_avg_d)
        bi_max_dists.append(bi_max_d)
        
        print(f"  Trial {trial + 1:>3}/{num_trials} | "
              f"FWD: avg={fwd_avg_d:.3f}m max={fwd_max_d:.3f}m | "
              f"BI: avg={bi_avg_d:.3f}m max={bi_max_d:.3f}m")

    fwd_avg_dists = np.array(fwd_avg_dists)
    fwd_max_dists = np.array(fwd_max_dists)
    bi_avg_dists = np.array(bi_avg_dists)
    bi_max_dists = np.array(bi_max_dists)

    print("\n--- Summary (Forward-Only) ---")
    print(f"Avg distance:  mean={fwd_avg_dists.mean():.3f}m  std={fwd_avg_dists.std():.3f}m  max={fwd_avg_dists.max():.3f}m")
    print(f"Max distance:  mean={fwd_max_dists.mean():.3f}m  std={fwd_max_dists.std():.3f}m  max={fwd_max_dists.max():.3f}m")

    print("\n--- Summary (Bidirectional) ---")
    print(f"Avg distance:  mean={bi_avg_dists.mean():.3f}m  std={bi_avg_dists.std():.3f}m  max={bi_avg_dists.max():.3f}m")
    print(f"Max distance:  mean={bi_max_dists.mean():.3f}m  std={bi_max_dists.std():.3f}m  max={bi_max_dists.max():.3f}m")

    return fwd_avg_dists, fwd_max_dists, bi_avg_dists, bi_max_dists


def plot_distributions(fwd_avg_dists, fwd_max_dists, bi_avg_dists, bi_max_dists):
    # --- Original Distribution Plot 1: Average Distance ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution of Point Estimation Error Across Trials", fontsize=14)

    ax = axes[0]
    ax.hist(fwd_avg_dists, bins=15, color='steelblue', edgecolor='white', alpha=0.6, label='Fwd Mean')
    ax.hist(bi_avg_dists, bins=15, color='seagreen', edgecolor='white', alpha=0.6, label='Bi Mean')
    ax.axvline(fwd_avg_dists.mean(), color='navy', linestyle='--', label=f'Fwd: {fwd_avg_dists.mean():.2f}m')
    ax.axvline(bi_avg_dists.mean(), color='darkgreen', linestyle='--', label=f'Bi: {bi_avg_dists.mean():.2f}m')
    ax.set_title("Average Distance Distribution")
    ax.set_xlabel("Average distance (m)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # --- Original Distribution Plot 2: Max Distance ---
    ax = axes[1]
    ax.hist(fwd_max_dists, bins=15, color='tomato', edgecolor='white', alpha=0.6, label='Fwd Max')
    ax.hist(bi_max_dists, bins=15, color='darkorange', edgecolor='white', alpha=0.6, label='Bi Max')
    ax.axvline(fwd_max_dists.mean(), color='darkred', linestyle='--', label=f'Fwd: {fwd_max_dists.mean():.2f}m')
    ax.axvline(bi_max_dists.mean(), color='chocolate', linestyle='--', label=f'Bi: {bi_max_dists.mean():.2f}m')
    ax.set_title("Max Distance Distribution")
    ax.set_xlabel("Max distance (m)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- NEW PLOT 1: Run Number vs Average Error ---
    runs = np.arange(1, len(fwd_avg_dists) + 1)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(runs, fwd_avg_dists, marker='o', color='steelblue', linewidth=1.5, label='Forward-Only Algorithm')
    ax2.plot(runs, bi_avg_dists, marker='s', color='seagreen', linewidth=1.5, label='Bidirectional Algorithm')
    ax2.set_title("Average Estimation Error Comparison Per Run")
    ax2.set_xlabel("Run Number")
    ax2.set_ylabel("Average Error (meters)")
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- NEW PLOT 2: Run Number vs Max Error ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(runs, fwd_max_dists, marker='o', color='tomato', linewidth=1.5, label='Forward-Only Algorithm')
    ax3.plot(runs, bi_max_dists, marker='s', color='darkorange', linewidth=1.5, label='Bidirectional Algorithm')
    ax3.set_title("Maximum Peak Error Comparison Per Run")
    ax3.set_xlabel("Run Number")
    ax3.set_ylabel("Max Peak Error (meters)")
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NUM_TRIALS = 50   # Increase for smoother metrics
    NUM_POINTS = 1001

    fwd_avg, fwd_max, bi_avg, bi_max = run_distribution(
        num_trials=NUM_TRIALS,
        num_anchors=5,
        repetes=1,
        num_points=NUM_POINTS
    )

    plot_distributions(fwd_avg, fwd_max, bi_avg, bi_max)