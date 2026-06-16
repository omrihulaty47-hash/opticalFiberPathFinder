"""
sweep_simulation.py
--------------------
Runs the fiber-localization simulation across a grid of:
  - num_anchors : 10, 15, 20, 25, 30
  - hearing_range: 200, 300, 400, 500, 600, 700

For every (num_anchors, hearing_range) combination the script:
  1. Runs `NUM_TRIALS` Monte-Carlo trials (forward-only pass).
  2. Saves four per-combination plots inside
       results/<num_anchors>anch_<hearing_range>range/
  3. After the full sweep, generates two heatmaps:
       • results/heatmap_avg_error.png
       • results/heatmap_max_error.png

Usage
-----
Drop this file next to your existing modules
(fiber_simulation, point_est, generate_anchors, point_smoother)
and run:

    python sweep_simulation.py

Tunable constants are at the top of the file.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for headless runs
import matplotlib.pyplot as plt

import fiber_simulation
import point_est
import generate_anchors
import point_smoother


# ──────────────────────────────────────────────
# SWEEP PARAMETERS  (edit these freely)
# ──────────────────────────────────────────────
ANCHOR_COUNTS   = list(range(25, 31, 5))          # [10, 15, 20, 25, 30]
HEARING_RANGES  = list(range(500, 701, 100))       # [200, 300, 400, 500, 600, 700]
NUM_TRIALS      = 20
NUM_POINTS      = 3000
REPETES         = 1
RESULTS_ROOT    = "extreme_test"


# ──────────────────────────────────────────────
# CORE SIMULATION  (mirrors your original code)
# ──────────────────────────────────────────────

def run_single_trial(anchors, repetes, hearing_range, num_points=3000):
    """
    One Monte-Carlo trial (forward pass only).
    `hearing_range` is passed to generate_noisy_distances so only anchors
    within that radius contribute – wire this into point_est as needed.
    Returns (mean_error, max_error).
    """
    known_anchors = anchors
    real_anchors  = generate_anchors.perturb_points_max_1m(known_anchors)

    fiber_X, fiber_Y, _, start_point, end_point = \
        fiber_simulation.generate_ultra_smooth_path()

    # --- FORWARD PASS ---
    est_forward    = np.zeros((num_points, 2))
    est_forward[0] = start_point

    for i in range(1, num_points):
        for _ in range(repetes):
            dists = point_est.generate_noisy_distances(
                [fiber_X[i], fiber_Y[i]], real_anchors,
                # Pass hearing_range if your function supports it;
                # remove the kwarg if it does not yet exist.
                hearing_range=hearing_range,
            )
            est_forward[i] += point_est.estimate_single_point(
                known_anchors, dists, i, est_forward[i - 1]
            )
        est_forward[i] /= repetes

    # --- SMOOTH ---
    smooth_fwd_x, smooth_fwd_y = \
        point_smoother.smooth_path_by_segments_with_overlap(
            est_forward, 15, 5, 50, 0
        )

    # --- Y-ALIGNED ERROR ---
    sort_fwd           = np.argsort(smooth_fwd_y)
    fwd_x_at_fiber_y   = np.interp(
        fiber_Y, smooth_fwd_y[sort_fwd], smooth_fwd_x[sort_fwd]
    )
    fwd_errors = np.abs(fiber_X - fwd_x_at_fiber_y)

    # Exclude last 100 points (matches your original)
    return float(np.mean(fwd_errors[:-100])), float(np.max(fwd_errors[:-100]))


def run_distribution(num_anchors, hearing_range,
                     num_trials=NUM_TRIALS, num_points=NUM_POINTS,
                     repetes=REPETES):
    """Runs `num_trials` trials and returns per-trial arrays."""
    anchors = generate_anchors.generate_linear_anchors(
        np.array([100, 1500]),
        3500 / num_anchors,
        num_anchors,
        90,
    )

    fwd_avg_dists, fwd_max_dists = [], []

    tag = f"anchors={num_anchors}  range={hearing_range}"
    print(f"\n{'─'*60}")
    print(f"  {tag}")
    print(f"{'─'*60}")

    for trial in range(num_trials):
        avg_d, max_d = run_single_trial(
            anchors, repetes, hearing_range, num_points=num_points
        )
        fwd_avg_dists.append(avg_d)
        fwd_max_dists.append(max_d)
        print(f"  Trial {trial + 1:>3}/{num_trials} | "
              f"avg={avg_d:.3f} m  max={max_d:.3f} m")

    fwd_avg_dists = np.array(fwd_avg_dists)
    fwd_max_dists = np.array(fwd_max_dists)

    print(f"\n  ► Avg error : mean={fwd_avg_dists.mean():.3f} m  "
          f"std={fwd_avg_dists.std():.3f} m  max={fwd_avg_dists.max():.3f} m")
    print(f"  ► Max error : mean={fwd_max_dists.mean():.3f} m  "
          f"std={fwd_max_dists.std():.3f} m  max={fwd_max_dists.max():.3f} m")

    return fwd_avg_dists, fwd_max_dists


# ──────────────────────────────────────────────
# PER-COMBINATION PLOTS
# ──────────────────────────────────────────────

def save_run_plots(fwd_avg_dists, fwd_max_dists,
                   num_anchors, hearing_range, out_dir):
    """Saves the four diagnostic plots for one (anchors, range) combo."""
    runs = np.arange(1, len(fwd_avg_dists) + 1)
    title_suffix = f"(anchors={num_anchors}, range={hearing_range} m)"

    # ── Plot 1 : histogram of average error ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fwd_avg_dists, bins=15, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(fwd_avg_dists.mean(), color="navy", linestyle="--",
               label=f"mean = {fwd_avg_dists.mean():.3f} m")
    ax.set_title(f"Avg Error Distribution  {title_suffix}")
    ax.set_xlabel("Average error (m)")
    ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_avg_error.png"), dpi=120)
    plt.close(fig)

    # ── Plot 2 : histogram of max error ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fwd_max_dists, bins=15, color="tomato",
            edgecolor="white", alpha=0.8)
    ax.axvline(fwd_max_dists.mean(), color="darkred", linestyle="--",
               label=f"mean = {fwd_max_dists.mean():.3f} m")
    ax.set_title(f"Max Error Distribution  {title_suffix}")
    ax.set_xlabel("Max error (m)")
    ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_max_error.png"), dpi=120)
    plt.close(fig)

    # ── Plot 3 : avg error per run ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(runs, fwd_avg_dists, marker="o", color="steelblue",
            linewidth=1.5, markersize=4)
    ax.set_title(f"Avg Error per Trial  {title_suffix}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Avg error (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "trace_avg_error.png"), dpi=120)
    plt.close(fig)

    # ── Plot 4 : max error per run ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(runs, fwd_max_dists, marker="o", color="tomato",
            linewidth=1.5, markersize=4)
    ax.set_title(f"Max Error per Trial  {title_suffix}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Max error (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "trace_max_error.png"), dpi=120)
    plt.close(fig)

    print(f"  ✓ Plots saved → {out_dir}/")


# ──────────────────────────────────────────────
# HEATMAPS
# ──────────────────────────────────────────────

def save_heatmaps(grid_avg, grid_max,
                  anchor_counts, hearing_ranges, out_root):
    """
    grid_avg / grid_max : 2-D arrays shaped
        (len(anchor_counts), len(hearing_ranges))
    rows = anchor counts, cols = hearing ranges
    """

    def _heatmap(data, title, filename, cmap, fmt):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       origin="lower",
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Error (m)", fontsize=11)

        # Axis ticks
        ax.set_xticks(range(len(hearing_ranges)))
        ax.set_xticklabels([str(r) for r in hearing_ranges])
        ax.set_yticks(range(len(anchor_counts)))
        ax.set_yticklabels([str(a) for a in anchor_counts])
        ax.set_xlabel("Hearing Range (m)", fontsize=12)
        ax.set_ylabel("Number of Anchors", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Annotate cells
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                text_color = "white" if val > (np.nanmax(data) * 0.6) else "black"
                ax.text(c, r, fmt.format(val),
                        ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(out_root, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Heatmap saved → {path}")

    _heatmap(grid_avg,
             "Mean Average Error Heatmap\n(anchors × hearing range)",
             "heatmap_avg_error.png",
             "YlOrRd", "{:.2f}")

    _heatmap(grid_max,
             "Mean Max Error Heatmap\n(anchors × hearing range)",
             "heatmap_max_error.png",
             "YlOrRd", "{:.2f}")


# ──────────────────────────────────────────────
# MAIN SWEEP
# ──────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Accumulators for heatmap matrices
    #   rows = anchor_counts index, cols = hearing_ranges index
    grid_avg = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_max = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)

    total = len(ANCHOR_COUNTS) * len(HEARING_RANGES)
    combo = 0

    for ai, num_anchors in enumerate(ANCHOR_COUNTS):
        for ri, hearing_range in enumerate(HEARING_RANGES):
            combo += 1
            print(f"\n{'═'*60}")
            print(f"  Combo {combo}/{total} │ "
                  f"anchors={num_anchors}  hearing_range={hearing_range} m")
            print(f"{'═'*60}")

            # Output folder for this combo
            out_dir = os.path.join(
                RESULTS_ROOT,
                f"{num_anchors:02d}anch_{hearing_range:03d}range",
            )
            os.makedirs(out_dir, exist_ok=True)

            # Run trials
            fwd_avg, fwd_max = run_distribution(
                num_anchors   = num_anchors,
                hearing_range = hearing_range,
                num_trials    = NUM_TRIALS,
                num_points    = NUM_POINTS,
                repetes       = REPETES,
            )

            # Save per-combo plots
            save_run_plots(fwd_avg, fwd_max, num_anchors, hearing_range, out_dir)

            # Record summary statistics for heatmaps
            grid_avg[ai, ri] = fwd_avg.mean()
            grid_max[ai, ri] = fwd_max.mean()

    # ── Final heatmaps ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Generating heatmaps …")
    print(f"{'═'*60}")
    save_heatmaps(grid_avg, grid_max,
                  ANCHOR_COUNTS, HEARING_RANGES, RESULTS_ROOT)

    # ── Print summary table ──────────────────────────────────────────────
    print("\n\n╔══ SWEEP COMPLETE ══════════════════════════════════════════╗")
    print(f"  Results folder : {os.path.abspath(RESULTS_ROOT)}/")
    print(f"  Trials per combo : {NUM_TRIALS}")
    print(f"  Combos run : {total}")
    print("\n  Mean Avg-Error grid (rows=anchors, cols=range):")
    header = "         " + "  ".join(f"{r:>6}" for r in HEARING_RANGES)
    print(header)
    for ai, na in enumerate(ANCHOR_COUNTS):
        row = f"  {na:>5}  " + "  ".join(f"{grid_avg[ai, ri]:6.3f}"
                                          for ri in range(len(HEARING_RANGES)))
        print(row)
    print("╚═══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()