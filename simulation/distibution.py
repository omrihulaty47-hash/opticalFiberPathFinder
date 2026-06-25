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
  3. After the full sweep, generates ten heatmaps:
       • results/heatmap_avg_error.png                 (mean of avg error)
       • results/heatmap_max_error.png                 (mean of max error)
       • results/heatmap_avg_error_core.png             (same, core window)
       • results/heatmap_max_error_core.png             (same, core window)
       • results/heatmap_max_of_max_error.png          (worst single trial's max error per combo)
       • results/heatmap_max_of_max_error_core.png      (same, core window)
       • results/heatmap_median_avg_error.png          (median of avg error per combo)
       • results/heatmap_median_avg_error_core.png      (same, core window)
       • results/heatmap_median_max_error.png          (median of max error per combo)
       • results/heatmap_median_max_error_core.png      (same, core window)

     The "Mean Average Error" and "Mean Max Error" heatmaps (full and core,
     four heatmaps total) additionally annotate each cell with the standard
     deviation across trials, shown as "mean ± std", so the spread is
     visible alongside the average.

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
ANCHOR_COUNTS   = [3,5,10,50,100,500,1000]    
HEARING_RANGES  = [50,100,500,1000,3000] 
NUM_TRIALS      = 20
NUM_POINTS      = 3000
REPETES         = 1
RESULTS_ROOT    = "ranged_results_moved"

# Trim used for the "core" error metric: error is computed only on the
# slice of points starting at the 50th point from the start and ending
# at the 50th point from the end (i.e. fwd_errors[CORE_TRIM:-CORE_TRIM]).
CORE_TRIM       = 50



def run_single_trial(anchors, repetes, hearing_range, num_points=3000):
    """
    One Monte-Carlo trial (forward pass only).
    `hearing_range` is passed to generate_noisy_distances so only anchors
    within that radius contribute – wire this into point_est as needed.

    Returns
    -------
    (mean_error_full, max_error_full, mean_error_core, max_error_core)
        * "full"  : computed on fwd_errors[:-100]   (matches your original)
        * "core"  : computed on fwd_errors[CORE_TRIM:-CORE_TRIM], i.e. from
                    the 50th point from the start to the 50th point from
                    the end.
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
            est_forward, 20, 6, 50, 0
        )

    # --- Y-ALIGNED ERROR ---
    sort_fwd           = np.argsort(smooth_fwd_y)
    fwd_x_at_fiber_y   = np.interp(
        fiber_Y, smooth_fwd_y[sort_fwd], smooth_fwd_x[sort_fwd]
    )
    fwd_errors = np.abs(fiber_X - fwd_x_at_fiber_y)

    # "Full" window: exclude last 100 points (matches your original)
    mean_full = float(np.mean(fwd_errors[:-100]))
    max_full  = float(np.max(fwd_errors[:-100]))

    # "Core" window: from the 50th point from the start to the 50th
    # point from the end.
    core_errors = fwd_errors[CORE_TRIM:-CORE_TRIM]
    mean_core = float(np.mean(core_errors))
    max_core  = float(np.max(core_errors))

    return mean_full, max_full, mean_core, max_core


def run_distribution(num_anchors, hearing_range,
                     num_trials=NUM_TRIALS, num_points=NUM_POINTS,
                     repetes=REPETES):
    """Runs `num_trials` trials and returns per-trial arrays."""
    anchors = generate_anchors.generate_linear_anchors(
        np.array([100, 1500]),
        3000 / num_anchors,
        num_anchors,
        90
    )

    fwd_avg_dists, fwd_max_dists = [], []
    fwd_avg_core_dists, fwd_max_core_dists = [], []

    tag = f"anchors={num_anchors}  range={hearing_range}"
    print(f"\n{'─'*60}")
    print(f"  {tag}")
    print(f"{'─'*60}")

    for trial in range(num_trials):
        avg_d, max_d, avg_core_d, max_core_d = run_single_trial(
            anchors, repetes, hearing_range, num_points=num_points
        )
        fwd_avg_dists.append(avg_d)
        fwd_max_dists.append(max_d)
        fwd_avg_core_dists.append(avg_core_d)
        fwd_max_core_dists.append(max_core_d)
        print(f"  Trial {trial + 1:>3}/{num_trials} | "
              f"avg={avg_d:.3f} m  max={max_d:.3f} m  |  "
              f"core_avg={avg_core_d:.3f} m  core_max={max_core_d:.3f} m")

    fwd_avg_dists = np.array(fwd_avg_dists)
    fwd_max_dists = np.array(fwd_max_dists)
    fwd_avg_core_dists = np.array(fwd_avg_core_dists)
    fwd_max_core_dists = np.array(fwd_max_core_dists)

    print(f"\n  ► Avg error : mean={fwd_avg_dists.mean():.3f} m  "
          f"std={fwd_avg_dists.std():.3f} m  max={fwd_avg_dists.max():.3f} m")
    print(f"  ► Max error : mean={fwd_max_dists.mean():.3f} m  "
          f"std={fwd_max_dists.std():.3f} m  max={fwd_max_dists.max():.3f} m")
    print(f"  ► Avg error (core, pts {CORE_TRIM}..-{CORE_TRIM}) : "
          f"mean={fwd_avg_core_dists.mean():.3f} m  "
          f"std={fwd_avg_core_dists.std():.3f} m  "
          f"max={fwd_avg_core_dists.max():.3f} m")
    print(f"  ► Max error (core, pts {CORE_TRIM}..-{CORE_TRIM}) : "
          f"mean={fwd_max_core_dists.mean():.3f} m  "
          f"std={fwd_max_core_dists.std():.3f} m  "
          f"max={fwd_max_core_dists.max():.3f} m")

    return fwd_avg_dists, fwd_max_dists, fwd_avg_core_dists, fwd_max_core_dists


# ──────────────────────────────────────────────
# PER-COMBINATION PLOTS
# ──────────────────────────────────────────────

def save_run_plots(fwd_avg_dists, fwd_max_dists,
                   fwd_avg_core_dists, fwd_max_core_dists,
                   num_anchors, hearing_range, out_dir):
    """Saves the diagnostic plots for one (anchors, range) combo:
    four for the "full" error metric, four more for the "core" error
    metric (points CORE_TRIM..-CORE_TRIM)."""
    runs = np.arange(1, len(fwd_avg_dists) + 1)
    title_suffix = f"(anchors={num_anchors}, range={hearing_range} m)"
    core_suffix = (f"(anchors={num_anchors}, range={hearing_range} m, "
                   f"pts {CORE_TRIM}..-{CORE_TRIM})")

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

    # ── Plot 5 : histogram of CORE average error ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fwd_avg_core_dists, bins=15, color="seagreen",
            edgecolor="white", alpha=0.8)
    ax.axvline(fwd_avg_core_dists.mean(), color="darkgreen", linestyle="--",
               label=f"mean = {fwd_avg_core_dists.mean():.3f} m")
    ax.set_title(f"Core Avg Error Distribution  {core_suffix}")
    ax.set_xlabel("Average error (m)")
    ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_avg_error_core.png"), dpi=120)
    plt.close(fig)

    # ── Plot 6 : histogram of CORE max error ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fwd_max_core_dists, bins=15, color="darkorange",
            edgecolor="white", alpha=0.8)
    ax.axvline(fwd_max_core_dists.mean(), color="chocolate", linestyle="--",
               label=f"mean = {fwd_max_core_dists.mean():.3f} m")
    ax.set_title(f"Core Max Error Distribution  {core_suffix}")
    ax.set_xlabel("Max error (m)")
    ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_max_error_core.png"), dpi=120)
    plt.close(fig)

    # ── Plot 7 : CORE avg error per run ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(runs, fwd_avg_core_dists, marker="o", color="seagreen",
            linewidth=1.5, markersize=4)
    ax.set_title(f"Core Avg Error per Trial  {core_suffix}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Avg error (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "trace_avg_error_core.png"), dpi=120)
    plt.close(fig)

    # ── Plot 8 : CORE max error per run ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(runs, fwd_max_core_dists, marker="o", color="darkorange",
            linewidth=1.5, markersize=4)
    ax.set_title(f"Core Max Error per Trial  {core_suffix}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Max error (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "trace_max_error_core.png"), dpi=120)
    plt.close(fig)

    print(f"  ✓ Plots saved → {out_dir}/")


# ──────────────────────────────────────────────
# HEATMAPS
# ──────────────────────────────────────────────

def save_heatmaps(grid_avg, grid_max,
                  grid_avg_core, grid_max_core,
                  grid_max_of_max, grid_max_of_max_core,
                  grid_median_avg, grid_median_max,
                  grid_median_avg_core, grid_median_max_core,
                  grid_std_avg, grid_std_avg_core,
                  grid_std_max, grid_std_max_core,
                  anchor_counts, hearing_ranges, out_root):
    """
    grid_avg / grid_max / grid_avg_core / grid_max_core : 2-D arrays shaped
        (len(anchor_counts), len(hearing_ranges))
    rows = anchor counts, cols = hearing ranges
    The "_core" grids hold the mean error computed only on points
    CORE_TRIM..-CORE_TRIM (i.e. 50th point from the start to the 50th
    point from the end).

    grid_max_of_max / grid_max_of_max_core : same shape, but instead of the
    *mean* of the per-trial max errors, each cell holds the single worst
    max-error observed across all trials for that combination (i.e. the
    maximum of all per-trial maximums) — the absolute worst case seen for
    that (anchors, hearing_range) combo.

    grid_median_avg / grid_median_max / grid_median_avg_core /
    grid_median_max_core : same shape again, but each cell holds the
    *median* (instead of the mean) of the per-trial avg/max errors for that
    combination — a robustness check against outlier trials skewing the
    mean-based heatmaps above.

    grid_std_avg / grid_std_avg_core : same shape, holding the standard
    deviation (across trials) of the avg error for that combo. Overlaid as
    "mean ± std" annotations on the Mean Average Error heatmaps so the
    spread of the underlying trials is visible alongside the mean.

    grid_std_max / grid_std_max_core : same idea, but for the max error —
    overlaid as "mean ± std" annotations on the Mean Max Error heatmaps.
    """

    def _heatmap(data, title, filename, cmap, fmt, std_data=None):
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

        # Annotate cells — "mean ± std" when a std grid is supplied,
        # otherwise just the value.
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                text_color = "white" if val > (np.nanmax(data) * 0.6) else "black"
                if std_data is not None:
                    label = f"{fmt.format(val)}\n±{fmt.format(std_data[r, c])}"
                else:
                    label = fmt.format(val)
                ax.text(c, r, label,
                        ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(out_root, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Heatmap saved → {path}")

    _heatmap(grid_avg,
             "Mean Average Error Heatmap (mean ± std across trials)\n"
             "(anchors × hearing range)",
             "heatmap_avg_error.png",
             "YlOrRd", "{:.2f}", std_data=grid_std_avg)

    _heatmap(grid_max,
             "Mean Max Error Heatmap (mean ± std across trials)\n"
             "(anchors × hearing range)",
             "heatmap_max_error.png",
             "YlOrRd", "{:.2f}", std_data=grid_std_max)

    _heatmap(grid_avg_core,
             f"Mean Average Error Heatmap (mean ± std across trials, Core, "
             f"pts {CORE_TRIM}..-{CORE_TRIM})\n"
             "(anchors × hearing range)",
             "heatmap_avg_error_core.png",
             "YlGnBu", "{:.2f}", std_data=grid_std_avg_core)

    _heatmap(grid_max_core,
             f"Mean Max Error Heatmap (mean ± std across trials, Core, "
             f"pts {CORE_TRIM}..-{CORE_TRIM})\n"
             "(anchors × hearing range)",
             "heatmap_max_error_core.png",
             "YlGnBu", "{:.2f}", std_data=grid_std_max_core)

    _heatmap(grid_max_of_max,
             "Worst-Case Max Error Heatmap\n"
             "(max of all per-trial max errors, per combo)",
             "heatmap_max_of_max_error.png",
             "PuRd", "{:.2f}")

    _heatmap(grid_max_of_max_core,
             f"Worst-Case Max Error Heatmap (Core, pts {CORE_TRIM}..-{CORE_TRIM})\n"
             "(max of all per-trial max errors, per combo)",
             "heatmap_max_of_max_error_core.png",
             "PuRd", "{:.2f}")

    _heatmap(grid_median_avg,
             "Median Average Error Heatmap\n(anchors × hearing range)",
             "heatmap_median_avg_error.png",
             "YlOrRd", "{:.2f}")

    _heatmap(grid_median_max,
             "Median Max Error Heatmap\n(anchors × hearing range)",
             "heatmap_median_max_error.png",
             "YlOrRd", "{:.2f}")

    _heatmap(grid_median_avg_core,
             f"Median Average Error Heatmap (Core, pts {CORE_TRIM}..-{CORE_TRIM})\n"
             "(anchors × hearing range)",
             "heatmap_median_avg_error_core.png",
             "YlGnBu", "{:.2f}")

    _heatmap(grid_median_max_core,
             f"Median Max Error Heatmap (Core, pts {CORE_TRIM}..-{CORE_TRIM})\n"
             "(anchors × hearing range)",
             "heatmap_median_max_error_core.png",
             "YlGnBu", "{:.2f}")


# ──────────────────────────────────────────────
# MAIN SWEEP
# ──────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Accumulators for heatmap matrices
    #   rows = anchor_counts index, cols = hearing_ranges index
    grid_avg = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_max = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_avg_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_max_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    # Worst-case (max of all per-trial maximums) for each combo
    grid_max_of_max = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_max_of_max_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    # Median (instead of mean) of avg/max error per combo
    grid_median_avg = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_median_max = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_median_avg_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_median_max_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    # Standard deviation (across trials) of the avg error per combo
    grid_std_avg = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_std_avg_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_std_max = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)
    grid_std_max_core = np.full((len(ANCHOR_COUNTS), len(HEARING_RANGES)), np.nan)

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
            fwd_avg, fwd_max, fwd_avg_core, fwd_max_core = run_distribution(
                num_anchors   = num_anchors,
                hearing_range = hearing_range,
                num_trials    = NUM_TRIALS,
                num_points    = NUM_POINTS,
                repetes       = REPETES,
            )

            # Save per-combo plots
            save_run_plots(fwd_avg, fwd_max, fwd_avg_core, fwd_max_core,
                            num_anchors, hearing_range, out_dir)

            # Record summary statistics for heatmaps
            grid_avg[ai, ri] = fwd_avg.mean()
            grid_max[ai, ri] = fwd_max.mean()
            grid_avg_core[ai, ri] = fwd_avg_core.mean()
            grid_max_core[ai, ri] = fwd_max_core.mean()
            # Worst single trial observed for this combo
            grid_max_of_max[ai, ri] = fwd_max.max()
            grid_max_of_max_core[ai, ri] = fwd_max_core.max()
            # Median (robust to outlier trials) of avg/max error
            grid_median_avg[ai, ri] = np.median(fwd_avg)
            grid_median_max[ai, ri] = np.median(fwd_max)
            grid_median_avg_core[ai, ri] = np.median(fwd_avg_core)
            grid_median_max_core[ai, ri] = np.median(fwd_max_core)
            # Spread (std dev) of the avg error across trials
            grid_std_avg[ai, ri] = fwd_avg.std()
            grid_std_avg_core[ai, ri] = fwd_avg_core.std()
            # Spread (std dev) of the max error across trials
            grid_std_max[ai, ri] = fwd_max.std()
            grid_std_max_core[ai, ri] = fwd_max_core.std()

    # ── Final heatmaps ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Generating heatmaps …")
    print(f"{'═'*60}")
    save_heatmaps(grid_avg, grid_max,
                  grid_avg_core, grid_max_core,
                  grid_max_of_max, grid_max_of_max_core,
                  grid_median_avg, grid_median_max,
                  grid_median_avg_core, grid_median_max_core,
                  grid_std_avg, grid_std_avg_core,
                  grid_std_max, grid_std_max_core,
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
    print(f"\n  Mean Avg-Error grid, CORE pts {CORE_TRIM}..-{CORE_TRIM} "
          "(rows=anchors, cols=range):")
    print(header)
    for ai, na in enumerate(ANCHOR_COUNTS):
        row = f"  {na:>5}  " + "  ".join(f"{grid_avg_core[ai, ri]:6.3f}"
                                          for ri in range(len(HEARING_RANGES)))
        print(row)
    print("╚═══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()