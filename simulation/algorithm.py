"""
localization_algorithm.py
==========================
The actual forward-only sequential estimation + smoothing algorithm,
factored out of main.py / distibution.py so it exists in exactly one place.

Design rule
-----------
`estimate_path()` is the algorithm. It is NOT allowed to know anything
about how the distance measurements were produced - no `real_anchors`,
no ground-truth fiber path, no noise model. It only ever sees:
    - `known_anchors`   : anchor positions as the algorithm believes them
    - `distances`       : pre-generated range measurements

Anything that touches the simulated "real world" (the true fiber path,
the perturbed `real_anchors`, `point_est.generate_noisy_distances`) is a
*simulation* concern, not an *algorithm* concern, and is kept out of this
function. The caller (main.py, distibution.py, or a unit test) generates
distances first, then hands them to `estimate_path`.

`run_full_pipeline()` is the one exception: it wires the simulation and
the algorithm together end-to-end in a single call. It exists purely
because it's convenient for quick tests / sweeps - it is NOT meant to be
the place that contains the algorithm's logic.
"""

import numpy as np

import point_est
import point_smoother


def generate_distances(fiber_X, fiber_Y, real_anchors, hearing_range, num_points, r=1):
    """
    Simulation-side helper: produces the noisy range measurements a caller
    would feed into `estimate_path`. This is the only place allowed to
    touch `real_anchors` / the true fiber path - the algorithm itself
    never sees either.

    Returns
    -------
    distances : list, length num_points
        distances[i] is a list of `r` measurement sets (one per repeated
        sample) for point i. distances[0] is left as None since the first
        node is seeded directly from start_point and never measured.
    """
    distances = [None] * num_points
    for i in range(1, num_points):
        distances[i] = [
            point_est.generate_noisy_distances(
                [fiber_X[i], fiber_Y[i]], real_anchors, hearing_range=hearing_range,
            )
            for _ in range(r)
        ]
    return distances


def estimate_path(known_anchors, distances, num_points, start_point,
                   end_point=None, r=1,
                   smoothing_window=20, smoothing_poly_order=6,
                   smoothing_overlap=50, smoothing_extra=0,
                   linearize_edges=True, edge_linearize_width=50):
    """
    The algorithm. Forward-only sequential point estimation, averaged
    over `r` repeated measurements per point, followed by overlapping
    segment smoothing.

    Parameters
    ----------
    known_anchors : array-like
        Anchors as known to the algorithm (NOT the perturbed real ones).
    distances : sequence, length num_points
        distances[i] is a sequence of `r` measurement sets for point i,
        as produced by `generate_distances`. distances[0] is ignored.
    num_points : int
    start_point : array-like
        Seeds est[0] directly.
    end_point : array-like or None
        If given, pins est[-1] to this value before smoothing (matches
        main.py's behaviour). If None, the last raw estimate is left as
        whatever the forward pass produced (matches distibution.py's
        original behaviour, which never pinned the last point).
    r : int
        Number of repeated measurements per point to average.
    smoothing_* / linearize_edges / edge_linearize_width :
        Passed straight through to
        point_smoother.smooth_path_by_segments_with_overlap.

    Returns
    -------
    est : (num_points, 2) ndarray
        Raw forward-pass estimates, pre-smoothing.
    est_x, est_y : ndarray
        Smoothed path.
    """
    est = np.zeros((num_points, 2))
    est[0] = start_point

    for i in range(1, num_points):
        for rep in range(r):
            est[i] += point_est.estimate_single_point(
                known_anchors, distances[i][rep], i, est[i - 1]
            )
        est[i] /= r

    if end_point is not None:
        est[-1] = end_point

    est_x, est_y = point_smoother.smooth_path_by_segments_with_overlap(
        est, smoothing_window, smoothing_poly_order, smoothing_overlap,
        smoothing_extra, edge_linearize_width if linearize_edges else 0,
    )

    return est, est_x, est_y


def compute_position_error(fiber_X, fiber_Y, est_x, est_y):
    """
    Cross-sectional error metric shared by both scripts: resample the
    estimated X path onto the true ground-truth Y positions and return
    the absolute horizontal offset at each cross-section.
    """
    sort_idx = np.argsort(est_y)
    est_y_sorted = est_y[sort_idx]
    est_x_sorted = est_x[sort_idx]
    est_x_at_fiber_y = np.interp(fiber_Y, est_y_sorted, est_x_sorted)
    return np.abs(fiber_X - est_x_at_fiber_y)


def run_full_pipeline(known_anchors, real_anchors, fiber_X, fiber_Y,
                       start_point, end_point, hearing_range,
                       num_points=None, r=1,
                       smoothing_window=20, smoothing_poly_order=6,
                       smoothing_overlap=50, smoothing_extra=0,
                       linearize_edges=True, edge_linearize_width=50):
    """
    Convenience/testing-only wrapper: runs the simulation (distance
    generation) and the algorithm (estimate_path) together in one call,
    plus the error metric. Kept around because it's much easier to write
    a quick test or a Monte-Carlo sweep against one function call than
    to wire generate_distances -> estimate_path -> compute_position_error
    by hand every time.

    This is intentionally the ONLY function in this module that is
    allowed to take `real_anchors` as an argument.

    Returns
    -------
    est, est_x, est_y, errors
    """
    if num_points is None:
        num_points = len(fiber_X)

    distances = generate_distances(
        fiber_X, fiber_Y, real_anchors, hearing_range, num_points, r=r,
    )

    est, est_x, est_y = estimate_path(
        known_anchors, distances, num_points, start_point,
        end_point=end_point, r=r,
        smoothing_window=smoothing_window, smoothing_poly_order=smoothing_poly_order,
        smoothing_overlap=smoothing_overlap, smoothing_extra=smoothing_extra,
        linearize_edges=linearize_edges, edge_linearize_width=edge_linearize_width,
    )

    errors = compute_position_error(fiber_X, fiber_Y, est_x, est_y)

    return est, est_x, est_y, errors