# yaw_estimator.py
"""
Yaw estimation between two point clouds captured from the same place.

Given two point clouds A and B that are known (or believed) to be from the
same physical location, estimate the relative yaw angle of B with respect
to A.

Method
------
1. Compute an angular density histogram for each cloud: for each point,
   compute theta = atan2(y - cy, x - cx) relative to the centroid, then
   bin the angles into N bins over (-pi, pi].

2. Compute the circular cross-correlation between the two histograms via
   FFT. The location of the peak gives the relative yaw.

3. Confidence: peak-to-sidelobe ratio. A tall, sharp peak means a
   confident estimate; a flat, ambiguous correlation means low confidence.

Optionally, refine with a multi-start 2D ICP restricted to yaw rotation.

Notes
-----
- The two clouds must be from the SAME place (same room, same area). If
  they are from different places, the yaw estimate is meaningless.
- Assumes the vertical axis is Z and the yaw is rotation around Z.
- The estimator is invariant to translation and works on XY projection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================
# Result container
# ============================================================
@dataclass
class YawEstimate:
    """Result of a yaw estimation."""
    yaw_rad: float              # estimated relative yaw in radians, in (-pi, pi]
    yaw_deg: float              # same in degrees, for convenience
    confidence: float           # peak-to-sidelobe ratio (higher = better)
    ambiguity: float            # 2nd-peak / 1st-peak ratio (lower = better)
    method: str                 # "histogram" | "histogram+icp"

    def __repr__(self):
        return (f"YawEstimate(yaw={self.yaw_deg:+.2f}°, "
                f"conf={self.confidence:.2f}, "
                f"ambiguity={self.ambiguity:.2f}, "
                f"method={self.method})")


# ============================================================
# 1. Angular histogram
# ============================================================
def angular_histogram(pc, n_bins: int = 360,
                     use_centroid: bool = True) -> np.ndarray:
    """
    Compute a circular angular density histogram for a point cloud.

    Parameters
    ----------
    pc : PointCloud or (N,3) array
        The point cloud. If an object with `.points`, that's used; otherwise
        the array is used directly.
    n_bins : int
        Number of angular bins. 360 gives 1° resolution.
    use_centroid : bool
        If True, angles are computed relative to the XY centroid.
        If False, angles are computed relative to the sensor origin.

    Returns
    -------
    hist : np.ndarray, shape (n_bins,)
        Normalized angular histogram (sums to 1).
    """
    points = pc.points if hasattr(pc, "points") else np.asarray(pc)
    xy = points[:, :2]

    if use_centroid:
        anchor = xy.mean(axis=0)
    else:
        anchor = np.array([0.0, 0.0])

    rel = xy - anchor
    theta = np.arctan2(rel[:, 1], rel[:, 0])   # in (-pi, pi]

    hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    # Normalize to density so that total point count does not matter
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = hist.astype(np.float64)
    return hist


# ============================================================
# 2. Cross-correlation of angular histograms
# ============================================================
def _circular_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the circular cross-correlation between two 1D histograms via FFT.

    Returns an array `corr` of length len(a), where corr[k] is the dot
    product between `a` and `b` shifted by k bins. Positive k means `b`
    is shifted counterclockwise relative to `a`, matching the sign of
    the yaw returned by estimate_yaw().
    """
    n = len(a)
    if len(b) != n:
        raise ValueError("Histograms must have the same length.")
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    # Using conj(fa) * fb gives corr[k] = sum a[n] * b[n+k] in the FFT
    # indexing convention, which matches "shift of b" with the same sign
    # as the yaw angle.
    corr = np.fft.irfft(np.conj(fa) * fb, n=n)
    return corr


def estimate_yaw_from_histograms(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    refine_peak: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate the relative yaw between two angular histograms.

    Parameters
    ----------
    hist_a, hist_b : np.ndarray
        Angular histograms of the same length.
    refine_peak : bool
        If True, refine the discrete peak with parabolic interpolation on
        the neighboring bins, giving sub-bin resolution.

    Returns
    -------
    yaw_rad : float
        Estimated relative yaw in radians, in (-pi, pi].
    confidence : float
        Peak-to-sidelobe ratio (higher is better).
    ambiguity : float
        Second-peak / first-peak ratio (lower is better).
    """
    n = len(hist_a)
    corr = _circular_cross_correlation(hist_a, hist_b)

    # --- Find the peak ---
    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    # --- Sub-bin refinement via parabolic fit on neighbors ---
    if refine_peak and n >= 3:
        left  = corr[(peak_idx - 1) % n]
        right = corr[(peak_idx + 1) % n]
        denom = 2.0 * peak_val - left - right
        if abs(denom) > 1e-12:
            delta = 0.5 * (right - left) / denom
            # Clip to avoid runaway
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        peak_pos = (peak_idx + delta) % n
    else:
        peak_pos = float(peak_idx)

    # --- Convert bin position to angle ---
    # corr[k] corresponds to shifting b by k bins
    # Each bin is 2pi / n radians
    yaw_rad = 2.0 * np.pi * peak_pos / n
    # Wrap to (-pi, pi]
    yaw_rad = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

    # --- Confidence via peak-to-sidelobe ratio ---
    # Zero out the peak (and a small neighborhood) to define the "sidelobes"
    n_exclude = max(2, n // 20)   # exclude 5% of bins around the peak
    mask = np.ones(n, dtype=bool)
    for d in range(-n_exclude, n_exclude + 1):
        mask[(peak_idx + d) % n] = False
    sidelobes = corr[mask]
    sidelobe_mean = float(sidelobes.mean())
    sidelobe_std  = float(sidelobes.std())
    confidence = (peak_val - sidelobe_mean) / max(sidelobe_std, 1e-12)

    # --- Ambiguity: second highest peak outside the primary neighborhood ---
    corr_copy = corr.copy()
    for d in range(-n_exclude, n_exclude + 1):
        corr_copy[(peak_idx + d) % n] = 0.0
    second_peak_val = float(corr_copy.max())
    ambiguity = second_peak_val / max(peak_val, 1e-12)

    return yaw_rad, confidence, ambiguity


# ============================================================
# 3. Optional ICP refinement (multi-start 2D)
# ============================================================
def _rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _icp_2d_yaw(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    initial_yaw: float,
    max_iter: int = 30,
    tol: float = 1e-5,
    max_correspondence_dist: float = 1.0,
) -> Tuple[float, float]:
    """
    Run a simple point-to-point ICP restricted to yaw rotation (Z).

    Parameters
    ----------
    src_xy : (M, 2)
        Source points, XY only, centered.
    dst_xy : (N, 2)
        Target points, XY only, centered.
    initial_yaw : float
        Initial rotation applied to src before matching.
    max_iter : int
    tol : float
        Convergence tolerance on the yaw update.
    max_correspondence_dist : float
        Reject correspondences beyond this distance.

    Returns
    -------
    yaw : float
        Refined yaw.
    rmse : float
        Final RMSE of the inlier correspondences.
    """
    from scipy.spatial import cKDTree

    yaw = initial_yaw
    tree = cKDTree(dst_xy)
    prev_rmse = np.inf

    for _ in range(max_iter):
        # Rotate source by current yaw
        R = _rotation_matrix_2d(yaw)
        src_rot = src_xy @ R.T

        # Find nearest neighbor in target
        dists, idxs = tree.query(src_rot, k=1)
        mask = dists < max_correspondence_dist
        if mask.sum() < 10:
            # Too few correspondences; break
            break

        src_matched = src_rot[mask]
        dst_matched = dst_xy[idxs[mask]]

        # Compute optimal yaw delta from Procrustes
        # We want the rotation that aligns src_matched to dst_matched
        # Optimal yaw update via 2D Procrustes
        H = src_matched.T @ dst_matched
        # Correct sign: rotate src by +delta to align it with dst
        delta_yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        yaw = yaw + delta_yaw

        # RMSE
        R = _rotation_matrix_2d(yaw)
        src_rot = src_xy @ R.T
        dists, _ = tree.query(src_rot, k=1)
        rmse = float(np.sqrt(np.mean(dists[dists < max_correspondence_dist] ** 2)))

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return yaw, prev_rmse


def refine_with_icp(
    pc_a,
    pc_b,
    initial_yaw: float,
    n_starts: int = 6,
    span_deg: float = 30.0,
    max_correspondence_dist: float = 1.0,
) -> Tuple[float, float]:
    """
    Refine the yaw estimate with multi-start 2D ICP.

    Parameters
    ----------
    pc_a, pc_b : PointCloud or (N,3) arrays
    initial_yaw : float
        Initial estimate from the histogram method.
    n_starts : int
        Number of ICP starts, evenly spaced within +-span_deg of initial_yaw.
    span_deg : float
        Angular span around the initial yaw for the multi-start.
    max_correspondence_dist : float
        ICP correspondence threshold in meters.

    Returns
    -------
    yaw : float
        Refined yaw.
    rmse : float
        Final RMSE (lower = better).
    """
    pts_a = pc_a.points if hasattr(pc_a, "points") else np.asarray(pc_a)
    pts_b = pc_b.points if hasattr(pc_b, "points") else np.asarray(pc_b)

    # Center both clouds on their respective centroids for ICP
    a_xy = pts_a[:, :2] - pts_a[:, :2].mean(axis=0)
    b_xy = pts_b[:, :2] - pts_b[:, :2].mean(axis=0)

    # Multi-start
    span_rad = np.deg2rad(span_deg)
    starts = np.linspace(initial_yaw - span_rad,
                         initial_yaw + span_rad,
                         n_starts)

    best_yaw, best_rmse = initial_yaw, np.inf
    for s in starts:
        yaw, rmse = _icp_2d_yaw(
            a_xy, b_xy, s,
            max_correspondence_dist=max_correspondence_dist,
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_yaw = yaw

    # Normalize to (-pi, pi]
    best_yaw = ((best_yaw + np.pi) % (2 * np.pi)) - np.pi
    return best_yaw, best_rmse


# ============================================================
# 4. High-level API
# ============================================================
def estimate_yaw(
    pc_a,
    pc_b,
    n_bins: int = 360,
    use_centroid: bool = True,
    refine_icp: bool = False,
    icp_n_starts: int = 6,
    icp_span_deg: float = 30.0,
    icp_max_dist: float = 1.0,
) -> YawEstimate:
    """
    Estimate the relative yaw of pc_b with respect to pc_a.

    The two point clouds must be from the same place (same room or area).
    The estimator is invariant to translation and does not depend on the
    placement of the origin.

    Parameters
    ----------
    pc_a, pc_b : PointCloud or (N,3) arrays
    n_bins : int
        Number of angular bins (360 -> 1° resolution).
    use_centroid : bool
        If True, angles are computed relative to the XY centroid.
        If False, relative to the sensor origin.
    refine_icp : bool
        If True, refine the histogram estimate with multi-start 2D ICP.
    icp_n_starts, icp_span_deg, icp_max_dist : parameters for ICP refinement.

    Returns
    -------
    YawEstimate
    """
    hist_a = angular_histogram(pc_a, n_bins=n_bins, use_centroid=use_centroid)
    hist_b = angular_histogram(pc_b, n_bins=n_bins, use_centroid=use_centroid)

    yaw_rad, confidence, ambiguity = estimate_yaw_from_histograms(hist_a, hist_b)

    method = "histogram"
    if refine_icp:
        yaw_rad, rmse = refine_with_icp(
            pc_a, pc_b, initial_yaw=yaw_rad,
            n_starts=icp_n_starts, span_deg=icp_span_deg,
            max_correspondence_dist=icp_max_dist,
        )
        method = "histogram+icp"

    return YawEstimate(
        yaw_rad=yaw_rad,
        yaw_deg=float(np.rad2deg(yaw_rad)),
        confidence=confidence,
        ambiguity=ambiguity,
        method=method,
    )


# ============================================================
# 5. Utility for visual debugging
# ============================================================
def plot_angular_histograms(
    pc_a,
    pc_b,
    n_bins: int = 360,
    use_centroid: bool = True,
    ax=None,
):
    """Plot the two angular histograms overlaid for visual inspection."""
    import matplotlib.pyplot as plt

    ha = angular_histogram(pc_a, n_bins=n_bins, use_centroid=use_centroid)
    hb = angular_histogram(pc_b, n_bins=n_bins, use_centroid=use_centroid)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    theta = np.linspace(-180, 180, n_bins)
    ax.plot(theta, ha, label="A", alpha=0.7)
    ax.plot(theta, hb, label="B", alpha=0.7)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("density")
    ax.set_title("Angular histograms")
    ax.legend()
    return ax