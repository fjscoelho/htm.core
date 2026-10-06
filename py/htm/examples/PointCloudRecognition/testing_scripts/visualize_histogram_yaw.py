# visualize_histogram_yaw.py
"""
Visualize the SIMPLE angular-histogram yaw estimator for revisit pairs.

The estimator works like this:
    1. For each cloud, compute theta = atan2(y - cy, x - cx) for every point
       (relative to the XY centroid, or the sensor origin).
    2. Bin the angles into N bins over (-pi, pi] -> an angular histogram.
    3. Circular cross-correlation (via FFT) between the two histograms.
       The position of the peak is the relative yaw.

This script produces, per pair, a 2x2 figure:
    (1) Polar angular histograms of A (stored) and B (query).
    (2) Linear overlay of the two histograms.
    (3) Circular cross-correlation with the estimated peak and GT line.
    (4) B's histogram re-aligned by the estimated yaw, overlaid on A.

Usage
-----
    python visualize_histogram_yaw.py                    # default 4 revisits
    python visualize_histogram_yaw.py --pairs 1:104 8:90
    python visualize_histogram_yaw.py --n-bins 360 --use-origin
    python visualize_histogram_yaw.py --save-plots hist_out
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator import (
    angular_histogram,
    _circular_cross_correlation,
    estimate_yaw_from_histograms,
    resolve_yaw_ambiguity_with_odometry,
)
from test_pipeline_revisits import (
    load_ground_truth_poses,
    load_odometry_poses,
    ground_truth_relative_yaw_deg,
    wrap_angle_deg,
)


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
TRAJ_JSON = DATA_DIR / "trajectory_map_ros2.json"

# (stored_wp, query_wp) — the revisits found by the pipeline
DEFAULT_PAIRS: List[Tuple[int, int]] = [(8, 90), (8, 94), (1, 97), (1, 104)]


def find_npy(wp: int) -> Path:
    matches = sorted(DATA_DIR.glob(f"wp_{wp:04d}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"wp_{wp:04d} not found in {DATA_DIR}")
    return matches[0]


def _bin_angle_axes(n_bins: int):
    """Return (theta_bin_centers, corr_shift_deg_sorted, sort_order)."""
    # Histogram bin centers (binning starts at -pi)
    theta = -np.pi + (np.arange(n_bins) + 0.5) * (2 * np.pi / n_bins)

    # Correlation index k -> angle 2*pi*k/n_bins, wrapped to (-pi, pi]
    shift = 2 * np.pi * np.arange(n_bins) / n_bins
    shift = ((shift + np.pi) % (2 * np.pi)) - np.pi
    shift_deg = np.rad2deg(shift)
    order = np.argsort(shift_deg)
    return theta, shift_deg, order


def visualize_pair(
    pc_a,
    pc_b,
    wp_a: int,
    wp_b: int,
    n_bins: int = 360,
    use_centroid: bool = True,
    gt_deg: Optional[float] = None,
    yaw_rel_odom_deg: Optional[float] = None,
    save_dir: Optional[Path] = None,
):
    """Create the 2x2 diagnostic figure for one pair.

    Returns (est_deg, est_odom_deg, conf, amb); est_odom_deg is None when
    `yaw_rel_odom_deg` is not provided.
    """
    ha = angular_histogram(pc_a, n_bins=n_bins, use_centroid=use_centroid)
    hb = angular_histogram(pc_b, n_bins=n_bins, use_centroid=use_centroid)

    yaw_rad, conf, amb = estimate_yaw_from_histograms(ha, hb, refine_peak=True)
    yaw_deg = float(np.rad2deg(yaw_rad))

    # --- Optional odometry disambiguation (180-degree ambiguity) ---
    yaw_odom_deg: Optional[float] = None
    flipped = False
    if yaw_rel_odom_deg is not None:
        yaw_odom_rad, _ = resolve_yaw_ambiguity_with_odometry(
            yaw_rad, 0.0, float(np.deg2rad(yaw_rel_odom_deg)))
        yaw_odom_deg = float(np.rad2deg(yaw_odom_rad))
        flipped = abs(wrap_angle_deg(yaw_odom_deg - yaw_deg)) > 1e-6

    corr = _circular_cross_correlation(ha, hb)
    peak_idx = int(np.argmax(corr))

    theta, shift_deg, order = _bin_angle_axes(n_bins)
    theta_deg = np.rad2deg(theta)

    # Align B onto A by shifting its histogram back by the discrete peak.
    # (verified convention: corr peaks at k = yaw_bins => roll by -k)
    shift_bins = peak_idx + (n_bins // 2 if flipped else 0)
    hb_aligned = np.roll(hb, -shift_bins)
    yaw_shown = yaw_odom_deg if yaw_odom_deg is not None else yaw_deg

    anchor = "centroid" if use_centroid else "sensor origin"
    err = wrap_angle_deg(yaw_deg - gt_deg) if gt_deg is not None else None
    suptitle = (f"Simple histogram yaw — A=wp{wp_a} (stored), B=wp{wp_b} (query)"
                f"  |  est={yaw_deg:+.2f}°")
    if yaw_odom_deg is not None:
        flag = " (flipped 180°)" if flipped else ""
        suptitle += f"  est+odom={yaw_odom_deg:+.2f}°{flag}"
    if gt_deg is not None:
        suptitle += f"  gt={gt_deg:+.2f}°"
        if yaw_odom_deg is not None:
            suptitle += (f"  err+odom="
                         f"{wrap_angle_deg(yaw_odom_deg - gt_deg):+.2f}°")
        else:
            suptitle += f"  err={err:+.2f}°"
    suptitle += (f"  |  conf={conf:.1f}  amb={amb:.2f}  |  {anchor}"
                 f"  |  {n_bins} bins")

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(suptitle, fontsize=12)

    # ---- (1) polar histograms ----
    ax = fig.add_subplot(2, 2, 1, projection="polar")
    ax.plot(theta, ha, color="tab:blue", lw=1.2, label="A (stored)")
    ax.plot(theta, hb, color="tab:orange", lw=1.2, alpha=0.8,
            label="B (query)")
    ax.set_title("Polar angular histograms")
    ax.legend(loc="upper right", fontsize=8)

    # ---- (2) linear overlay ----
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(theta_deg, ha, color="tab:blue", lw=1.2, label="A (stored)")
    ax.plot(theta_deg, hb, color="tab:orange", lw=1.2, alpha=0.8,
            label="B (query)")
    ax.set_xlabel("angle (deg)"); ax.set_ylabel("density")
    ax.set_title("Angular histograms (overlay)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---- (3) circular cross-correlation ----
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(shift_deg[order], corr[order], color="tab:green", lw=1.2)
    ax.axvline(shift_deg[peak_idx], color="red", ls="--", lw=1.5,
               label=f"peak @ {shift_deg[peak_idx]:+.1f}°")
    if yaw_odom_deg is not None:
        ax.axvline(wrap_angle_deg(yaw_odom_deg), color="purple", ls=":",
                   lw=1.5, label=f"est+odom = {yaw_odom_deg:+.2f}°")
    if gt_deg is not None:
        ax.axvline(wrap_angle_deg(gt_deg), color="black", ls="-.", lw=1.5,
                   label=f"gt = {gt_deg:+.2f}°")
    ax.set_xlabel("shift (deg)"); ax.set_ylabel("correlation")
    ax.set_title("Circular cross-correlation")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---- (4) aligned overlay ----
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(theta_deg, ha, color="tab:blue", lw=1.4, label="A (stored)")
    ax.plot(theta_deg, hb_aligned, color="tab:red", lw=1.2, alpha=0.85,
            label=f"B shifted by {yaw_shown:+.1f}°")
    ax.set_xlabel("angle (deg)"); ax.set_ylabel("density")
    ax.set_title("A vs. B aligned by the estimate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"hist_yaw_wp{wp_a:04d}_wp{wp_b:04d}.png"
        fig.savefig(out, dpi=120)
        print(f"[hist] saved {out}")

    plt.show()
    return yaw_deg, yaw_odom_deg, conf, amb


def parse_pairs(items: List[str]) -> List[Tuple[int, int]]:
    pairs = []
    for item in items:
        a, b = item.split(":")
        pairs.append((int(a), int(b)))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Visualize the simple angular-histogram yaw estimator."
    )
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Pairs as stored:query, e.g. 1:104 8:90 "
                         "(default: the 4 pipeline revisits).")
    ap.add_argument("--n-bins", type=int, default=360,
                    help="Number of angular bins (default 360 = 1°).")
    ap.add_argument("--use-origin", action="store_true",
                    help="Anchor angles at the sensor origin instead of "
                         "the XY centroid.")
    ap.add_argument("--odom-disambiguate", action="store_true",
                    help="Resolve the 180-degree ambiguity with an odometry "
                         "prediction (from the trajectory poses).")
    ap.add_argument("--odom-noise-deg", type=float, default=0.0,
                    help="Add Gaussian noise to the odometry used for "
                         "disambiguation. The error is still measured against "
                         "the true (noise-free) GT poses.")
    ap.add_argument("--save-plots", type=Path, default=None,
                    help="Directory to save figures as PNG.")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs) if args.pairs else DEFAULT_PAIRS

    gt_poses = (load_ground_truth_poses(TRAJ_JSON)
                if TRAJ_JSON.exists() else None)
    odom_poses = None
    if args.odom_disambiguate and gt_poses is not None:
        odom_poses = (load_odometry_poses(TRAJ_JSON, args.odom_noise_deg)
                      if args.odom_noise_deg > 0 else gt_poses)

    def _f(val, width):
        if val is None or not np.isfinite(val):
            return f"{'-':>{width}}"
        return f"{val:>{width}.2f}"

    header = (f"{'stored':>6} {'query':>6} {'est':>9} {'est+odom':>9} "
              f"{'gt':>9} {'err':>9} {'err+odom':>9} {'conf':>7} {'amb':>6}")
    print(header)
    print("-" * len(header))

    errs, errs_odom = [], []
    for wp_a, wp_b in pairs:
        pc_a = PointCloud.from_npy(find_npy(wp_a))
        pc_b = PointCloud.from_npy(find_npy(wp_b))
        gt = (ground_truth_relative_yaw_deg(gt_poses, wp_a, wp_b)
              if gt_poses is not None else None)
        rel_odom = (ground_truth_relative_yaw_deg(odom_poses, wp_a, wp_b)
                    if odom_poses is not None else None)

        est, est_odom, conf, amb = visualize_pair(
            pc_a, pc_b, wp_a, wp_b,
            n_bins=args.n_bins,
            use_centroid=not args.use_origin,
            gt_deg=gt,
            yaw_rel_odom_deg=rel_odom,
            save_dir=args.save_plots,
        )
        err = wrap_angle_deg(est - gt) if gt is not None else float("nan")
        err_odom = (wrap_angle_deg(est_odom - gt)
                    if (est_odom is not None and gt is not None)
                    else float("nan"))
        errs.append(abs(err))
        errs_odom.append(abs(err_odom))

        print(f"{wp_a:>6} {wp_b:>6} {est:>+9.2f} {_f(est_odom, 9)} "
              f"{_f(gt, 9)} {_f(err, 9)} {_f(err_odom, 9)} "
              f"{conf:>7.1f} {amb:>6.2f}")

    valid = [e for e in errs if np.isfinite(e)]
    if valid:
        print("-" * len(header))
        print(f"mean |err|      = {np.mean(valid):.2f}°  "
              f"max = {np.max(valid):.2f}°")
    valid_o = [e for e in errs_odom if np.isfinite(e)]
    if valid_o:
        print(f"mean |err+odom| = {np.mean(valid_o):.2f}°  "
              f"max = {np.max(valid_o):.2f}°")


if __name__ == "__main__":
    main()
