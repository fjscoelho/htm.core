# test_yaw_diagnostics.py
"""
Visualize the histograms and correlation used by the yaw estimator.

Usage
-----
    python test_yaw_diagnostics.py
    python test_yaw_diagnostics.py --wp-a 8 --wp-b 94
    python test_yaw_diagnostics.py --method simple
"""

import sys
import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator import estimate_yaw, plot_yaw_diagnostics

# Optional: ground-truth poses
try:
    from test_pipeline_revisits import (
        load_ground_truth_poses,
        ground_truth_relative_yaw_deg,
    )
    HAS_GT_MODULE = True
except ImportError:
    HAS_GT_MODULE = False


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
TRAJ_JSON = DATA_DIR / "trajectory_map_ros2.json"


def find_npy_by_wp(wp_number: int) -> Path:
    """Find the .npy file whose name contains 'wp_XXXX'."""
    pattern = f"wp_{wp_number:04d}_*.npy"
    matches = sorted(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {DATA_DIR}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize histograms and correlation of the yaw estimator."
    )
    parser.add_argument("--wp-a", type=int, default=8,
                        help="Stored waypoint number (default: 8).")
    parser.add_argument("--wp-b", type=int, default=94,
                        help="Query waypoint number (default: 94).")
    parser.add_argument("--n-bins", type=int, default=360,
                        help="Number of angular bins (default: 360).")
    parser.add_argument("--n-rings", type=int, default=10,
                        help="Number of radial rings (default: 10).")
    parser.add_argument("--method", type=str, default="stratified",
                        choices=["simple", "stratified"],
                        help="Histogram method (default: stratified).")
    parser.add_argument("--use-centroid", action="store_true",
                        help="Use XY centroid as anchor "
                             "(default: sensor origin).")
    args = parser.parse_args()

    # ---- Load clouds ----
    path_a = find_npy_by_wp(args.wp_a)
    path_b = find_npy_by_wp(args.wp_b)
    print(f"A (stored): {path_a.name}")
    print(f"B (query) : {path_b.name}")

    pc_a = PointCloud.from_npy(path_a)
    pc_b = PointCloud.from_npy(path_b)
    print(f"A: {pc_a.n_points} points")
    print(f"B: {pc_b.n_points} points")

    # ---- Estimate yaw ----
    res = estimate_yaw(
        pc_a, pc_b,
        n_bins=args.n_bins,
        method=args.method,
        n_rings=args.n_rings,
        use_centroid=args.use_centroid,
        refine_icp=True,
    )
    print(f"\nEstimated yaw : {res.yaw_deg:+.2f}°")
    print(f"Confidence    : {res.confidence:.3f}")
    print(f"Ambiguity     : {res.ambiguity:.3f}")

    # ---- Ground truth (if available) ----
    gt_deg = None
    if HAS_GT_MODULE and TRAJ_JSON.exists():
        poses = load_ground_truth_poses(TRAJ_JSON)
        # Note: "A" is stored (wp_a) and "B" is query (wp_b), so the
        # relative yaw B w.r.t. A is gt(wp_a -> wp_b).
        gt_deg = ground_truth_relative_yaw_deg(poses, args.wp_a, args.wp_b)
        if gt_deg is not None:
            err = ((res.yaw_deg - gt_deg + 180) % 360) - 180
            print(f"Ground-truth  : {gt_deg:+.2f}°")
            print(f"Error         : {err:+.2f}°")

    # ---- Plot ----
    title = (f"Yaw diagnostics: wp_{args.wp_a:04d} (A) vs "
             f"wp_{args.wp_b:04d} (B)  —  method={args.method}")
    plot_yaw_diagnostics(
        pc_a, pc_b,
        n_bins=args.n_bins,
        method=args.method,
        n_rings=args.n_rings,
        use_centroid=args.use_centroid,
        estimated_yaw_deg=res.yaw_deg,
        gt_yaw_deg=gt_deg,
        title=title,
    )
    plt.show()


if __name__ == "__main__":
    main()