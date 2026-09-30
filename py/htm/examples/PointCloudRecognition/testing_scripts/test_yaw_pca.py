# test_yaw_pca.py
"""
Test the PCA-based yaw estimator on the dataset with ground truth.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator_pca import estimate_yaw_pca
from test_pipeline_revisits import (
    load_ground_truth_poses,
    ground_truth_relative_yaw_deg,
)


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
TRAJ_JSON = DATA_DIR / "trajectory_map_ros2.json"


# The 4 revisits found earlier
REVISITS = [
    (90, 8),
    (94, 8),
    (97, 1),
    (104, 1),
]


def find_npy(wp: int) -> Path:
    matches = sorted(DATA_DIR.glob(f"wp_{wp:04d}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"wp_{wp:04d} not found")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Clip points beyond this XY radius (optional).")
    parser.add_argument("--use-centroid", action="store_true", default=True)
    parser.add_argument("--resolve-180", action="store_true",
                        help="Try to disambiguate 180° using Z asymmetry.")
    args = parser.parse_args()

    # Load ground truth
    poses = load_ground_truth_poses(TRAJ_JSON) if TRAJ_JSON.exists() else None

    print(f"{'wp_new':>6} | {'wp_stored':>9} | "
          f"{'est':>9} | {'gt':>9} | {'err':>9} | "
          f"{'err_wrap180':>11} | {'aniso':>6}")
    print("-" * 82)

    errors = []
    errors_wrap = []

    for wp_new, wp_stored in REVISITS:
        pc_a = PointCloud.from_npy(find_npy(wp_stored))
        pc_b = PointCloud.from_npy(find_npy(wp_new))

        res = estimate_yaw_pca(
            pc_a, pc_b,
            max_radius=args.max_radius,
            use_centroid=args.use_centroid,
            resolve_180_with_z=args.resolve_180,
        )

        # GT
        gt_deg: Optional[float] = None
        if poses is not None:
            gt_deg = ground_truth_relative_yaw_deg(poses, wp_stored, wp_new)

        # Errors
        if gt_deg is not None:
            err = ((res.yaw_deg - gt_deg + 180) % 360) - 180
            # Error modulo 180 (since PCA is ambiguous mod 180)
            err_180 = ((res.yaw_deg - gt_deg + 90) % 180) - 90
        else:
            err = None
            err_180 = None

        errors.append(err)
        errors_wrap.append(err_180)

        print(f"{wp_new:>6} | {wp_stored:>9} | "
              f"{res.yaw_deg:>+9.2f} | "
              f"{(gt_deg if gt_deg is not None else float('nan')):>+9.2f} | "
              f"{(err if err is not None else float('nan')):>+9.2f} | "
              f"{(err_180 if err_180 is not None else float('nan')):>+11.2f} | "
              f"{res.anisotropy:>6.2f}")

    errors_arr = np.array([e for e in errors if e is not None])
    errors_wrap_arr = np.array([e for e in errors_wrap if e is not None])

    print("-" * 82)
    print(f"Mean |err| (mod 360°): {np.abs(errors_arr).mean():.2f}°")
    print(f"Mean |err| (mod 180°): {np.abs(errors_wrap_arr).mean():.2f}°")
    print(f"Max |err|  (mod 180°): {np.abs(errors_wrap_arr).max():.2f}°")


if __name__ == "__main__":
    main()