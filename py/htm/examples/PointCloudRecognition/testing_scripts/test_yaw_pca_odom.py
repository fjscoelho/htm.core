# test_yaw_pca_odom.py
"""
Test PCA + odometry disambiguation on the 4 revisits.
"""

import sys
from pathlib import Path
import numpy as np

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator_pca import (
    estimate_yaw_pca,
    estimate_yaw_pca_with_odometry,
)
from test_pipeline_revisits import (
    load_ground_truth_poses,
    ground_truth_relative_yaw_deg,
)


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
TRAJ_JSON = DATA_DIR / "trajectory_map_ros2.json"

REVISITS = [(90, 8), (94, 8), (97, 1), (104, 1)]


def _wrap_pi(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def add_odom_noise(poses, std_deg, seed=42):
    """Return a copy of poses with Gaussian yaw noise added."""
    rng = np.random.default_rng(seed)
    out = {}
    for wp, entry in poses.items():
        entry = dict(entry)
        noise = np.deg2rad(rng.normal(0, std_deg))
        entry["yaw_rad"] = _wrap_pi(entry["yaw_rad"] + noise)
        out[wp] = entry
    return out


def main():
    gt = load_ground_truth_poses(TRAJ_JSON)

    for noise_std in [0.0, 5.0, 15.0, 30.0]:
        odom = add_odom_noise(gt, noise_std, seed=42) if noise_std > 0 else gt
        print(f"\n=== Odom noise std = {noise_std}° ===")
        print(f"{'wp_new':>6} | {'wp_stored':>9} | {'aniso':>6} | "
              f"{'pca':>9} | {'pca+odom':>10} | {'gt':>9} | "
              f"{'err_pca':>8} | {'err_pca+odom':>13}")
        print("-" * 95)

        errors_pca = []
        errors_pca_odom = []

        for wp_new, wp_stored in REVISITS:
            pc_a = PointCloud.from_npy(
                sorted(DATA_DIR.glob(f"wp_{wp_stored:04d}_*.npy"))[0])
            pc_b = PointCloud.from_npy(
                sorted(DATA_DIR.glob(f"wp_{wp_new:04d}_*.npy"))[0])

            # GT yaw relative
            gt_deg = ground_truth_relative_yaw_deg(gt, wp_stored, wp_new)

            # PCA (mod 180)
            pca_res = estimate_yaw_pca(pc_a, pc_b)

            # PCA + odom
            pca_odom = estimate_yaw_pca_with_odometry(
                pc_a, pc_b,
                yaw_a_odom_rad=odom[wp_stored]["yaw_rad"],
                yaw_b_odom_rad=odom[wp_new]["yaw_rad"],
            )

            err_pca = ((pca_res.yaw_deg - gt_deg + 180) % 360) - 180
            err_pca_odom = ((pca_odom.yaw_deg - gt_deg + 180) % 360) - 180

            errors_pca.append(err_pca)
            errors_pca_odom.append(err_pca_odom)

            print(f"{wp_new:>6} | {wp_stored:>9} | "
                  f"{pca_res.anisotropy:>6.2f} | "
                  f"{pca_res.yaw_deg:>+9.2f} | "
                  f"{pca_odom.yaw_deg:>+10.2f} | "
                  f"{gt_deg:>+9.2f} | "
                  f"{err_pca:>+8.2f} | {err_pca_odom:>+13.2f}")

        errors_pca = np.array(errors_pca)
        errors_pca_odom = np.array(errors_pca_odom)
        print(f"\n  PCA only     — mean |err| = {np.abs(errors_pca).mean():.2f}°, "
              f"max |err| = {np.abs(errors_pca).max():.2f}°")
        print(f"  PCA + odom   — mean |err| = {np.abs(errors_pca_odom).mean():.2f}°, "
              f"max |err| = {np.abs(errors_pca_odom).max():.2f}°")


if __name__ == "__main__":
    main()