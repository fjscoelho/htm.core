# test_pipeline_revisits.py
"""
End-to-end test of the place-recognition pipeline with yaw estimation.

Workflow
--------
For each point cloud in chronological order:
    1. Encode to SDR.
    2. Query the PlaceDatabase for the best match.
    3. If matched AND the matched place was created > TEMPORAL_FILTER waypoints
       ago, treat this as a candidate "revisit".
    4. For each candidate revisit, estimate the relative yaw between the
       stored and new clouds.
    5. Store the result.

Outputs
-------
    revisits.csv       — tabular results
    (optional) plots

Usage
-----
    python test_pipeline_revisits.py
    python test_pipeline_revisits.py --plot
    python test_pipeline_revisits.py --match-threshold 0.75 --temporal-filter 5
"""

import sys
import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional
import json
from scipy.spatial.transform import Rotation as SciRotation

import numpy as np
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase
from src.yaw_estimator import estimate_yaw


# ---------- Config ----------
DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")
OUTPUT_CSV = Path("revisits.csv")

DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_TEMPORAL_FILTER = 5     # ignore matches where |wp_new - wp_stored| <= this
DEFAULT_YAW_TOLERANCE_DEG = 15.0


# ---------- Helpers ----------
def encode_path(encoder, npy_path: Path) -> np.ndarray:
    pc = PointCloud.from_npy(npy_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    return encoder.encode(desc.to_vector())


def find_best_match(sdr: np.ndarray, db: PlaceDatabase):
    """
    Return (place_idx, overlap_count, overlap_ratio) for the best matching
    place in the database, or (-1, 0, 0.0) if the database is empty.
    """
    if not db.places:
        return -1, 0, 0.0
    n_active = int(sdr.sum())
    best_idx, best_overlap = -1, 0
    for idx, place in enumerate(db.places):
        ov = int(np.logical_and(sdr, place.sdr).sum())
        if ov > best_overlap:
            best_overlap = ov
            best_idx = idx
    ratio = best_overlap / n_active if n_active > 0 else 0.0
    return best_idx, best_overlap, ratio


# ---------- Main pipeline ----------
def run_pipeline(
    encoder,
    npy_paths: List[Path],
    match_threshold: float,
    temporal_filter: int,
    yaw_tolerance_deg: float,
    gt_poses: Optional[dict[int, dict]] = None,
    verbose: bool = True,
):
    """
    Run the full pipeline over the dataset.

    If `gt_poses` is provided, adds the ground-truth relative yaw to each
    revisit record for comparison.
    """
    db = PlaceDatabase(
        sdr_size=encoder.total_size,
        match_threshold=match_threshold,
        yaw_tolerance_deg=yaw_tolerance_deg,
    )

    place_source_wp_idx: dict[int, int] = {}
    place_source_path: dict[int, Path] = {}

    revisits: List[dict] = []

    if verbose:
        header = (f"\n{'wp':>5} | {'ratio':>6} | {'place':>5} | {'match':>5} "
                  f"| {'stored_wp':>9} | {'Δwp':>4} | {'decision':>18}")
        print(header)
        print("-" * 72)

    for wp_idx, npy_path in enumerate(npy_paths):
        sdr = encode_path(encoder, npy_path)

        best_idx, best_overlap, best_ratio = find_best_match(sdr, db)
        stored_wp_idx: Optional[int] = None
        stored_path: Optional[Path] = None
        if best_idx >= 0:
            best_place = db.places[best_idx]
            stored_wp_idx = place_source_wp_idx.get(best_place.place_id)
            stored_path = place_source_path.get(best_place.place_id)

        place_id, matched, _ = db.match_or_create(sdr, yaw_rad=0.0,
                                                  label=npy_path.name)

        if not matched:
            place_source_wp_idx[place_id] = wp_idx
            place_source_path[place_id] = npy_path

        if not matched:
            decision = "new place"
        else:
            delta_wp = abs(wp_idx - stored_wp_idx) if stored_wp_idx is not None else None
            if delta_wp is None:
                decision = "matched (no src?)"
            elif delta_wp <= temporal_filter:
                decision = f"recent (Δ={delta_wp}) → skip"
            else:
                decision = f"revisit (Δ={delta_wp}) → yaw"

        if matched and stored_path is not None and stored_wp_idx is not None:
            delta_wp = abs(wp_idx - stored_wp_idx)
            if delta_wp > temporal_filter:
                t0 = time.perf_counter()
                yaw_res = estimate_yaw(
                    PointCloud.from_npy(stored_path),
                    PointCloud.from_npy(npy_path),
                    n_bins=360,
                    refine_icp=False,
                )
                t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Ground-truth relative yaw, if available
                yaw_gt_deg: Optional[float] = None
                yaw_gt_err: Optional[float] = None
                if gt_poses is not None:
                    yaw_gt_deg = ground_truth_relative_yaw_deg(
                        gt_poses, stored_wp_idx, wp_idx
                    )
                    if yaw_gt_deg is not None:
                        yaw_gt_err = wrap_angle_deg(yaw_res.yaw_deg - yaw_gt_deg)

                revisits.append({
                    "wp_new":         wp_idx,
                    "wp_stored":      stored_wp_idx,
                    "wp_delta":       delta_wp,
                    "place_id":       place_id,
                    "overlap_ratio":  best_ratio,
                    "yaw_deg":        yaw_res.yaw_deg,
                    "yaw_gt_deg":     yaw_gt_deg,
                    "yaw_err_deg":    yaw_gt_err,
                    "confidence":     yaw_res.confidence,
                    "ambiguity":      yaw_res.ambiguity,
                    "time_ms":        t_elapsed_ms,
                    "wp_new_name":    npy_path.name,
                    "wp_stored_name": stored_path.name,
                })

        if verbose:
            wp_tag = npy_path.name.split("_")[1]
            stored_tag = f"wp_{stored_wp_idx:04d}" if stored_wp_idx is not None else "-"
            delta_str = (str(abs(wp_idx - stored_wp_idx))
                         if stored_wp_idx is not None and matched else "-")
            print(f"{wp_tag:>5} | {best_ratio:>6.3f} | {place_id:>5d} | "
                  f"{str(matched):>5} | {stored_tag:>9} | {delta_str:>4} | "
                  f"{decision:>18}")

    return revisits, db


# ---------- Persistence ----------
def save_revisits_csv(revisits: List[dict], path: Path) -> None:
    if not revisits:
        print(f"\n[CSV] No revisits to save.")
        return
    fieldnames = list(revisits[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(revisits)
    print(f"\n[CSV] Saved {len(revisits)} revisits to: {path.resolve()}")


# ---------- Plotting ----------
def plot_summary(revisits: List[dict]) -> None:
    if not revisits:
        print("[plot] No revisits to plot.")
        return

    # ... (código existente) ...

    # Extra plot: yaw comparison (only if gt is available)
    rows_gt = [r for r in revisits if r["yaw_gt_deg"] is not None]
    if rows_gt:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Estimated vs. GT scatter
        ax = axes[0]
        est = [r["yaw_deg"] for r in rows_gt]
        gt = [r["yaw_gt_deg"] for r in rows_gt]
        delta_wp = [r["wp_delta"] for r in rows_gt]
        sc = ax.scatter(gt, est, c=delta_wp, cmap="viridis",
                        s=100, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="|Δwp|")
        # Diagonal reference line
        lims = [-185, 185]
        ax.plot(lims, lims, "r--", alpha=0.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("ground-truth yaw (deg)")
        ax.set_ylabel("estimated yaw (deg)")
        ax.set_title("Estimated vs. GT yaw")
        ax.legend()
        ax.grid(alpha=0.3)

        # Error vs. Delta wp
        ax = axes[1]
        err = [r["yaw_err_deg"] for r in rows_gt]
        ax.scatter(delta_wp, err, s=100, edgecolors="k", linewidths=0.5,
                   color="tab:red")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("|Δwp|")
        ax.set_ylabel("yaw error (est - gt) [deg]")
        ax.set_title("Yaw error vs. temporal distance")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ---------- Summary ----------
def print_summary(revisits: List[dict], n_total: int) -> None:
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total point clouds processed : {n_total}")
    print(f"Revisits detected             : {len(revisits)} "
          f"({100*len(revisits)/n_total:.1f}%)")

    if not revisits:
        print("No revisits to summarize.")
        return

    yaw = np.array([r["yaw_deg"] for r in revisits])
    conf = np.array([r["confidence"] for r in revisits])
    ambig = np.array([r["ambiguity"] for r in revisits])
    ratio = np.array([r["overlap_ratio"] for r in revisits])
    delta = np.array([r["wp_delta"] for r in revisits])
    t_ms = np.array([r["time_ms"] for r in revisits])

    print(f"\nYaw estimate (deg):")
    print(f"  mean   = {yaw.mean():+.3f}")
    print(f"  std    = {yaw.std():.3f}")
    print(f"  min    = {yaw.min():+.3f}")
    print(f"  max    = {yaw.max():+.3f}")

    print(f"\nConfidence:")
    print(f"  mean   = {conf.mean():.3f}")
    print(f"  min    = {conf.min():.3f}")
    print(f"  # >5   = {(conf > 5).sum()} / {len(conf)} "
          f"({100*(conf>5).mean():.1f}%)")
    print(f"  # >8   = {(conf > 8).sum()} / {len(conf)} "
          f"({100*(conf>8).mean():.1f}%)")

    print(f"\nAmbiguity:")
    print(f"  mean   = {ambig.mean():.3f}")
    print(f"  # <0.5 = {(ambig < 0.5).sum()} / {len(ambig)} "
          f"({100*(ambig<0.5).mean():.1f}%)")
    print(f"  # <0.3 = {(ambig < 0.3).sum()} / {len(ambig)} "
          f"({100*(ambig<0.3).mean():.1f}%)")

    print(f"\nOverlap ratio:")
    print(f"  mean   = {ratio.mean():.3f}")
    print(f"  min    = {ratio.min():.3f}")
    print(f"  max    = {ratio.max():.3f}")

    print(f"\nTemporal distance (|Δwp|):")
    print(f"  mean   = {delta.mean():.1f}")
    print(f"  max    = {delta.max()}")

    print(f"\nTime per yaw estimate:")
    print(f"  mean   = {t_ms.mean():.2f} ms")
    print(f"  max    = {t_ms.max():.2f} ms")


# ---------- Ground-truth poses ----------
def load_ground_truth_poses(json_path: Path) -> dict[int, dict]:
    """
    Load ground-truth waypoint poses from a GraphNav trajectory_map_ros2.json.

    Returns
    -------
    poses : dict[int, dict]
        Maps waypoint number (int) -> {
            "yaw_rad": float,         # yaw around Z, in (-pi, pi]
            "yaw_deg": float,
            "position": (x, y, z),    # seed_tform_waypoint position
            "name": str,
        }
    """
    data = json.loads(Path(json_path).read_text())
    poses: dict[int, dict] = {}

    for entry in data:
        num = int(entry["number"])
        name = entry.get("name", f"waypoint_{num}")

        tf = entry.get("transforms", {}).get("seed_tform_waypoint")
        if tf is None:
            continue

        pos = tf.get("position")
        quat = tf.get("rotation")  # [x, y, z, w]

        if pos is None or quat is None:
            continue

        # Convert quaternion -> Euler (XYZ) and extract yaw (rotation around Z)
        rot = SciRotation.from_quat(quat)   # expects [x, y, z, w]
        # 'zyx' means: apply Z, then Y, then X — the yaw is the Z component
        euler = rot.as_euler("zyx")
        yaw_rad = float(euler[0])
        # Wrap to (-pi, pi]
        yaw_rad = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

        poses[num] = {
            "yaw_rad":  yaw_rad,
            "yaw_deg":  float(np.rad2deg(yaw_rad)),
            "position": tuple(float(v) for v in pos),
            "name":     name,
        }
    return poses


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees into (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def ground_truth_relative_yaw_deg(
    poses: dict[int, dict],
    wp_a: int,
    wp_b: int,
) -> Optional[float]:
    """
    Compute the relative yaw of waypoint B with respect to waypoint A,
    using ground-truth orientations.

    Convention: if the histogram-based estimator returns the rotation that
    aligns cloud A to cloud B (i.e., how much A must rotate to match B),
    then the GT equivalent is `yaw_B - yaw_A` (wrapped).

    Returns None if either waypoint is missing.
    """
    if wp_a not in poses or wp_b not in poses:
        return None
    yaw_a = poses[wp_a]["yaw_rad"]
    yaw_b = poses[wp_b]["yaw_rad"]
    delta = yaw_b - yaw_a
    return float(np.rad2deg(((delta + np.pi) % (2 * np.pi)) - np.pi))

def print_yaw_comparison(revisits: List[dict]) -> None:
    """
    Print a per-revisit table comparing estimated vs. ground-truth yaw.
    """
    rows_with_gt = [r for r in revisits if r["yaw_gt_deg"] is not None]
    if not rows_with_gt:
        print("\n[yaw comparison] No ground-truth data available.")
        return

    print("\n" + "=" * 92)
    print("YAW COMPARISON: ESTIMATED vs. GROUND-TRUTH")
    print("=" * 92)
    print(f"{'wp_new':>6} | {'wp_stored':>9} | {'Δwp':>4} | "
          f"{'est_yaw':>9} | {'gt_yaw':>9} | {'err':>8} | "
          f"{'conf':>6} | {'ambig':>6} | {'ratio':>6}")
    print("-" * 92)
    for r in rows_with_gt:
        print(f"{r['wp_new']:>6} | {r['wp_stored']:>9} | "
              f"{r['wp_delta']:>4} | "
              f"{r['yaw_deg']:>+9.2f} | {r['yaw_gt_deg']:>+9.2f} | "
              f"{r['yaw_err_deg']:>+8.2f} | "
              f"{r['confidence']:>6.2f} | {r['ambiguity']:>6.2f} | "
              f"{r['overlap_ratio']:>6.3f}")

    errors = np.array([r["yaw_err_deg"] for r in rows_with_gt])
    abs_err = np.abs(errors)
    print("-" * 92)
    print(f"Yaw error (est - gt):")
    print(f"  mean       = {errors.mean():+.3f}°")
    print(f"  mean |err| = {abs_err.mean():.3f}°")
    print(f"  median |err| = {np.median(abs_err):.3f}°")
    print(f"  max |err|  = {abs_err.max():.3f}°")
    print(f"  # < 10°    = {(abs_err < 10).sum()} / {len(abs_err)}")
    print(f"  # < 30°    = {(abs_err < 30).sum()} / {len(abs_err)}")
    print(f"  # > 90°    = {(abs_err > 90).sum()} / {len(abs_err)}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Run the full place-recognition pipeline with yaw estimation."
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--match-threshold", type=float,
                        default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--temporal-filter", type=int,
                        default=DEFAULT_TEMPORAL_FILTER)
    parser.add_argument("--yaw-tolerance-deg", type=float,
                        default=DEFAULT_YAW_TOLERANCE_DEG)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--trajectory-json", type=Path,
                        default=DATA_DIR / "trajectory_map_ros2.json",
                        help="Path to trajectory_map_ros2.json (ground truth).")
    args = parser.parse_args()

    # ---- 1. Load encoder ----
    print(f"Loading encoder from {CONFIG_PATH}...")
    encoder = load_encoder_from_config(CONFIG_PATH)
    print(f"  SDR size: {encoder.total_size} bits")

    # ---- 2. Load ground-truth poses ----
    gt_poses: Optional[dict[int, dict]] = None
    if args.trajectory_json.exists():
        print(f"Loading ground-truth poses from {args.trajectory_json}...")
        gt_poses = load_ground_truth_poses(args.trajectory_json)
        print(f"  Loaded {len(gt_poses)} waypoint poses.")
    else:
        print(f"[warn] No trajectory JSON at {args.trajectory_json}; "
              f"yaw comparison will be skipped.")

    # ---- 3. Discover dataset ----
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    print(f"Found {len(npy_paths)} point clouds\n")
    if not npy_paths:
        raise SystemExit("No .npy files found. Check DATA_DIR.")

    # ---- 4. Run pipeline ----
    t_start = time.time()
    revisits, db = run_pipeline(
        encoder,
        npy_paths,
        match_threshold=args.match_threshold,
        temporal_filter=args.temporal_filter,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
        gt_poses=gt_poses,
        verbose=not args.no_verbose,
    )
    t_total = time.time() - t_start

    # ---- 5. Summary ----
    print_summary(revisits, n_total=len(npy_paths))
    print_yaw_comparison(revisits)
    print(f"\nTotal pipeline wall time: {t_total:.2f} s")

    # ---- 6. Save CSV ----
    save_revisits_csv(revisits, OUTPUT_CSV)

    # ---- 7. Plots ----
    if args.plot:
        plot_summary(revisits)


if __name__ == "__main__":
    main()