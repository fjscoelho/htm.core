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

Yaw estimation strategy
-----------------------
    - If odometry is available: use PCA-2D + odometry.
        * PCA gives the principal axis (ambiguous modulo 180 degrees).
        * Odometry (approximate heading) breaks the ambiguity.
    - If no odometry is available: fall back to the stratified histogram.

Outputs
-------
    revisits.csv       — tabular results
    (optional) plots

Usage
-----
    python test_pipeline_revisits.py
    python test_pipeline_revisits.py --plot
    python test_pipeline_revisits.py --odom-noise-deg 15.0
    python test_pipeline_revisits.py --match-threshold 0.75 --temporal-filter 5
"""

import sys
import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as SciRotation

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase
from src.yaw_estimator import (
    angular_histogram,
    angular_histogram_stratified,
    estimate_yaw_from_histograms,
    estimate_yaw_from_histograms_with_odometry,
    estimate_yaw_from_stratified_histograms,
    estimate_yaw_from_stratified_histograms_with_odometry,
)
from src.yaw_estimator_pca import (
    estimate_yaw_pca,
    estimate_yaw_pca_with_odometry,
)


# ---------- Config ----------
DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")
OUTPUT_CSV = Path("revisits.csv")

DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_TEMPORAL_FILTER = 5     # ignore matches where |wp_new - wp_stored| <= this
DEFAULT_YAW_TOLERANCE_DEG = 15.0


# ============================================================
# Helpers
# ============================================================
def _wrap_pi(angle_rad: float) -> float:
    """Wrap an angle in radians into (-pi, pi]."""
    return ((angle_rad + np.pi) % (2 * np.pi)) - np.pi


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees into (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


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


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(
    encoder,
    npy_paths: List[Path],
    match_threshold: float,
    temporal_filter: int,
    yaw_tolerance_deg: float,
    gt_poses: Optional[dict[int, dict]] = None,
    verbose: bool = True,
    yaw_method_opt: str = "auto",
    hist_bins: int = 360,
    hist_rings: int = 10,
):
    """
    Run the full pipeline over the dataset.

    Yaw strategy (`yaw_method_opt`):
        - "auto": PCA-2D + odometry if `gt_poses` is available, otherwise
          the stratified angular histogram.
        - "pca": plain 2D PCA.
        - "pca+odom": 2D PCA disambiguated by odometry (needs `gt_poses`).
        - "simple": single angular histogram + circular cross-correlation.
        - "stratified": radial-ring angular histograms.
        - "simple+odom", "stratified+odom": same as above, with the
          180-degree ambiguity resolved by odometry (needs `gt_poses`).

    Parameters
    ----------
    gt_poses : dict[int, dict] or None
        Odometry-like poses (yaw_rad per waypoint). Used both to disambiguate
        PCA and to compute ground-truth relative yaw for reporting.
    hist_bins : int
        Angular bins for the histogram methods (default 360 = 1 deg).
    hist_rings : int
        Number of radial rings for "stratified".
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
            delta_wp = (abs(wp_idx - stored_wp_idx)
                        if stored_wp_idx is not None else None)
            if delta_wp is None:
                decision = "matched (no src?)"
            elif delta_wp <= temporal_filter:
                decision = f"recent (Δ={delta_wp}) → skip"
            else:
                decision = f"revisit (Δ={delta_wp}) → yaw"

        # ---- Yaw estimation for genuine revisits ----
        if matched and stored_path is not None and stored_wp_idx is not None:
            delta_wp = abs(wp_idx - stored_wp_idx)
            if delta_wp > temporal_filter:
                pc_a = PointCloud.from_npy(stored_path)
                pc_b = PointCloud.from_npy(npy_path)

                t_start_yaw = time.perf_counter()

                yaw_deg: float = float("nan")
                yaw_method: str = "?"
                yaw_conf: float = float("nan")
                yaw_ambig: float = float("nan")
                anisotropy: float = float("nan")
                hist_ms: float = float("nan")
                corr_ms: float = float("nan")

                # ---- Pick the effective method ----
                if yaw_method_opt == "auto":
                    eff = "pca+odom" if gt_poses is not None else "stratified"
                else:
                    eff = yaw_method_opt

                if eff == "pca":
                    pca_res = estimate_yaw_pca(pc_a, pc_b)
                    yaw_deg = pca_res.yaw_deg
                    anisotropy = pca_res.anisotropy
                    yaw_method = "pca2d"
                elif eff == "pca+odom":
                    if gt_poses is None:
                        raise ValueError(
                            "yaw method 'pca+odom' requires ground-truth poses"
                        )
                    pca_final = estimate_yaw_pca_with_odometry(
                        pc_a, pc_b,
                        yaw_a_odom_rad=gt_poses[stored_wp_idx]["yaw_rad"],
                        yaw_b_odom_rad=gt_poses[wp_idx]["yaw_rad"],
                    )
                    yaw_deg = pca_final.yaw_deg
                    anisotropy = pca_final.anisotropy
                    yaw_method = "pca2d+odom"
                else:
                    # --- Angular-histogram family ---
                    hist_kind = eff.split("+")[0]     # simple | stratified
                    use_odom = eff.endswith("+odom")
                    if use_odom and gt_poses is None:
                        raise ValueError(
                            f"yaw method '{eff}' requires ground-truth poses"
                        )

                    # Time the histogram construction and the comparison
                    # (cross-correlation) separately.
                    t_h0 = time.perf_counter()
                    if hist_kind == "stratified":
                        ha = angular_histogram_stratified(
                            pc_a, n_bins=hist_bins, n_rings=hist_rings)
                        hb = angular_histogram_stratified(
                            pc_b, n_bins=hist_bins, n_rings=hist_rings)
                    else:
                        ha = angular_histogram(pc_a, n_bins=hist_bins)
                        hb = angular_histogram(pc_b, n_bins=hist_bins)
                    t_h1 = time.perf_counter()

                    if use_odom:
                        yaw_a_odom = gt_poses[stored_wp_idx]["yaw_rad"]
                        yaw_b_odom = gt_poses[wp_idx]["yaw_rad"]
                        if hist_kind == "stratified":
                            yaw_rad_h, yaw_conf, yaw_ambig = \
                                estimate_yaw_from_stratified_histograms_with_odometry(
                                    ha, hb, yaw_a_odom, yaw_b_odom)
                        else:
                            yaw_rad_h, yaw_conf, yaw_ambig = \
                                estimate_yaw_from_histograms_with_odometry(
                                    ha, hb, yaw_a_odom, yaw_b_odom)
                    elif hist_kind == "stratified":
                        yaw_rad_h, yaw_conf, yaw_ambig = \
                            estimate_yaw_from_stratified_histograms(ha, hb)
                    else:
                        yaw_rad_h, yaw_conf, yaw_ambig = \
                            estimate_yaw_from_histograms(ha, hb)
                    t_h2 = time.perf_counter()

                    yaw_deg = float(np.rad2deg(yaw_rad_h))
                    yaw_method = eff
                    hist_ms = (t_h1 - t_h0) * 1000.0
                    corr_ms = (t_h2 - t_h1) * 1000.0

                t_elapsed_ms = (time.perf_counter() - t_start_yaw) * 1000.0

                # --- Ground-truth relative yaw, if available ---
                yaw_gt_deg: Optional[float] = None
                yaw_gt_err: Optional[float] = None
                if gt_poses is not None:
                    yaw_gt_deg = ground_truth_relative_yaw_deg(
                        gt_poses, stored_wp_idx, wp_idx
                    )
                    if yaw_gt_deg is not None:
                        yaw_gt_err = wrap_angle_deg(yaw_deg - yaw_gt_deg)

                revisits.append({
                    "wp_new":         wp_idx,
                    "wp_stored":      stored_wp_idx,
                    "wp_delta":       delta_wp,
                    "place_id":       place_id,
                    "overlap_ratio":  best_ratio,
                    "yaw_deg":        yaw_deg,
                    "yaw_gt_deg":     yaw_gt_deg,
                    "yaw_err_deg":    yaw_gt_err,
                    "confidence":     yaw_conf,
                    "ambiguity":      yaw_ambig,
                    "anisotropy":     anisotropy,
                    "yaw_method":     yaw_method,
                    "hist_ms":        hist_ms,
                    "corr_ms":        corr_ms,
                    "time_ms":        t_elapsed_ms,
                    "wp_new_name":    npy_path.name,
                    "wp_stored_name": stored_path.name,
                })

        if verbose:
            wp_tag = npy_path.name.split("_")[1]
            stored_tag = (f"wp_{stored_wp_idx:04d}"
                          if stored_wp_idx is not None else "-")
            delta_str = (str(abs(wp_idx - stored_wp_idx))
                         if stored_wp_idx is not None and matched else "-")
            print(f"{wp_tag:>5} | {best_ratio:>6.3f} | {place_id:>5d} | "
                  f"{str(matched):>5} | {stored_tag:>9} | {delta_str:>4} | "
                  f"{decision:>18}")

    return revisits, db


# ============================================================
# Persistence
# ============================================================
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


# ============================================================
# Ground-truth poses
# ============================================================
def load_ground_truth_poses(json_path: Path) -> dict[int, dict]:
    """
    Load ground-truth waypoint poses from a GraphNav trajectory_map_ros2.json.
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

        rot = SciRotation.from_quat(quat)
        euler = rot.as_euler("zyx")
        yaw_rad = float(euler[0])
        yaw_rad = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

        poses[num] = {
            "yaw_rad":  yaw_rad,
            "yaw_deg":  float(np.rad2deg(yaw_rad)),
            "position": tuple(float(v) for v in pos),
            "name":     name,
        }
    return poses


def load_odometry_poses(
    json_path: Path,
    noise_std_deg: float = 15.0,
    seed: int = 42,
) -> dict[int, dict]:
    """
    Load GT poses and add Gaussian yaw noise to simulate odometric drift.
    Set noise_std_deg=0 to get the exact GT back.
    """
    poses = load_ground_truth_poses(json_path)
    if noise_std_deg <= 0:
        return poses

    rng = np.random.default_rng(seed)
    for wp, entry in poses.items():
        noise = np.deg2rad(rng.normal(0, noise_std_deg))
        entry["yaw_rad"] = _wrap_pi(entry["yaw_rad"] + noise)
        entry["yaw_deg"] = float(np.rad2deg(entry["yaw_rad"]))
    return poses


def ground_truth_relative_yaw_deg(
    poses: dict[int, dict],
    wp_a: int,
    wp_b: int,
) -> Optional[float]:
    """Relative yaw of B w.r.t. A from GT poses, wrapped to (-180, 180]."""
    if wp_a not in poses or wp_b not in poses:
        return None
    yaw_a = poses[wp_a]["yaw_rad"]
    yaw_b = poses[wp_b]["yaw_rad"]
    delta = yaw_b - yaw_a
    return float(np.rad2deg(((delta + np.pi) % (2 * np.pi)) - np.pi))


# ============================================================
# Summary / comparison tables
# ============================================================
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

    yaw   = np.array([r["yaw_deg"]     for r in revisits])
    ratio = np.array([r["overlap_ratio"] for r in revisits])
    delta = np.array([r["wp_delta"]    for r in revisits])
    t_ms  = np.array([r["time_ms"]     for r in revisits])
    aniso = np.array([r["anisotropy"]  for r in revisits])

    methods = [r["yaw_method"] for r in revisits]
    method_counts = {m: methods.count(m) for m in set(methods)}

    print(f"\nMethod breakdown:")
    for m, c in method_counts.items():
        print(f"  {m:<13}: {c}")

    print(f"\nYaw estimate (deg):")
    print(f"  mean   = {yaw.mean():+.3f}")
    print(f"  std    = {yaw.std():.3f}")
    print(f"  min    = {yaw.min():+.3f}")
    print(f"  max    = {yaw.max():+.3f}")

    print(f"\nAnisotropy (PCA):")
    if np.isfinite(aniso).any():
        print(f"  mean   = {np.nanmean(aniso):.3f}")
        print(f"  min    = {np.nanmin(aniso):.3f}")
        print(f"  max    = {np.nanmax(aniso):.3f}")
    else:
        print(f"  n/a (PCA not used)")

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


def print_yaw_comparison(revisits: List[dict]) -> None:
    """Per-revisit table: estimated vs. GT yaw, plus timing breakdown."""
    rows_with_gt = [r for r in revisits if r["yaw_gt_deg"] is not None]
    if not rows_with_gt:
        print("\n[yaw comparison] No ground-truth data available.")
        return

    def _fmt(val, width: int):
        if val is None or not np.isfinite(val):
            return f"{'-':>{width}}"
        return f"{val:>{width}.2f}"

    print("\n" + "=" * 122)
    print("YAW COMPARISON: ESTIMATED vs. GROUND-TRUTH")
    print("=" * 122)
    print(f"{'wp_new':>6} | {'wp_stored':>9} | {'Δwp':>4} | "
          f"{'est_yaw':>9} | {'gt_yaw':>9} | {'err':>8} | "
          f"{'aniso':>6} | {'method':>13} | {'hist_ms':>8} | "
          f"{'corr_ms':>8} | {'t_ms':>7}")
    print("-" * 122)
    for r in rows_with_gt:
        print(f"{r['wp_new']:>6} | {r['wp_stored']:>9} | "
              f"{r['wp_delta']:>4} | "
              f"{r['yaw_deg']:>+9.2f} | {r['yaw_gt_deg']:>+9.2f} | "
              f"{r['yaw_err_deg']:>+8.2f} | "
              f"{_fmt(r.get('anisotropy'), 6)} | "
              f"{r['yaw_method']:>13} | "
              f"{_fmt(r.get('hist_ms'), 8)} | "
              f"{_fmt(r.get('corr_ms'), 8)} | "
              f"{r['time_ms']:>7.2f}")

    errors = np.array([r["yaw_err_deg"] for r in rows_with_gt])
    abs_err = np.abs(errors)
    print("-" * 122)
    print(f"Yaw error (est - gt):")
    print(f"  mean       = {errors.mean():+.3f}°")
    print(f"  mean |err| = {abs_err.mean():.3f}°")
    print(f"  median |err| = {np.median(abs_err):.3f}°")
    print(f"  max |err|  = {abs_err.max():.3f}°")
    print(f"  # < 10°    = {(abs_err < 10).sum()} / {len(abs_err)}")
    print(f"  # < 30°    = {(abs_err < 30).sum()} / {len(abs_err)}")
    print(f"  # > 90°    = {(abs_err > 90).sum()} / {len(abs_err)}")

    # --- Timing breakdown for the histogram methods ---
    hist_vals = np.array([r.get("hist_ms", np.nan) for r in rows_with_gt],
                         dtype=float)
    corr_vals = np.array([r.get("corr_ms", np.nan) for r in rows_with_gt],
                         dtype=float)
    tot_vals = np.array([r["time_ms"] for r in rows_with_gt], dtype=float)
    if np.isfinite(hist_vals).any():
        print("\nHistogram timing per comparison (ms):")
        print(f"  hist build : mean={np.nanmean(hist_vals):.3f}  "
              f"median={np.nanmedian(hist_vals):.3f}  "
              f"max={np.nanmax(hist_vals):.3f}")
        print(f"  comparison : mean={np.nanmean(corr_vals):.3f}  "
              f"median={np.nanmedian(corr_vals):.3f}  "
              f"max={np.nanmax(corr_vals):.3f}")
        print(f"  total yaw  : mean={np.nanmean(tot_vals):.3f}  "
              f"median={np.nanmedian(tot_vals):.3f}  "
              f"max={np.nanmax(tot_vals):.3f}")


# ============================================================
# Plotting
# ============================================================
def plot_summary(revisits: List[dict]) -> None:
    if not revisits:
        print("[plot] No revisits to plot.")
        return

    yaw   = np.array([r["yaw_deg"]       for r in revisits])
    ratio = np.array([r["overlap_ratio"] for r in revisits])
    delta = np.array([r["wp_delta"]      for r in revisits])
    t_ms  = np.array([r["time_ms"]       for r in revisits])
    aniso = np.array([r["anisotropy"]    for r in revisits])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Yaw distribution
    ax = axes[0]
    ax.hist(yaw, bins=20, color="tab:blue", alpha=0.8)
    ax.set_xlabel("yaw (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"Yaw distribution (n={len(yaw)})")
    ax.grid(alpha=0.3)

    # 2. Anisotropy vs. temporal distance
    ax = axes[1]
    ax.scatter(delta, aniso, s=80, alpha=0.8,
               edgecolors="k", linewidths=0.5, c="tab:green")
    ax.set_xlabel("|Δwp|")
    ax.set_ylabel("anisotropy")
    ax.set_title("Anisotropy vs. temporal distance")
    ax.grid(alpha=0.3)

    # 3. Time per revisit
    ax = axes[2]
    ax.hist(t_ms, bins=15, color="tab:orange", alpha=0.8)
    ax.set_xlabel("yaw estimation time (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"Time per revisit (mean={t_ms.mean():.2f} ms)")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Extra plot: yaw comparison if GT is available
    rows_gt = [r for r in revisits if r["yaw_gt_deg"] is not None]
    if rows_gt:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        est = [r["yaw_deg"] for r in rows_gt]
        gt  = [r["yaw_gt_deg"] for r in rows_gt]
        dwp = [r["wp_delta"] for r in rows_gt]
        err = [r["yaw_err_deg"] for r in rows_gt]

        ax = axes2[0]
        sc = ax.scatter(gt, est, c=dwp, cmap="viridis",
                        s=100, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="|Δwp|")
        lims = [-185, 185]
        ax.plot(lims, lims, "r--", alpha=0.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("ground-truth yaw (deg)")
        ax.set_ylabel("estimated yaw (deg)")
        ax.set_title("Estimated vs. GT yaw")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes2[1]
        ax.scatter(dwp, err, s=100, edgecolors="k", linewidths=0.5,
                   color="tab:red")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("|Δwp|")
        ax.set_ylabel("yaw error (est - gt) [deg]")
        ax.set_title("Yaw error vs. temporal distance")
        ax.grid(alpha=0.3)

        plt.tight_layout()

    plt.show()


# ============================================================
# PCA-2D pair visualization
# ============================================================
def _centered_xy(pc, max_radius: Optional[float] = None) -> np.ndarray:
    """Return XY points centered on their centroid, optionally radius-clipped.

    Matches the centering used inside `estimate_yaw_pca`, so the plotted
    principal axes pass through the plotted centroid at the origin.
    """
    xy = pc.points[:, :2].astype(np.float64)
    xy = xy - xy.mean(axis=0)
    if max_radius is not None:
        r = np.linalg.norm(xy, axis=1)
        xy = xy[r <= max_radius]
    return xy


def _draw_principal_axes(
    ax,
    phi_rad: float,
    half_len: float,
    color: str,
    label: Optional[str] = None,
    lw: float = 2.0,
) -> None:
    """Draw the PCA principal axis (solid) and its perpendicular (dashed)."""
    dx, dy = np.cos(phi_rad), np.sin(phi_rad)
    ax.plot([-dx * half_len, dx * half_len],
            [-dy * half_len, dy * half_len],
            color=color, lw=lw, label=label, zorder=5)
    px, py = -np.sin(phi_rad), np.cos(phi_rad)
    ax.plot([-px * half_len, px * half_len],
            [-py * half_len, py * half_len],
            color=color, lw=lw * 0.7, ls="--", alpha=0.6, zorder=5)


def _plot_pca_pair(
    pc_a,
    pc_b,
    record: dict,
    max_radius: Optional[float] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """One figure (1x3) comparing the PCA-2D of a revisit pair.

    Panels:
        1. Cloud A (stored) with its principal axis phi_a.
        2. Cloud B (new) with its principal axis phi_b.
        3. Overlay of both (centroid-centered) with both axes and the
           arc illustrating the angular difference Delta_phi = phi_b - phi_a.
    """
    res = estimate_yaw_pca(pc_a, pc_b, max_radius=max_radius,
                           use_centroid=True)

    xy_a = _centered_xy(pc_a, max_radius=max_radius)
    xy_b = _centered_xy(pc_b, max_radius=max_radius)

    scale = float(max(np.abs(xy_a).max(), np.abs(xy_b).max(), 1e-6)) * 1.15
    phi_a, phi_b = res.phi_a_rad, res.phi_b_rad
    delta_rad = _wrap_pi(phi_b - phi_a)
    dphi_deg = float(np.rad2deg(delta_rad))

    wp_a, wp_b = record["wp_stored"], record["wp_new"]
    title = (f"revisit wp {wp_a} -> {wp_b}  | Δwp={record['wp_delta']}  "
             f"| overlap={record['overlap_ratio']:.3f}  "
             f"| anisotropy={record['anisotropy']:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle(title, fontsize=13)

    # ---- Panel 1: cloud A ----
    ax = axes[0]
    ax.scatter(xy_a[:, 0], xy_a[:, 1], s=2, alpha=0.35, c="tab:blue")
    _draw_principal_axes(ax, phi_a, scale, "tab:blue",
                         label=f"φ_A={np.rad2deg(phi_a):+.1f}°")
    ax.set_title(f"A — stored wp {wp_a}  (N={len(xy_a)})")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- Panel 2: cloud B ----
    ax = axes[1]
    ax.scatter(xy_b[:, 0], xy_b[:, 1], s=2, alpha=0.35, c="tab:orange")
    _draw_principal_axes(ax, phi_b, scale, "tab:orange",
                         label=f"φ_B={np.rad2deg(phi_b):+.1f}°")
    ax.set_title(f"B — new wp {wp_b}  (N={len(xy_b)})")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- Panel 3: overlay + angular difference ----
    ax = axes[2]
    ax.scatter(xy_a[:, 0], xy_a[:, 1], s=2, alpha=0.25,
               c="tab:blue", label=f"A (wp {wp_a})")
    ax.scatter(xy_b[:, 0], xy_b[:, 1], s=2, alpha=0.25,
               c="tab:orange", label=f"B (wp {wp_b})")
    _draw_principal_axes(ax, phi_a, scale, "tab:blue")
    _draw_principal_axes(ax, phi_b, scale, "tab:orange")

    # Arc from phi_a sweeping to phi_b, illustrating the angular difference.
    r_arc = scale * 0.55
    theta = np.linspace(phi_a, phi_a + delta_rad, 64)
    ax.plot(r_arc * np.cos(theta), r_arc * np.sin(theta),
            color="k", lw=1.6, zorder=6)

    r_lab = r_arc * 1.3
    theta_mid = phi_a + delta_rad / 2.0
    ax.annotate(f"Δφ={dphi_deg:+.1f}°",
                xy=(r_lab * np.cos(theta_mid), r_lab * np.sin(theta_mid)),
                fontsize=11, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="black", alpha=0.85), zorder=7)

    ax.set_title("Overlay (centroid-centered) — principal axes")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-scale, scale); ax.set_ylim(-scale, scale)
    ax.legend(fontsize=9, loc="upper right")

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"pca_pair_wp{wp_a:04d}_wp{wp_b:04d}.png"
        fig.savefig(out, dpi=120)
        print(f"[plot-pca] saved {out}")

    plt.show()


def plot_pca_pairs(
    revisits: List[dict],
    data_dir: Path = DATA_DIR,
    max_pairs: Optional[int] = None,
    max_radius: Optional[float] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """Plot the PCA-2D comparison for each revisit pair."""
    if not revisits:
        print("[plot-pca] No revisits to plot.")
        return

    pairs = revisits[:max_pairs] if max_pairs else revisits
    for record in pairs:
        path_a = data_dir / record["wp_stored_name"]
        path_b = data_dir / record["wp_new_name"]
        if not path_a.exists() or not path_b.exists():
            print(f"[plot-pca] Missing cloud(s) for wp "
                  f"{record['wp_stored']} -> {record['wp_new']}; skipping.")
            continue
        pc_a = PointCloud.from_npy(path_a)
        pc_b = PointCloud.from_npy(path_b)
        _plot_pca_pair(pc_a, pc_b, record,
                       max_radius=max_radius, save_dir=save_dir)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run the full place-recognition pipeline with yaw estimation."
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-pca", action="store_true",
                        help="Plot the PCA-2D (principal axes + angular "
                             "difference) for each revisit pair.")
    parser.add_argument("--plot-pca-max", type=int, default=None,
                        help="Limit the number of PCA pair plots "
                             "(default: all revisits).")
    parser.add_argument("--pca-max-radius", type=float, default=None,
                        help="Clip points beyond this XY radius (from the "
                             "centroid) when computing PCA.")
    parser.add_argument("--save-plots", type=Path, default=None,
                        help="Directory to save PCA pair figures as PNG.")
    parser.add_argument("--match-threshold", type=float,
                        default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--temporal-filter", type=int,
                        default=DEFAULT_TEMPORAL_FILTER)
    parser.add_argument("--yaw-tolerance-deg", type=float,
                        default=DEFAULT_YAW_TOLERANCE_DEG)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--trajectory-json", type=Path,
                        default=DATA_DIR / "trajectory_map_ros2.json",
                        help="Path to trajectory_map_ros2.json.")
    parser.add_argument("--odom-noise-deg", type=float, default=0.0,
                        help="Add Gaussian noise to GT yaw to simulate "
                             "odometry drift (default: 0.0).")
    parser.add_argument("--yaw-method",
                        choices=["auto", "pca", "pca+odom",
                                 "simple", "stratified",
                                 "simple+odom", "stratified+odom"],
                        default="auto",
                        help="Yaw estimator: 'simple'/'stratified' = angular "
                             "histograms; the '-odom' variants resolve the "
                             "180-degree ambiguity with odometry; "
                             "'pca'/'pca+odom' = PCA-2D; 'auto' = pca+odom "
                             "if GT poses exist, else stratified.")
    parser.add_argument("--hist-bins", type=int, default=360,
                        help="Angular bins for the histogram methods.")
    parser.add_argument("--hist-rings", type=int, default=10,
                        help="Rings for the stratified histogram method.")
    args = parser.parse_args()

    # ---- 1. Load encoder ----
    print(f"Loading encoder from {CONFIG_PATH}...")
    encoder = load_encoder_from_config(CONFIG_PATH)
    print(f"  SDR size: {encoder.total_size} bits")

    # ---- 2. Load ground-truth / odometry poses ----
    gt_poses: Optional[dict[int, dict]] = None
    if args.trajectory_json.exists():
        print(f"Loading ground-truth poses from {args.trajectory_json}...")
        gt_poses = load_ground_truth_poses(args.trajectory_json)
        print(f"  Loaded {len(gt_poses)} waypoint poses.")

        if args.odom_noise_deg > 0:
            print(f"  Simulating odometry noise: std = "
                  f"{args.odom_noise_deg}°")
            gt_poses = load_odometry_poses(
                args.trajectory_json,
                noise_std_deg=args.odom_noise_deg,
                seed=42,
            )
    else:
        print(f"[warn] No trajectory JSON at {args.trajectory_json}; "
              f"yaw comparison will be skipped, and the pipeline will "
              f"fall back to the stratified histogram.")

    if args.yaw_method.endswith("+odom") and gt_poses is None:
        raise SystemExit(f"[error] --yaw-method {args.yaw_method} requires a "
                         "trajectory JSON (ground-truth/odometry poses).")

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
        yaw_method_opt=args.yaw_method,
        hist_bins=args.hist_bins,
        hist_rings=args.hist_rings,
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

    # ---- 8. PCA-2D pair plots ----
    if args.plot_pca:
        plot_pca_pairs(
            revisits,
            data_dir=DATA_DIR,
            max_pairs=args.plot_pca_max,
            max_radius=args.pca_max_radius,
            save_dir=args.save_plots,
        )


if __name__ == "__main__":
    main()