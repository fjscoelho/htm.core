# test_loop_closure.py
"""
End-to-end demo of pipeline blocks 4 + 5 (HTM context + loop closure).

For every point cloud in chronological order it:
    1. encodes the cloud to an SDR (blocks 2-3);
    2. presents the SDR to the PlaceDatabase, which advances the HTM and
       stores the winner cells as temporal context (block 4);
    3. if the place was visited long ago (loop closure), estimates the
       relative yaw, adds it to the place's angular templates and relates it
       to the temporal context (block 5);
    4. asks the context which stored angle is the correct one.

Outputs
-------
    loop_closure_events.csv     - one row per observation
    test_db/loop_closure_*      - the database + HTM snapshot

Usage
-----
    python test_loop_closure.py
    python test_loop_closure.py --yaw-method stratified --no-verbose
    python test_loop_closure.py --match-threshold 0.80 --temporal-filter 5
    python test_loop_closure.py --odom-noise-deg 15 --plot
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))
sys.path.append(str(Path(__file__).resolve().parent))  # for the pose loaders

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase
from src.loop_closure import LoopClosureManager

# Reuse the odometry/GT loaders already used by the yaw pipeline.
from test_pipeline_revisits import (
    load_ground_truth_poses,
    load_odometry_poses,
)


# ---------- Config ----------
DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")
OUTPUT_CSV = Path("loop_closure_events.csv")
DB_PREFIX = "test_db/loop_closure"

DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_TEMPORAL_FILTER = 5
DEFAULT_YAW_TOLERANCE_DEG = 15.0


# ============================================================
# Helpers
# ============================================================
def encode_path(encoder, npy_path: Path) -> np.ndarray:
    pc = PointCloud.from_npy(npy_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    return encoder.encode(desc.to_vector())


def wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def run_localize_demo(encoder, npy_paths, loop_events, db_prefix) -> None:
    """
    Reload the saved database and re-localize the loop-closure waypoints
    *without learning*, to show that the HTM context selects the stored angle.
    """
    print("\n" + "=" * 70)
    print("LOCALIZATION (read-only, context -> angle)")
    print("=" * 70)

    db = PlaceDatabase.load(db_prefix)
    print(f"Reloaded DB: {len(db.places)} places, "
          f"context={'yes' if db.context_memory is not None else 'no'}")

    mgr = LoopClosureManager(db, yaw_method="stratified")

    print(f"{'wp':>5} | {'matched':>7} | {'place':>5} | {'ratio':>6} | "
          f"{'ctxT':>4} | {'sim':>5} | {'angle°':>8} | {'map°(ref)':>9}")
    print("-" * 72)
    ok = 0
    for ev in loop_events:
        npy_path = npy_paths[ev.wp_new]
        sdr = encode_path(encoder, npy_path)
        res = mgr.localize(sdr)   # learn=False by default
        ref_angle = None
        place = db.get_place(res["place_id"]) if res["matched"] else None
        if place is not None and ev.template_idx >= 0 and ev.template_idx < place.n_templates:
            ref_angle = float(np.rad2deg(place.angular_templates[ev.template_idx].mean_yaw_rad))
        if res["context_template_idx"] == ev.template_idx:
            ok += 1
        angle_str = (f"{res['angle_deg']:+.2f}"
                     if res["angle_deg"] is not None else "n/a")
        ref_str = f"{ref_angle:+.2f}" if ref_angle is not None else "n/a"
        print(f"{ev.wp_new:>5d} | {str(res['matched']):>7} | {res['place_id']:>5d} | "
              f"{res['overlap_ratio']:>6.3f} | {res['context_template_idx']:>4d} | "
              f"{res['context_similarity']:>5.2f} | {angle_str:>8} | {ref_str:>9}")
    print("-" * 72)
    print(f"Context picked the same template as mapping in {ok}/{len(loop_events)} loop closures.")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Blocks 4+5: HTM temporal context + loop closure."
    )
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--temporal-filter", type=int, default=DEFAULT_TEMPORAL_FILTER)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=DEFAULT_YAW_TOLERANCE_DEG)
    parser.add_argument("--yaw-method", choices=["pca_odom", "stratified"],
                        default="pca_odom")
    parser.add_argument("--odom-noise-deg", type=float, default=0.0)
    parser.add_argument("--trajectory-json", type=Path,
                        default=DATA_DIR / "trajectory_map_ros2.json")
    parser.add_argument("--cells-per-column", type=int, default=8)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--localize", action="store_true",
                        help="After mapping, reload the DB and re-localize the "
                             "loop-closure waypoints (read-only, no learning).")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # ---- 1. Encoder ----
    print(f"Loading encoder from {CONFIG_PATH}...")
    encoder = load_encoder_from_config(CONFIG_PATH)
    print(f"  SDR size: {encoder.total_size} bits")

    # ---- 2. Odometry / GT poses ----
    gt_poses: Optional[dict] = None
    yaw_method = args.yaw_method
    if args.trajectory_json.exists():
        gt_poses = load_ground_truth_poses(args.trajectory_json)
        if args.odom_noise_deg > 0:
            print(f"  Simulating odometry noise std={args.odom_noise_deg}°")
            gt_poses = load_odometry_poses(
                args.trajectory_json,
                noise_std_deg=args.odom_noise_deg,
                seed=42,
            )
        print(f"  Loaded {len(gt_poses)} waypoint poses (yaw method: {yaw_method}).")
    else:
        print(f"[warn] no trajectory JSON at {args.trajectory_json}; "
              f"falling back to yaw method 'stratified'.")
        yaw_method = "stratified"

    # ---- 3. Dataset ----
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    print(f"Found {len(npy_paths)} point clouds\n")
    if not npy_paths:
        raise SystemExit("No .npy files found. Check DATA_DIR.")

    # ---- 4. Database + manager (block 4 HTM lives in the DB) ----
    db = PlaceDatabase(
        sdr_size=encoder.total_size,
        match_threshold=args.match_threshold,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
        enable_context=True,
        cells_per_column=args.cells_per_column,
    )
    print(f"HTM context: {db.context_memory}")

    lc = LoopClosureManager(
        db,
        temporal_filter=args.temporal_filter,
        yaw_method=yaw_method,
    )

    # ---- 5. Run the pipeline ----
    verbose = not args.no_verbose
    if verbose:
        print(f"\n{'wp':>5} | {'ratio':>6} | {'place':>5} | {'match':>5} "
              f"| {'Δwp':>4} | {'loop':>4} | {'anom':>5} | {'#win':>5} "
              f"| {'yaw°':>8} | {'ctxT':>4} | {'ctxS':>5} | decision")
        print("-" * 108)

    for wp_idx, npy_path in enumerate(npy_paths):
        pc = PointCloud.from_npy(npy_path)
        sdr = encode_path(encoder, npy_path)
        odom_yaw = gt_poses[wp_idx]["yaw_rad"] if gt_poses and wp_idx in gt_poses else None

        ev = lc.observe(sdr, pc=pc, wp_index=wp_idx, odom_yaw_rad=odom_yaw,
                        label=npy_path.name)

        if verbose:
            delta = "-" if ev.wp_delta is None else str(ev.wp_delta)
            decision = ("new place" if not ev.matched
                        else ("loop closure" if ev.is_loop_closure else "recent"))
            wp_tag = npy_path.name.split("_")[1]
            print(f"{wp_tag:>5} | {ev.overlap_ratio:>6.3f} | {ev.place_id:>5d} | "
                  f"{str(ev.matched):>5} | {delta:>4} | "
                  f"{str(ev.is_loop_closure):>4} | {ev.anomaly:>5.2f} | "
                  f"{ev.n_winner_cells:>5d} | {ev.yaw_deg:>+8.2f} | "
                  f"{ev.context_template_idx:>4d} | {ev.context_similarity:>5.2f} | "
                  f"{decision}")

    # ---- 6. Summary ----
    print("\n" + "=" * 70)
    print("BLOCK 4 + 5 SUMMARY")
    print("=" * 70)
    print(lc.summary())

    n_candidates = sum(1 for e in lc.events if e.matched and e.wp_delta is not None
                       and e.wp_delta > args.temporal_filter)
    print(f"\nLoop-closure candidates: {n_candidates}")
    print(f"Loop closures processed: {lc.n_loop_closures}")

    loops = [e for e in lc.events if e.is_loop_closure]
    if loops:
        yaws = np.array([e.yaw_deg for e in loops])
        anom = np.array([e.anomaly for e in loops])
        print(f"\nYaw estimates (deg): mean={yaws.mean():+.2f}, "
              f"std={yaws.std():.2f}, min={yaws.min():+.2f}, max={yaws.max():+.2f}")
        print(f"Anomaly at loop closures: mean={anom.mean():.3f}")

    # ---- 7. How well does the CONTEXT indicate the angle? ----
    print("\n" + "=" * 70)
    print("CONTEXT -> ANGLE CHECK")
    print("=" * 70)
    checks = []
    for e in loops:
        if e.context_template_idx < 0 or e.template_idx < 0:
            continue
        place = db.get_place(e.place_id)
        if place is None:
            continue
        ctx_t = place.angular_templates[e.context_template_idx]
        yaw_t = place.angular_templates[e.template_idx]
        dyaw = wrap_angle_deg(np.rad2deg(ctx_t.mean_yaw_rad) - np.rad2deg(yaw_t.mean_yaw_rad))
        checks.append((e, np.rad2deg(ctx_t.mean_yaw_rad), np.rad2deg(yaw_t.mean_yaw_rad),
                       dyaw, e.context_similarity))
    if checks:
        err = np.array([abs(c[3]) for c in checks])
        print(f"{'wp':>5} | {'ctx_yaw°':>9} | {'yaw_templ°':>10} | {'Δ°':>7} | {'sim':>5}")
        print("-" * 50)
        for e, ay, by, dy, sim in checks:
            print(f"{e.wp_new:>5d} | {ay:>+9.2f} | {by:>+10.2f} | {dy:>+7.2f} | {sim:>5.2f}")
        print("-" * 50)
        print(f"|Δ| mean={err.mean():.2f}°, median={np.median(err):.2f}°, max={err.max():.2f}°")
    else:
        print("No loop closure produced a context-resolved angle.")

    # ---- 8. Persist ----
    lc.save_events_csv(OUTPUT_CSV)
    print(f"\n[CSV] {len(lc.events)} events -> {OUTPUT_CSV.resolve()}")

    db.save(DB_PREFIX)
    print(f"[DB ] saved to {Path(DB_PREFIX).resolve()}_* "
          f"(+ _tm.bin, _context_log.json)")

    # ---- 9. Read-only localization demo ----
    if args.localize and loops:
        run_localize_demo(encoder, npy_paths, loops, DB_PREFIX)

    # ---- 10. Optional plot ----
    if args.plot:
        import matplotlib.pyplot as plt

        it = np.arange(len(lc.events))
        anomaly = np.array([e.anomaly for e in lc.events])
        winner = np.array([e.n_winner_cells for e in lc.events])
        place_ids = np.array([e.place_id for e in lc.events])

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(it, place_ids, ".", ms=4)
        axes[0].set_ylabel("place id")
        axes[0].set_title("Place/Database over time")
        axes[1].plot(it, anomaly, "-o", ms=3, color="tab:red")
        axes[1].set_ylabel("HTM anomaly")
        axes[2].plot(it, winner, "-o", ms=3, color="tab:blue")
        axes[2].set_ylabel("# winner cells")
        axes[2].set_xlabel("iteration (waypoint)")
        for ax in axes:
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
