# test_place_database_full.py
"""
Run the entire dataset through the PlaceDatabase (no synthetic rotations).

The dataset is a chronological sequence of waypoints (wp_0000 ... wp_0104).
Consecutive waypoints often look alike (same place), so we expect a mix of
matches and new-place creations. This script reports per-waypoint behavior
and global statistics.

The `matched_wp` column always shows the waypoint that produced the best
overlap, even when the match was rejected by the threshold (useful to see
"almost matches").

Usage
-----
    python test_place_database_full.py
    python test_place_database_full.py --plot-matches
    python test_place_database_full.py --plot-matches --max-plots 5
"""

import sys
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent  # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase


# ---------- Config ----------
DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")
DB_PREFIX = "test_db/dataset_full"

MATCH_THRESHOLD = 0.85      # min overlap ratio to consider same place
YAW_TOLERANCE_DEG = 15.0    # min angular separation to create a new template


# ---------- Helpers ----------
def encode_path(encoder, npy_path: Path) -> np.ndarray:
    """Convenience: load a cloud, compute descriptors, encode to SDR."""
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


def plot_match_pair(
    npy_new: Path,
    npy_stored: Path,
    overlap_ratio: float,
    place_id: int,
    wp_new: str,
    wp_stored: str,
):
    """Plot the two point clouds (top view) side by side."""
    pc_new = PointCloud.from_npy(npy_new)
    pc_stored = PointCloud.from_npy(npy_stored)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    pc_stored.plot_top_view(ax=axes[0])
    axes[0].set_title(f"STORED  (place {place_id}, wp {wp_stored})\n"
                      f"N = {pc_stored.n_points}")

    pc_new.plot_top_view(ax=axes[1])
    axes[1].set_title(f"NEW QUERY  (wp {wp_new})\n"
                      f"N = {pc_new.n_points}")

    fig.suptitle(f"Match overlap ratio = {overlap_ratio:.3f}  "
                 f"(place {place_id})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Run the dataset through PlaceDatabase, optionally plotting matches."
    )
    parser.add_argument(
        "--plot-matches", action="store_true",
        help="Plot each matching pair of point clouds for visual inspection."
    )
    parser.add_argument(
        "--max-plots", type=int, default=10,
        help="Maximum number of match pairs to plot (default: 10)."
    )
    parser.add_argument(
        "--match-threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Minimum overlap ratio to consider same place "
             f"(default: {MATCH_THRESHOLD})."
    )
    args = parser.parse_args()

    # ---- 1. Load encoder ----
    print(f"Loading encoder from {CONFIG_PATH}...")
    encoder = load_encoder_from_config(CONFIG_PATH)
    print(f"  SDR size: {encoder.total_size} bits\n")

    # ---- 2. Discover dataset ----
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    print(f"Found {len(npy_paths)} point clouds\n")

    # ---- 3. Initialize database ----
    db = PlaceDatabase(
        sdr_size=encoder.total_size,
        match_threshold=args.match_threshold,
        yaw_tolerance_deg=YAW_TOLERANCE_DEG,
    )

    # Track the source waypoint index and file per place_id
    place_source_wp_idx: dict[int, int] = {}     # place_id -> wp_idx
    place_source_file: dict[int, Path] = {}      # place_id -> file path

    # Collect matches for later plotting
    match_records: list[dict] = []

    # ---- 4. Feed dataset in chronological order ----
    print(f"{'wp':>5} | {'N pts':>7} | {'active':>7} | {'overlap':>8} "
          f"| {'ratio':>6} | {'place':>5} | {'match':>5} | {'templ':>5} "
          f"| {'best_wp':>8}")
    print("-" * 90)

    t0 = time.time()
    for wp_idx, npy_path in enumerate(npy_paths):
        sdr = encode_path(encoder, npy_path)
        n_active = int(sdr.sum())
        pc = PointCloud.from_npy(npy_path)

        # ---- Diagnostic: best match before committing ----
        best_idx, best_overlap, best_ratio = find_best_match(sdr, db)

        # Determine the wp that produced the best overlap.
        # This works even if best_idx == -1 (empty database).
        best_wp_idx: int | None = None
        stored_npy_path: Path | None = None
        if best_idx >= 0:
            best_place = db.places[best_idx]
            best_wp_idx = place_source_wp_idx.get(best_place.place_id)
            stored_npy_path = place_source_file.get(best_place.place_id)

        # Determine whether this will be a match
        will_match = (
            best_idx >= 0
            and best_ratio >= args.match_threshold
        )

        # ---- Commit to the database ----
        place_id, matched, template_idx = db.match_or_create(
            sdr, yaw_rad=0.0, label=npy_path.name,
        )

        # If this was a new place, remember which waypoint and file created it
        if not matched:
            place_source_wp_idx[place_id] = wp_idx
            place_source_file[place_id] = npy_path

        # ---- Record match info for later plotting ----
        # Only record actual matches (matched=True) as candidates for plotting.
        if matched and stored_npy_path is not None:
            match_records.append({
                "place_id":       place_id,
                "wp_new":         npy_path.name.split("_")[1],
                "wp_stored":      stored_npy_path.name.split("_")[1],
                "npy_new":        npy_path,
                "npy_stored":     stored_npy_path,
                "overlap_ratio":  best_ratio,
            })

        # ---- Format the "best_wp" column ----
        # Always shows the wp that produced the best overlap, even if the
        # match was rejected by the threshold.
        if best_wp_idx is not None:
            best_wp_str = f"wp_{best_wp_idx:04d}"
        else:
            best_wp_str = "-"

        # Current waypoint tag
        wp_tag = npy_path.name.split("_")[1]  # e.g. "0000"

        print(f"{wp_tag:>5} | {pc.n_points:>7d} | {n_active:>7d} | "
              f"{best_overlap:>8d} | {best_ratio:>6.3f} | "
              f"{place_id:>5d} | {str(matched):>5} | {template_idx:>5d} | "
              f"{best_wp_str:>8}")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(npy_paths)} clouds in {elapsed:.2f}s "
          f"({1000 * elapsed / len(npy_paths):.1f} ms/cloud)\n")

    # ---- 5. Summary ----
    print("=== Database summary ===")
    print(db.summary())

    # ---- 6. Save ----
    db.save(DB_PREFIX)
    print(f"\nDatabase saved to: {DB_PREFIX}_meta.json / {DB_PREFIX}_sdrs.npy")

    # ---- 7. Plot matches if requested ----
    if args.plot_matches:
        if not match_records:
            print("\n[plot] No matches were recorded, nothing to plot.")
            return

        n_to_plot = min(len(match_records), args.max_plots)
        print(f"\n[plot] Plotting {n_to_plot} of {len(match_records)} matches "
              f"(use --max-plots to change).")

        match_records_sorted = sorted(
            match_records, key=lambda r: -r["overlap_ratio"]
        )

        for i, rec in enumerate(match_records_sorted[:n_to_plot]):
            print(f"  [{i + 1}/{n_to_plot}] "
                  f"wp_{rec['wp_new']} ↔ wp_{rec['wp_stored']} "
                  f"(place {rec['place_id']}, "
                  f"ratio={rec['overlap_ratio']:.3f})")
            plot_match_pair(
                npy_new=rec["npy_new"],
                npy_stored=rec["npy_stored"],
                overlap_ratio=rec["overlap_ratio"],
                place_id=rec["place_id"],
                wp_new=rec["wp_new"],
                wp_stored=rec["wp_stored"],
            )

        print("\n[plot] Close the plot windows to finish.")
        plt.show()


if __name__ == "__main__":
    main()