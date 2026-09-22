# test_place_database_full.py
"""
Run the entire dataset through the PlaceDatabase (no synthetic rotations).

The dataset is a chronological sequence of waypoints (wp_0000 ... wp_0104).
Consecutive waypoints often look alike (same place), so we expect a mix of
matches and new-place creations. This script reports per-waypoint behavior
and global statistics.
"""

import sys
from pathlib import Path
import time
import numpy as np

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase


# ---------- Config ----------
DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")
DB_PREFIX = "test_db/dataset_full"

MATCH_THRESHOLD = 0.85      # min overlap ratio to consider same place
YAW_TOLERANCE_DEG = 15.0   # min angular separation to create a new template


def encode_path(encoder, npy_path: Path) -> np.ndarray:
    """Convenience: load a cloud, compute descriptors, encode to SDR."""
    pc = PointCloud.from_npy(npy_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    return encoder.encode(desc.to_vector())


def main():
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
        match_threshold=MATCH_THRESHOLD,
        yaw_tolerance_deg=YAW_TOLERANCE_DEG,
    )

    # ---- 4. Feed dataset in chronological order ----
    # Since there is no yaw estimation yet, we use a placeholder yaw = 0.0
    # for every observation. This is fine: the yaw only affects the number
    # of templates per place, not the place-matching logic.
    print(f"{'wp':>5} | {'N pts':>7} | {'active':>7} | {'overlap':>8} "
          f"| {'ratio':>6} | {'place':>5} | {'match':>5} | {'templ':>5}")
    print("-" * 72)

    t0 = time.time()
    for i, npy_path in enumerate(npy_paths):
        sdr = encode_path(encoder, npy_path)
        n_active = int(sdr.sum())

        # We need to know the best overlap before calling match_or_create
        # to display it. Compute it here for diagnostics.
        best_ratio = 0.0
        if db.places:
            for place in db.places:
                overlap = int(np.logical_and(sdr, place.sdr).sum())
                ratio = overlap / n_active
                if ratio > best_ratio:
                    best_ratio = ratio

        # yaw placeholder = 0.0
        place_id, matched, template_idx = db.match_or_create(
            sdr, yaw_rad=0.0, label=npy_path.name,
        )

        # Extract waypoint tag from filename
        wp_tag = npy_path.name.split("_")[1]  # e.g. "0000"
        print(f"{wp_tag:>5} | {PointCloud.from_npy(npy_path).n_points:>7d} | "
              f"{n_active:>7d} | {best_ratio * n_active:>8.0f} "
              f"| {best_ratio:>6.3f} | {place_id:>5d} | "
              f"{str(matched):>5} | {template_idx:>5d}")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(npy_paths)} clouds in {elapsed:.2f}s "
          f"({1000 * elapsed / len(npy_paths):.1f} ms/cloud)\n")

    # ---- 5. Summary ----
    print("=== Database summary ===")
    print(db.summary())

    # ---- 6. Save ----
    db.save(DB_PREFIX)
    print(f"\nDatabase saved to: {DB_PREFIX}_meta.json / {DB_PREFIX}_sdrs.npy")


if __name__ == "__main__":
    main()