# test_place_database_revisit.py
"""
Test the PlaceDatabase's template accumulation by forcing revisits.

Strategy:
    1. Insert every 10th waypoint (learning phase).
    2. Insert the remaining waypoints (revisit phase).
    3. Re-insert the first 5 waypoints with different fake yaws to
       demonstrate template accumulation.
"""
import sys
from pathlib import Path
import numpy as np

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase, angular_distance


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")

MATCH_THRESHOLD = 0.5
YAW_TOLERANCE_DEG = 15.0


def encode_path(encoder, npy_path):
    pc = PointCloud.from_npy(npy_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    return encoder.encode(desc.to_vector())


def main():
    encoder = load_encoder_from_config(CONFIG_PATH)
    npy_paths = sorted(DATA_DIR.glob("*.npy"))

    print(f"Encoder: {encoder.total_size} bits")
    print(f"Dataset: {len(npy_paths)} clouds\n")

    db = PlaceDatabase(
        sdr_size=encoder.total_size,
        match_threshold=MATCH_THRESHOLD,
        yaw_tolerance_deg=YAW_TOLERANCE_DEG,
    )

    # ---- Phase 1: learning (every 10th waypoint) ----
    print("=== Phase 1: learning (every 10th waypoint) ===")
    learning_paths = npy_paths[::10]
    for npy_path in learning_paths:
        sdr = encode_path(encoder, npy_path)
        pid, matched, _ = db.match_or_create(sdr, yaw_rad=0.0)
        tag = npy_path.name.split("_")[1]
        print(f"  wp_{tag}: place={pid}, matched={matched}")

    n_places_after_learning = len(db.places)
    print(f"\n  → Places after learning: {n_places_after_learning}\n")

    # ---- Phase 2: revisit (remaining waypoints) ----
    print("=== Phase 2: revisit (remaining waypoints) ===")
    learning_set = set(p.name for p in learning_paths)
    revisit_paths = [p for p in npy_paths if p.name not in learning_set]

    n_matched = 0
    n_new = 0
    for npy_path in revisit_paths:
        sdr = encode_path(encoder, npy_path)
        pid, matched, _ = db.match_or_create(sdr, yaw_rad=0.0)
        if matched:
            n_matched += 1
        else:
            n_new += 1

    print(f"  Revisited clouds: {len(revisit_paths)}")
    print(f"    matched to existing place : {n_matched} "
          f"({100 * n_matched / len(revisit_paths):.1f}%)")
    print(f"    created new place         : {n_new} "
          f"({100 * n_new / len(revisit_paths):.1f}%)")
    print(f"\n  → Total places now: {len(db.places)}\n")

    # ---- Phase 3: template accumulation demo ----
    print("=== Phase 3: template accumulation on same waypoint ===")
    # Pick the first waypoint and revisit it with different fake yaws
    demo_path = npy_paths[0]
    demo_sdr = encode_path(encoder, demo_path)

    # Fake yaws to simulate revisiting the same place from different angles
    fake_yaws_deg = [0, 10, 45, 50, 90, 180, 185, 270]
    print(f"Reinserting {demo_path.name} with fake yaws:")
    for yaw_deg in fake_yaws_deg:
        yaw_rad = np.deg2rad(yaw_deg)
        pid, matched, tidx = db.match_or_create(demo_sdr, yaw_rad=yaw_rad)
        print(f"  yaw={yaw_deg:>4}° → place={pid}, matched={matched}, "
              f"template={tidx}")

    place = db.get_place(0)
    if place is not None:
        print(f"\nPlace 0 templates:")
        for t in place.angular_templates:
            print(f"  yaw={np.rad2deg(t.mean_yaw_rad):>+7.1f}°, "
                  f"visits={t.visit_count}")

    # ---- Final summary ----
    print("\n=== Final database summary ===")
    print(db.summary())


if __name__ == "__main__":
    main()