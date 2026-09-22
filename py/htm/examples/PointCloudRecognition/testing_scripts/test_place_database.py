# test_place_database.py
"""
Test PlaceDatabase with a synthetic stream:
    - Visit place A from 3 different yaws.
    - Visit place B from 2 yaws.
    - Revisit place A from a new yaw (should match A, not create new place).
"""

import sys
from pathlib import Path
import numpy as np

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config
from src.place_database import PlaceDatabase


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
CONFIG_PATH = Path(root_project / "src/encoder_config.json")


def encode_cloud(encoder, npy_path):
    pc = PointCloud.from_npy(npy_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    return encoder.encode(desc.to_vector())


def main():
    encoder = load_encoder_from_config(CONFIG_PATH)

    # Pick two clouds from different places in the dataset
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    cloud_a = npy_paths[0]
    cloud_b = npy_paths[1]

    print(f"Place A source: {cloud_a.name}")
    print(f"Place B source: {cloud_b.name}\n")

    db = PlaceDatabase(
        sdr_size=encoder.total_size,
        match_threshold=0.85,
        yaw_tolerance_deg=15.0,
    )

    # Simulated yaws (radians). We fake them because the encoder is yaw-invariant;
    # in a real system these would come from the yaw estimator.
    yaws_a = [0.0, np.pi / 2, np.pi]        # place A seen from 3 yaws
    yaws_b = [0.0, np.pi / 4]               # place B seen from 2 yaws
    yaws_a_new = [np.pi / 8]                # place A seen from a new yaw (close to 0)

    print("=== Inserting observations ===")
    for y in yaws_a:
        sdr = encode_cloud(encoder, cloud_a)
        pid, matched, tidx = db.match_or_create(sdr, yaw_rad=y)
        print(f"A @ yaw={np.rad2deg(y):+6.1f}° → place={pid}, "
              f"matched={matched}, template={tidx}")

    for y in yaws_b:
        sdr = encode_cloud(encoder, cloud_b)
        pid, matched, tidx = db.match_or_create(sdr, yaw_rad=y)
        print(f"B @ yaw={np.rad2deg(y):+6.1f}° → place={pid}, "
              f"matched={matched}, template={tidx}")

    print("\n=== Revisiting place A with a new yaw ===")
    for y in yaws_a_new:
        sdr = encode_cloud(encoder, cloud_a)
        pid, matched, tidx = db.match_or_create(sdr, yaw_rad=y)
        print(f"A @ yaw={np.rad2deg(y):+6.1f}° → place={pid}, "
              f"matched={matched}, template={tidx}")

    print("\n=== Database summary ===")
    print(db.summary())

    # Save and reload
    db.save("test_db/places")
    print("\nSaved. Reloading...")
    db2 = PlaceDatabase.load("test_db/places")
    print(db2.summary())


if __name__ == "__main__":
    main()