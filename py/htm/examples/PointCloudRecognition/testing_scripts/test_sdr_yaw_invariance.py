# test_sdr_yaw_invariance.py
"""
Test that rotating a point cloud around Z produces SDRs with high overlap.

The encoder is yaw-invariant by design (its features are invariant to yaw),
so rotating the input cloud should produce an SDR that overlaps the
original in ~ all active bits.
"""

import sys
from pathlib import Path
import numpy as np

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor
from src.calibrate_encoder import load_encoder_from_config


def sdr_overlap(sdr_a: np.ndarray, sdr_b: np.ndarray) -> int:
    """Number of bits that are 1 in both SDRs."""
    return int(np.logical_and(sdr_a, sdr_b).sum())


def test_yaw_invariance_sdr(
    encoder,
    pc: PointCloud,
    angles_deg=(0, 15, 30, 45, 60, 90, 120, 135, 180, 270),
    verbose: bool = True,
):
    """
    Rotate a point cloud around Z and check SDR overlap with the original.

    Expected: overlap should be close to the number of active bits (588),
    because the descriptors are yaw-invariant by construction.
    """
    desc_ref = PlaceDescriptor.from_pointcloud(pc)
    sdr_ref = encoder.encode(desc_ref.to_vector())
    n_active = int(sdr_ref.sum())

    print(f"Reference cloud: {pc.n_points} points, "
          f"SDR active bits = {n_active}")
    print()
    print(f"{'angle':>8} | {'overlap':>8} | {'overlap %':>10} | "
          f"{'SDR active':>10} | {'Δeigvals':>12}")
    print("-" * 65)

    results = []
    for ang in angles_deg:
        pc_rot = pc.rotated_yaw(np.deg2rad(ang))
        desc = PlaceDescriptor.from_pointcloud(pc_rot)
        sdr = encoder.encode(desc.to_vector())

        overlap = sdr_overlap(sdr_ref, sdr)
        overlap_pct = 100.0 * overlap / n_active
        n_active_rot = int(sdr.sum())
        deig = float(np.abs(desc.eigvals - desc_ref.eigvals).max())

        results.append((ang, overlap, overlap_pct, n_active_rot))
        print(f"{ang:>7}° | {overlap:>8d} | {overlap_pct:>9.2f} % | "
              f"{n_active_rot:>10d} | {deig:>12.2e}")

    # Summary
    overlaps = [r[1] for r in results]
    print()
    print(f"Overlap min:  {min(overlaps)} / {n_active} "
          f"({100 * min(overlaps) / n_active:.1f} %)")
    print(f"Overlap mean: {np.mean(overlaps):.1f} / {n_active} "
          f"({100 * np.mean(overlaps) / n_active:.1f} %)")
    print(f"Overlap max:  {max(overlaps)} / {n_active} "
          f"({100 * max(overlaps) / n_active:.1f} %)")

    return results


def test_place_discrimination(
    encoder,
    pc_a: PointCloud,
    pc_b: PointCloud,
    label_a: str = "A",
    label_b: str = "B",
):
    """
    Encode two different clouds and report overlap.

    Expected: overlap should be much lower than the yaw-invariance overlap,
    showing the encoder discriminates different places.
    """
    sdr_a = encoder.encode(PlaceDescriptor.from_pointcloud(pc_a).to_vector())
    sdr_b = encoder.encode(PlaceDescriptor.from_pointcloud(pc_b).to_vector())

    n_a, n_b = int(sdr_a.sum()), int(sdr_b.sum())
    overlap = sdr_overlap(sdr_a, sdr_b)
    union = int(np.logical_or(sdr_a, sdr_b).sum())
    jaccard = overlap / union if union > 0 else 0.0

    print(f"\nPlace discrimination:")
    print(f"  {label_a}: active={n_a}")
    print(f"  {label_b}: active={n_b}")
    print(f"  Overlap      : {overlap}")
    print(f"  Union        : {union}")
    print(f"  Jaccard index: {jaccard:.3f}  (lower = more distinct)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
    CONFIG_PATH = Path(root_project / "src/encoder_config.json")

    # 1. Load calibrated encoder
    print(f"Loading encoder from {CONFIG_PATH}...")
    encoder = load_encoder_from_config(CONFIG_PATH)
    print(f"  Total SDR size: {encoder.total_size} bits")

    # 2. Pick a sample cloud
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    if not npy_paths:
        raise SystemExit(f"No .npy files found in {DATA_DIR}")

    sample_path = npy_paths[0]
    print(f"\nLoading sample: {sample_path.name}")
    pc = PointCloud.from_npy(sample_path)

    # 3. Yaw invariance test
    print("\n=== Yaw invariance in SDR space ===")
    test_yaw_invariance_sdr(encoder, pc)

    # 4. Discrimination test (optional): compare with a different cloud
    if len(npy_paths) >= 2:
        other_path = npy_paths[len(npy_paths) // 2]  # middle of dataset
        print(f"\nLoading different sample: {other_path.name}")
        pc_other = PointCloud.from_npy(other_path)

        print("\n=== Place discrimination ===")
        test_place_discrimination(
            encoder, pc, pc_other,
            label_a=sample_path.name[:30],
            label_b=other_path.name[:30],
        )