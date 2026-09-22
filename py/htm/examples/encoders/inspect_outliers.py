# inspect_outliers.py
"""
Inspect the distribution of high-level descriptors to choose rejection
thresholds rationally.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from place_encoder import PointCloud, PlaceDescriptor


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")


def main():
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    print(f"Analyzing {len(npy_paths)} point clouds...\n")

    # Collect descriptors for every cloud
    rows = []
    for path in npy_paths:
        try:
            pc = PointCloud.from_npy(path)
            desc = PlaceDescriptor.from_pointcloud(pc)
            rows.append({
                "name":         path.name,
                "n_points":     pc.n_points,
                "mean_radius":  desc.mean_radius,
                "std_radius":   desc.std_radius,
                "volume":       desc.volume,
                "eigval_1":     desc.eigvals[0],
                "eigval_2":     desc.eigvals[1],
                "eigval_3":     desc.eigvals[2],
                "height":       desc.height,
            })
        except Exception as e:
            print(f"  ! {path.name}: {e}")

    names = [r["name"] for r in rows]
    n_points    = np.array([r["n_points"]    for r in rows])
    mean_radius = np.array([r["mean_radius"] for r in rows])
    volume      = np.array([r["volume"]      for r in rows])
    eigval_1    = np.array([r["eigval_1"]    for r in rows])
    height      = np.array([r["height"]      for r in rows])

    # ---------- Percentile summary ----------
    def pct_line(name, arr):
        p = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
        print(f"{name:<14} | "
              f"p05={p[0]:>8.2f}  p25={p[1]:>8.2f}  p50={p[2]:>8.2f}  "
              f"p75={p[3]:>8.2f}  p90={p[4]:>8.2f}  p95={p[5]:>8.2f}  "
              f"p99={p[6]:>8.2f}  max={arr.max():>8.2f}")

    print("Percentile summary")
    print("-" * 110)
    pct_line("n_points",    n_points)
    pct_line("mean_radius", mean_radius)
    pct_line("volume",      volume)
    pct_line("eigval_1",    eigval_1)
    pct_line("height",      height)
    print()

    # ---------- Rejection simulation ----------
    def simulate(max_mean_radius, max_volume, max_eigval_1, min_points):
        reasons = {
            "too few points": n_points < min_points,
            "mean_radius":    mean_radius > max_mean_radius,
            "volume":         volume > max_volume,
            "eigval_1":       eigval_1 > max_eigval_1,
        }
        rejected_mask = np.zeros(len(rows), dtype=bool)
        counts = {}
        for reason, mask in reasons.items():
            counts[reason] = int((mask & ~rejected_mask).sum())
            rejected_mask |= mask
        return rejected_mask, counts

    # Test a few threshold sets
    threshold_sets = [
        ("current",  20.0, 1000.0, 200.0, 100),
        ("tight",    15.0,  500.0, 100.0, 100),
        ("loose",    30.0, 2000.0, 300.0, 100),
        ("p95-based",
         float(np.percentile(mean_radius, 95)),
         float(np.percentile(volume, 95)),
         float(np.percentile(eigval_1, 95)),
         100),
    ]

    print("Rejection simulation")
    print("-" * 78)
    print(f"{'name':<12} | {'total':>5} | {'few pts':>7} | {'radius':>6} "
          f"| {'volume':>6} | {'eig1':>5} | {'rejected':>8}")
    for label, mr, mv, me, mp in threshold_sets:
        mask, counts = simulate(mr, mv, me, mp)
        print(f"{label:<12} | {len(rows):>5} | "
              f"{counts['too few points']:>7} | {counts['mean_radius']:>6} | "
              f"{counts['volume']:>6} | {counts['eigval_1']:>5} | "
              f"{mask.sum():>8} "
              f"({100 * mask.sum() / len(rows):.1f} %)")
    print()

    # ---------- Scatter plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(volume, mean_radius, s=20, alpha=0.6)
    ax.axhline(20.0, color='r', linestyle='--', label='mean_radius = 20')
    ax.axvline(1000.0, color='b', linestyle='--', label='volume = 1000')
    ax.set_xlabel("volume"); ax.set_ylabel("mean_radius")
    ax.set_title("volume × mean_radius")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(eigval_1, mean_radius, s=20, alpha=0.6)
    ax.axhline(20.0, color='r', linestyle='--', label='mean_radius = 20')
    ax.axvline(200.0, color='b', linestyle='--', label='eigval_1 = 200')
    ax.set_xlabel("eigval_1"); ax.set_ylabel("mean_radius")
    ax.set_title("eigval_1 × mean_radius")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(n_points, bins=30, alpha=0.7)
    ax.axvline(100, color='r', linestyle='--', label='min_points = 100')
    ax.set_xlabel("n_points"); ax.set_ylabel("count")
    ax.set_title("n_points distribution")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------- Detail of rejected clouds ----------
    mask, _ = simulate(20.0, 1000.0, 200.0, 100)
    rejected_idx = np.where(mask)[0]
    print(f"Clouds that would be rejected with current thresholds "
          f"({len(rejected_idx)}):")
    print(f"{'filename':<55} | {'N':>6} | {'radius':>7} | "
          f"{'volume':>8} | {'eig1':>7}")
    print("-" * 100)
    for i in rejected_idx:
        r = rows[i]
        print(f"{r['name']:<55} | {r['n_points']:>6d} | "
              f"{r['mean_radius']:>7.2f} | {r['volume']:>8.2f} | "
              f"{r['eigval_1']:>7.2f}")


if __name__ == "__main__":
    main()