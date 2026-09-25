# test_yaw_estimator.py
"""
Test the yaw estimator with synthetic rotations.

Takes a point cloud, rotates it by a known angle, and verifies that the
estimator recovers the rotation within a tolerance. Also measures and
plots execution time per method.
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

import sys
root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator import estimate_yaw


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")


# ---------- Timing helpers ----------
def time_call(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) once and return (result, elapsed_seconds).

    Uses time.perf_counter for high-resolution wall-clock timing.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def wrap_err_deg(est_deg: float, true_deg: float) -> float:
    """Wrap the angular error into (-180, 180]."""
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0


# ---------- Main ----------
def main():
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    sample = npy_paths[0]

    print(f"Sample: {sample.name}")
    pc = PointCloud.from_npy(sample)
    print(f"Points: {pc.n_points}\n")

    angles_deg = [15, 30, 45, 90, 135, 180, -45, -90, -135]

    # Track results for the plots
    results = {
        "angle_deg": [],
        "err_hist": [],
        "err_icp": [],
        "conf": [],
        "ambig": [],
        "t_hist": [],
        "t_icp": [],
    }

    print(f"{'true':>8} | {'est':>8} | {'error':>8} | {'conf':>8} "
          f"| {'ambig':>6} | {'t_hist':>9} | {'t_icp':>9} | {'method':>15}")
    print("-" * 100)

    for true_deg in angles_deg:
        true_rad = np.deg2rad(true_deg)
        pc_rot = pc.rotated_yaw(true_rad)

        # --- Histogram only ---
        res_hist, t_hist = time_call(
            estimate_yaw, pc, pc_rot, n_bins=360, refine_icp=False
        )

        # --- Histogram + ICP ---
        res_icp, t_icp = time_call(
            estimate_yaw, pc, pc_rot, n_bins=360, refine_icp=True
        )

        err_hist = wrap_err_deg(res_hist.yaw_deg, true_deg)
        err_icp  = wrap_err_deg(res_icp.yaw_deg,  true_deg)

        # Store for plots
        results["angle_deg"].append(true_deg)
        results["err_hist"].append(err_hist)
        results["err_icp"].append(err_icp)
        results["conf"].append(res_hist.confidence)
        results["ambig"].append(res_hist.ambiguity)
        results["t_hist"].append(t_hist)
        results["t_icp"].append(t_icp)

        print(f"{true_deg:>+8.1f} | {res_hist.yaw_deg:>+8.1f} "
              f"| {err_hist:>+8.2f} | {res_hist.confidence:>8.2f} "
              f"| {res_hist.ambiguity:>6.2f} | {t_hist*1000:>8.2f}ms "
              f"| {'':>9} | {'histogram':>15}")
        print(f"{'':>8} | {res_icp.yaw_deg:>+8.1f} "
              f"| {err_icp:>+8.2f} | {res_icp.confidence:>8.2f} "
              f"| {res_icp.ambiguity:>6.2f} | {'':>9} "
              f"| {t_icp*1000:>8.2f}ms | {'histogram+icp':>15}")

    # ---------- Summary statistics ----------
    err_hist_arr = np.abs(np.array(results["err_hist"]))
    err_icp_arr  = np.abs(np.array(results["err_icp"]))
    t_hist_arr   = np.array(results["t_hist"])
    t_icp_arr    = np.array(results["t_icp"])

    print("\n=== Summary ===")
    print(f"Histogram only  | mean |err| = {err_hist_arr.mean():>6.3f}° | "
          f"max |err| = {err_hist_arr.max():>6.3f}° | "
          f"mean time = {t_hist_arr.mean()*1000:>7.2f} ms")
    print(f"Histogram + ICP | mean |err| = {err_icp_arr.mean():>6.3f}° | "
          f"max |err| = {err_icp_arr.max():>6.3f}° | "
          f"mean time = {t_icp_arr.mean()*1000:>7.2f} ms")
    print(f"Speedup (hist vs. hist+icp): "
          f"{t_icp_arr.mean() / t_hist_arr.mean():.1f}×")
    print(f"Accuracy improvement: "
          f"{err_hist_arr.mean() - err_icp_arr.mean():+.4f}° "
          f"(mean), "
          f"{err_hist_arr.max() - err_icp_arr.max():+.4f}° "
          f"(max)")

    # ---------- Plots ----------
    plot_timing_results(results)


# ---------- Plotting ----------
def plot_timing_results(results: dict):
    """Plot timing and accuracy results."""
    angles = np.array(results["angle_deg"])
    t_hist_ms = np.array(results["t_hist"]) * 1000.0
    t_icp_ms  = np.array(results["t_icp"]) * 1000.0
    err_hist = np.abs(np.array(results["err_hist"]))
    err_icp  = np.abs(np.array(results["err_icp"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Plot 1: Time per angle, grouped bars ----
    ax = axes[0]
    x = np.arange(len(angles))
    width = 0.38
    b1 = ax.bar(x - width/2, t_hist_ms, width,
                label="histogram", color="tab:blue", alpha=0.85)
    b2 = ax.bar(x + width/2, t_icp_ms, width,
                label="histogram + ICP", color="tab:orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:+d}°" for a in angles], rotation=45)
    ax.set_ylabel("time (ms)")
    ax.set_title("Execution time per yaw estimate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    # ---- Plot 2: Mean time bar chart ----
    ax = axes[1]
    means = [t_hist_ms.mean(), t_icp_ms.mean()]
    stds  = [t_hist_ms.std(), t_icp_ms.std()]
    bars = ax.bar(["histogram", "histogram + ICP"], means,
                  yerr=stds, capsize=6,
                  color=["tab:blue", "tab:orange"], alpha=0.85)
    ax.set_ylabel("mean time (ms)")
    ax.set_title("Mean time per method (error bars = std)")
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:.1f} ms",
                    xy=(bar.get_x() + bar.get_width()/2, m),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    # ---- Plot 3: Error vs. angle ----
    ax = axes[2]
    ax.plot(angles, err_hist, "o-", label="histogram",
            color="tab:blue", markersize=6)
    ax.plot(angles, err_icp, "s--", label="histogram + ICP",
            color="tab:orange", markersize=6)
    ax.set_xlabel("true yaw (deg)")
    ax.set_ylabel("|error| (deg)")
    ax.set_title("Accuracy per angle")
    ax.set_ylim(bottom=-0.05)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()