# report_yaw_experiments.py
"""
Generate the figures and metrics for the yaw-estimation report.

Outputs
-------
    yaw_report/figures/*.png   figures referenced by the Markdown report
    yaw_report/results.csv     per-pair metrics for every method
    (also prints a Markdown-ready summary table to stdout)

Methods compared
----------------
    pca              : 2D PCA (no odometry)
    pca+odom         : 2D PCA, 180-degree ambiguity resolved by odometry
    simple           : single angular histogram + circular cross-correlation
    stratified       : radial-ring angular histograms
    simple+odom      : simple histogram + odometry disambiguation
    stratified+odom  : stratified histogram + odometry disambiguation

Usage
-----
    python report_yaw_experiments.py
"""

import sys
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent
sys.path.append(str(root_project))

from src.place_encoder import PointCloud
from src.yaw_estimator import (
    angular_histogram,
    angular_histogram_stratified,
    estimate_yaw_from_histograms,
    estimate_yaw_from_stratified_histograms,
    estimate_yaw_from_histograms_with_odometry,
    estimate_yaw_from_stratified_histograms_with_odometry,
    resolve_yaw_ambiguity_with_odometry,
    _circular_cross_correlation,
)
from src.yaw_estimator_pca import (
    estimate_yaw_pca,
    estimate_yaw_pca_with_odometry,
)
from test_pipeline_revisits import (
    load_ground_truth_poses,
    ground_truth_relative_yaw_deg,
    wrap_angle_deg,
)


DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
TRAJ_JSON = DATA_DIR / "trajectory_map_ros2.json"
OUTDIR = Path(__file__).resolve().parent / "yaw_report"
FIGDIR = OUTDIR / "figures"

PAIRS = [(8, 90), (8, 94), (1, 97), (1, 104)]   # (stored_wp, query_wp)
N_BINS = 360
N_RINGS = 10

METHODS = ["pca", "pca+odom", "simple", "stratified",
           "simple+odom", "stratified+odom"]

COLORS = {
    "pca": "#4C72B0", "pca+odom": "#1F3B6F",
    "simple": "#DD8452", "stratified": "#937860",
    "simple+odom": "#C44E52", "stratified+odom": "#8C2D2D",
}


# ============================================================
# Helpers
# ============================================================
def load_pc(wp: int) -> PointCloud:
    matches = sorted(DATA_DIR.glob(f"wp_{wp:04d}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"wp_{wp:04d} not found")
    return PointCloud.from_npy(matches[0])


def histograms(kind: str, pc_a, pc_b):
    """Return (ha, hb) for 'simple' or 'stratified'."""
    if kind == "stratified":
        ha = angular_histogram_stratified(pc_a, n_bins=N_BINS, n_rings=N_RINGS)
        hb = angular_histogram_stratified(pc_b, n_bins=N_BINS, n_rings=N_RINGS)
    else:
        ha = angular_histogram(pc_a, n_bins=N_BINS)
        hb = angular_histogram(pc_b, n_bins=N_BINS)
    return ha, hb


def estimate(method: str, pc_a, pc_b, odom_a: float, odom_b: float):
    """Return (yaw_deg, hist_ms, corr_ms, total_ms)."""
    t0 = time.perf_counter()
    hist_ms = corr_ms = float("nan")

    if method == "pca":
        yaw = estimate_yaw_pca(pc_a, pc_b).yaw_deg
    elif method == "pca+odom":
        yaw = estimate_yaw_pca_with_odometry(
            pc_a, pc_b, odom_a, odom_b).yaw_deg
    else:
        kind = method.split("+")[0]
        use_odom = method.endswith("+odom")

        th0 = time.perf_counter()
        ha, hb = histograms(kind, pc_a, pc_b)
        th1 = time.perf_counter()

        if use_odom:
            if kind == "stratified":
                r = estimate_yaw_from_stratified_histograms_with_odometry(
                    ha, hb, odom_a, odom_b)
            else:
                r = estimate_yaw_from_histograms_with_odometry(
                    ha, hb, odom_a, odom_b)
        elif kind == "stratified":
            r = estimate_yaw_from_stratified_histograms(ha, hb)
        else:
            r = estimate_yaw_from_histograms(ha, hb)

        th2 = time.perf_counter()
        yaw = float(np.rad2deg(r[0]))
        hist_ms = (th1 - th0) * 1000.0
        corr_ms = (th2 - th1) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    return yaw, hist_ms, corr_ms, total_ms


# ============================================================
# Collect all results
# ============================================================
def collect(gt_poses):
    rows = []
    for wp_a, wp_b in PAIRS:
        pc_a, pc_b = load_pc(wp_a), load_pc(wp_b)
        odom_a = gt_poses[wp_a]["yaw_rad"]
        odom_b = gt_poses[wp_b]["yaw_rad"]
        gt = ground_truth_relative_yaw_deg(gt_poses, wp_a, wp_b)

        # Repeat each timing a few times and keep the median (more stable)
        for method in METHODS:
            timings, build_list, comp_list = [], [], []
            yaw = None
            for _ in range(5):
                yaw, hm, cm, tm = estimate(method, pc_a, pc_b, odom_a, odom_b)
                timings.append(tm)
                if not np.isnan(hm):
                    build_list.append(hm)
                    comp_list.append(cm)
            rows.append({
                "wp_stored": wp_a,
                "wp_new": wp_b,
                "method": method,
                "yaw_deg": yaw,
                "gt_deg": gt,
                "err_deg": wrap_angle_deg(yaw - gt),
                "hist_ms": float(np.median(build_list)) if build_list
                else float("nan"),
                "corr_ms": float(np.median(comp_list)) if comp_list
                else float("nan"),
                "total_ms": float(np.median(timings)),
            })
    return rows


# ============================================================
# Figures
# ============================================================
def fig_methods_summary(rows):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(METHODS))
    means, medians, maxs = [], [], []
    for m in METHODS:
        e = np.abs([r["err_deg"] for r in rows if r["method"] == m])
        means.append(e.mean())
        medians.append(np.median(e))
        maxs.append(e.max())

    w = 0.27
    ax.bar(x - w, means, w, label="mean |err|", color="#4C72B0")
    ax.bar(x, medians, w, label="median |err|", color="#55A868")
    ax.bar(x + w, maxs, w, label="max |err|", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("yaw error (deg)")
    ax.set_title("Yaw error by method (4 revisits)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    p = FIGDIR / "fig2_methods_summary.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_per_pair_errors(rows):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(PAIRS))
    w = 0.13
    for i, m in enumerate(METHODS):
        errs = []
        for wp_a, wp_b in PAIRS:
            e = [r["err_deg"] for r in rows
                 if r["method"] == m and r["wp_stored"] == wp_a
                 and r["wp_new"] == wp_b][0]
            errs.append(abs(e))
        ax.bar(x + (i - len(METHODS) / 2) * w + w / 2, errs, w,
               label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"wp{a}\u2192wp{b}" for a, b in PAIRS])
    ax.set_ylabel("|yaw error| (deg)")
    ax.set_title("Per-revisit absolute yaw error")
    ax.axhline(90, color="k", ls=":", lw=1, alpha=0.6)
    ax.text(0.02, 92, "90\u00b0 (ambiguity scale)", fontsize=8, alpha=0.7)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    p = FIGDIR / "fig3_per_pair_errors.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_timing(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    hist_methods = ["simple", "stratified"]
    x = np.arange(len(hist_methods) + 1)

    build = []
    corr = []
    for m in hist_methods:
        build.append(np.nanmean([r["hist_ms"] for r in rows
                                 if r["method"] == m]))
        corr.append(np.nanmean([r["corr_ms"] for r in rows
                                if r["method"] == m]))
    pca_total = np.nanmean([r["total_ms"] for r in rows
                            if r["method"] == "pca+odom"])

    labels = ["simple", "stratified", "pca+odom"]
    ax.bar(x[0], build[0], color="#DD8452", label="histogram build")
    ax.bar(x[0], corr[0], bottom=build[0], color="#C44E52",
           label="comparison (cross-corr.)")
    ax.bar(x[1], build[1], color="#DD8452")
    ax.bar(x[1], corr[1], bottom=build[1], color="#C44E52")
    ax.bar(x[2], pca_total, color="#4C72B0", label="PCA+odom (total)")

    for xi, (b, c) in zip(x[:2], zip(build, corr)):
        ax.text(xi, b + c + 0.05, f"{b:.2f}+{c:.2f}", ha="center",
                fontsize=9)
    ax.text(x[2], pca_total + 0.05, f"{pca_total:.2f}", ha="center",
            fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("time per comparison (ms)")
    ax.set_title("Timing: histogram build vs. comparison")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    p = FIGDIR / "fig4_timing.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_histogram_illustration(gt_poses):
    """Simple histogram: polar, overlay, correlation, aligned (pair 1->104)."""
    wp_a, wp_b = 1, 104
    pc_a, pc_b = load_pc(wp_a), load_pc(wp_b)
    gt = ground_truth_relative_yaw_deg(gt_poses, wp_a, wp_b)

    ha = angular_histogram(pc_a, n_bins=N_BINS)
    hb = angular_histogram(pc_b, n_bins=N_BINS)
    yaw_rad, conf, amb = estimate_yaw_from_histograms(ha, hb)
    corr = _circular_cross_correlation(ha, hb)
    k = int(np.argmax(corr))

    theta = -np.pi + (np.arange(N_BINS) + 0.5) * (2 * np.pi / N_BINS)
    theta_deg = np.rad2deg(theta)
    shift = 2 * np.pi * np.arange(N_BINS) / N_BINS
    shift_deg = np.rad2deg(((shift + np.pi) % (2 * np.pi)) - np.pi)
    order = np.argsort(shift_deg)
    hb_aligned = np.roll(hb, -k)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"Angular-histogram estimator (simple) \u2014 wp{wp_a} (A) vs "
                 f"wp{wp_b} (B)  |  est={np.rad2deg(yaw_rad):+.2f}\u00b0  "
                 f"gt={gt:+.2f}\u00b0  |  conf={conf:.1f}  amb={amb:.2f}",
                 fontsize=12)

    ax = fig.add_subplot(2, 2, 1, projection="polar")
    ax.plot(theta, ha, color="#4C72B0", lw=1.2, label="A (stored)")
    ax.plot(theta, hb, color="#DD8452", lw=1.2, alpha=0.85, label="B (query)")
    ax.set_title("Polar angular histograms")
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(theta_deg, ha, color="#4C72B0", lw=1.2, label="A (stored)")
    ax.plot(theta_deg, hb, color="#DD8452", lw=1.2, alpha=0.85,
            label="B (query)")
    ax.set_xlabel("angle (deg)"); ax.set_ylabel("density")
    ax.set_title("Histograms overlaid")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(shift_deg[order], corr[order], color="#55A868", lw=1.2)
    ax.axvline(shift_deg[k], color="red", ls="--", lw=1.5,
               label=f"peak @ {shift_deg[k]:+.1f}\u00b0")
    ax.axvline(wrap_angle_deg(gt), color="black", ls="-.", lw=1.5,
               label=f"gt = {gt:+.2f}\u00b0")
    ax.set_xlabel("shift (deg)"); ax.set_ylabel("correlation")
    ax.set_title("Circular cross-correlation")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(theta_deg, ha, color="#4C72B0", lw=1.4, label="A (stored)")
    ax.plot(theta_deg, hb_aligned, color="#C44E52", lw=1.2, alpha=0.85,
            label=f"B shifted by {np.rad2deg(yaw_rad):+.1f}\u00b0")
    ax.set_xlabel("angle (deg)"); ax.set_ylabel("density")
    ax.set_title("A vs. B aligned by the estimate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGDIR / "fig8_histogram_illustration.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_odom_flip(gt_poses):
    """Show the 180-degree ambiguity and its odometry fix (pair 8->94)."""
    wp_a, wp_b = 8, 94
    pc_a, pc_b = load_pc(wp_a), load_pc(wp_b)
    gt = ground_truth_relative_yaw_deg(gt_poses, wp_a, wp_b)
    odom_a = gt_poses[wp_a]["yaw_rad"]
    odom_b = gt_poses[wp_b]["yaw_rad"]

    ha, hb = histograms("simple", pc_a, pc_b)
    yaw_rad, conf, amb = estimate_yaw_from_histograms(ha, hb)
    yaw_odom, rel_odom = resolve_yaw_ambiguity_with_odometry(
        yaw_rad, odom_a, odom_b)
    corr = _circular_cross_correlation(ha, hb)

    shift = 2 * np.pi * np.arange(N_BINS) / N_BINS
    shift_deg = np.rad2deg(((shift + np.pi) % (2 * np.pi)) - np.pi)
    order = np.argsort(shift_deg)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(shift_deg[order], corr[order], color="#55A868", lw=1.3)
    ax.axvline(np.rad2deg(yaw_rad), color="red", ls="--", lw=1.6,
               label=f"histogram peak = {np.rad2deg(yaw_rad):+.1f}\u00b0")
    ax.axvline(np.rad2deg(yaw_odom), color="purple", ls=":", lw=1.8,
               label=f"after odometry = {np.rad2deg(yaw_odom):+.1f}\u00b0")
    ax.axvline(wrap_angle_deg(gt), color="black", ls="-.", lw=1.6,
               label=f"ground truth = {gt:+.2f}\u00b0")
    ax.axvline(np.rad2deg(rel_odom), color="tab:cyan", ls="-", lw=1.0,
               alpha=0.7, label=f"odom prediction = {np.rad2deg(rel_odom):+.1f}\u00b0")
    ax.set_xlabel("shift (deg)"); ax.set_ylabel("correlation")
    ax.set_title("180\u00b0 ambiguity on wp8\u2192wp94 and the odometry fix")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = FIGDIR / "fig5_odom_flip.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_pca_outliers(gt_poses):
    """PCA failure mode: axis set by far outliers (pair 1->104)."""
    pc_a, pc_b = load_pc(1), load_pc(104)
    radii = [None, 6.0, 5.0, 4.0, 3.0, 2.0]
    phi_a, phi_b, aniso, yaw, xs = [], [], [], [], []
    for R in radii:
        ra = estimate_yaw_pca(pc_a, pc_b, max_radius=R)
        phi_a.append(np.rad2deg(ra.phi_a_rad))
        phi_b.append(np.rad2deg(ra.phi_b_rad))
        aniso.append(ra.anisotropy)
        yaw.append(ra.yaw_deg)
        xs.append(0 if R is None else R)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    xpos = np.arange(len(radii))
    labels = ["all"] + [f"{r:g}" for r in radii[1:]]

    ax = axes[0]
    ax.plot(xpos, phi_a, "o-", label="\u03c6_A (wp1)")
    ax.plot(xpos, phi_b, "s-", label="\u03c6_B (wp104)")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_xlabel("max radius (m)"); ax.set_ylabel("principal axis (deg)")
    ax.set_title("PCA axis swings with the radius clip")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(xpos, aniso, "d-", color="#C44E52", label="anisotropy (min)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_xlabel("max radius (m)"); ax.set_ylabel("anisotropy  \u03bb\u2081/\u03bb\u2082")
    ax.set_title("Dense core is near-isotropic (aniso\u21921)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    p = FIGDIR / "fig6_pca_outliers.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_synthetic():
    """Recover a known rotation on a synthetic asymmetric scene."""
    rng = np.random.default_rng(7)
    # asymmetric room: three walls of different lengths
    def wall(p0, p1, n):
        t = np.linspace(0, 1, n)[:, None]
        return p0 + t * (np.asarray(p1) - np.asarray(p0))
    pts = np.vstack([
        wall((-5, 0), (5, 0), 1500),
        wall((5, 0), (5, 3), 600),
        wall((-5, 0), (-5, 1.5), 300),
    ])
    pts += rng.normal(0, 0.05, pts.shape)
    pts3 = np.column_stack([pts, rng.normal(1.0, 0.1, len(pts))])

    angles = np.arange(-150, 151, 15)
    est_simple, est_strat = [], []
    for ang in angles:
        th = np.deg2rad(ang)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pc_a = PointCloud(pts3.copy())
        pc_b = PointCloud(pts3 @ R.T)
        ha = angular_histogram(pc_a, n_bins=N_BINS)
        hb = angular_histogram(pc_b, n_bins=N_BINS)
        ys, _, _ = estimate_yaw_from_histograms(ha, hb)
        hs = angular_histogram_stratified(pc_a, n_bins=N_BINS, n_rings=N_RINGS)
        ht = angular_histogram_stratified(pc_b, n_bins=N_BINS, n_rings=N_RINGS)
        yt, _, _ = estimate_yaw_from_stratified_histograms(hs, ht)
        est_simple.append(np.rad2deg(ys))
        est_strat.append(np.rad2deg(yt))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lims = [-190, 190]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.scatter(angles, est_simple, s=45, label="simple", color="#DD8452")
    ax.scatter(angles, est_strat, s=45, marker="s", label="stratified",
               color="#937860", alpha=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true rotation (deg)")
    ax.set_ylabel("estimated yaw (deg)")
    ax.set_title("Synthetic validation (asymmetric room)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    p = FIGDIR / "fig1_synthetic.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p, np.array(est_simple), np.array(est_strat), angles


def fig_odom_robustness(gt_poses):
    """% correct 180-degree flips vs odometry noise."""
    noise_levels = [0, 10, 20, 30, 45, 60, 75, 90]
    trials = 300

    # Precompute raw yaw + true relative yaw per pair (per method)
    raw = {}
    for kind in ("simple", "stratified"):
        raw[kind] = []
        for wp_a, wp_b in PAIRS:
            pc_a, pc_b = load_pc(wp_a), load_pc(wp_b)
            ha, hb = histograms(kind, pc_a, pc_b)
            if kind == "stratified":
                y, _, _ = estimate_yaw_from_stratified_histograms(ha, hb)
            else:
                y, _, _ = estimate_yaw_from_histograms(ha, hb)
            gt = ground_truth_relative_yaw_deg(gt_poses, wp_a, wp_b)
            raw[kind].append((y, gt_poses[wp_a]["yaw_rad"],
                              gt_poses[wp_b]["yaw_rad"],
                              gt))

    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, color, marker in (("simple", "#DD8452", "o"),
                                ("stratified", "#937860", "s")):
        acc = []
        for nl in noise_levels:
            ok = 0
            tot = 0
            for (y, oa, ob, gt) in raw[kind]:
                for t in range(trials):
                    rng = np.random.default_rng(1000 + t)
                    na = oa + np.deg2rad(rng.normal(0, nl))
                    nb = ob + np.deg2rad(rng.normal(0, nl))
                    cand, _ = resolve_yaw_ambiguity_with_odometry(y, na, nb)
                    err = abs(wrap_angle_deg(np.rad2deg(cand) - gt))
                    ok += (err < 30)
                    tot += 1
            acc.append(100.0 * ok / tot)
        ax.plot(noise_levels, acc, marker=marker, color=color,
                label=f"{kind}+odom")
    ax.set_xlabel("odometry yaw noise std (deg)")
    ax.set_ylabel("estimates within 30\u00b0 of GT (%)")
    ax.set_title("Robustness of the odometry disambiguation")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    p = FIGDIR / "fig7_odom_robustness.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ============================================================
# Main
# ============================================================
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    gt_poses = load_ground_truth_poses(TRAJ_JSON)

    print("Collecting results...")
    rows = collect(gt_poses)

    # CSV
    csv_path = OUTDIR / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[csv]  {csv_path}")

    # Figures
    print("Generating figures...")
    fig_methods_summary(rows)
    fig_per_pair_errors(rows)
    fig_timing(rows)
    fig_histogram_illustration(gt_poses)
    fig_odom_flip(gt_poses)
    fig_pca_outliers(gt_poses)
    _, es, est_s, ang = fig_synthetic()
    fig_odom_robustness(gt_poses)

    print(f"[fig]  {FIGDIR}/*.png")

    # Markdown table
    print("\n### Per-method summary\n")
    print("| method | mean \\|err\\| | median \\|err\\| | max \\|err\\| | "
          "time (ms) |")
    print("|---|---|---|---|---|")
    for m in METHODS:
        e = np.abs([r["err_deg"] for r in rows if r["method"] == m])
        t = [r["total_ms"] for r in rows if r["method"] == m]
        print(f"| `{m}` | {e.mean():.2f}° | {np.median(e):.2f}° | "
              f"{e.max():.2f}° | {np.mean(t):.2f} |")

    print("\n### Per-pair estimated yaw (deg)\n")
    print("| pair | " + " | ".join(METHODS) + " | gt |")
    print("|---" * (len(METHODS) + 2) + "|")
    for wp_a, wp_b in PAIRS:
        vals = []
        for m in METHODS:
            y = [r["yaw_deg"] for r in rows
                 if r["method"] == m and r["wp_stored"] == wp_a
                 and r["wp_new"] == wp_b][0]
            vals.append(f"{y:+.2f}")
        gt = [r["gt_deg"] for r in rows
              if r["wp_stored"] == wp_a and r["wp_new"] == wp_b][0]
        print(f"| wp{wp_a}→wp{wp_b} | " + " | ".join(vals) +
              f" | {gt:+.2f} |")

    print("\n### Synthetic validation (simple)\n")
    err_s = np.abs(((es - ang + 180) % 360) - 180)
    err_st = np.abs(((est_s - ang + 180) % 360) - 180)
    print(f"| method | mean \\|err\\| | max \\|err\\| |")
    print(f"|---|---|---|")
    print(f"| simple | {err_s.mean():.2f}° | {err_s.max():.2f}° |")
    print(f"| stratified | {err_st.mean():.2f}° | {err_st.max():.2f}° |")


if __name__ == "__main__":
    main()
