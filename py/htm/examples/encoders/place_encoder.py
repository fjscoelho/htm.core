# place_encoder.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from htm.bindings.encoders import RDSE, RDSE_Parameters


# ============================================================
# PointCloud
# ============================================================
@dataclass
class PointCloud:
    """
    Encapsulates a (N, 3) point cloud with basic geometric operations.
    Assumes Z is the vertical axis (yaw = rotation around Z).
    """
    points: np.ndarray  # (N, 3)

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(f"points must be (N,3), got {self.points.shape}")
        if len(self.points) == 0:
            raise ValueError("points is empty")
        self.points = self.points[:, :3].astype(np.float64)

    # ---------- Properties ----------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        """3D centroid."""
        return self.points.mean(axis=0)

    @property
    def centroid_xy(self) -> np.ndarray:
        """Centroid projected onto XY plane."""
        return self.centroid[:2]

    @property
    def centered(self) -> np.ndarray:
        """Points relative to centroid (N, 3)."""
        return self.points - self.centroid

    @property
    def xy(self) -> np.ndarray:
        """XY coordinates relative to centroid (N, 2)."""
        return self.centered[:, :2]

    @property
    def z(self) -> np.ndarray:
        """Z coordinate relative to centroid (N,)."""
        return self.centered[:, 2]

    @property
    def r_xy(self) -> np.ndarray:
        """Radial distance in XY relative to centroid (N,)."""
        return np.linalg.norm(self.xy, axis=1)

    # ---------- Constructors ----------
    @classmethod
    def from_npy(cls, path: str | Path) -> "PointCloud":
        """Load from a .npy file."""
        pc = np.load(str(path))
        return cls(points=pc) # == PointCloud(points=pc)

    # ---------- Operations ----------
    def rotated_yaw(self, angle_rad: float) -> "PointCloud":
        """Return a new PointCloud rotated around the centroid (Z axis)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
        centered = self.centered
        rotated = centered @ R.T + self.centroid
        return PointCloud(points=rotated)

    # ---------- Visualization ----------
    def plot_top_view(self, ax=None, show_centroid: bool = True):
        """Plot top view (XY) with centroid highlighted."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.points[:, 0], self.points[:, 1],
                   s=1, alpha=0.4, label="points")

        if show_centroid:
            cx, cy = self.centroid_xy
            ax.scatter([cx], [cy], s=120, c="red", marker="X",
                       edgecolors="black", linewidths=1.0, zorder=5,
                       label=f"centroid ({cx:.2f}, {cy:.2f})")

        ax.set_aspect("equal")
        ax.set_title(f"Top view (N={self.n_points})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)
        return ax


# ============================================================
# PlaceDescriptor
# ============================================================
@dataclass
class PlaceDescriptor:
    """
    Yaw-invariant descriptors of a point cloud.
    Use PlaceDescriptor.from_pointcloud(pc) to construct.
    """
    eigvals: np.ndarray         # (3,) eigenvalues of 3D covariance (descending)
    hist_z: np.ndarray          # (n_bins_z,) normalized height histogram
    hist_r: np.ndarray          # (n_bins_r,) normalized radial-distance histogram
    height: float               # height = z_max - z_min
    density: float              # density = log1p(N)
    volume: float               # volume of the covariance ellipsoid
    mean_radius: float          # mean radial distance
    std_radius: float           # std of radial distance

    # ---------- Main constructor ----------
    @classmethod
    def from_pointcloud(
        cls,
        pc: PointCloud,
        n_bins_z: int = 10,
        n_bins_r: int = 10,
        z_range: Optional[Tuple[float, float]] = None,
        r_range: Optional[Tuple[float, float]] = None,
    ) -> "PlaceDescriptor":
        """
        Compute yaw-invariant descriptors.

        Parameters
        ----------
        pc : PointCloud
        n_bins_z, n_bins_r : int
        z_range, r_range : optional (min, max)
            If given, uses fixed ranges (recommended for cross-cloud comparison).
        """
        N = pc.n_points
        z = pc.z
        r = pc.r_xy
        centered = pc.centered

        # --- 3D covariance eigenvalues (rotation-invariant) ---
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # λ1 ≥ λ2 ≥ λ3

        # Guard 1: minimum points
        if N < 100:
            raise ValueError(f"Too few points: {N}")

        # Guard 2: non-negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)

        # --- Histograms ---
        z_range = cls._fix_range(z_range, z,
                                 default_min=z.min(), default_max=z.max())
        r_range = cls._fix_range(r_range, r,
                                 default_min=0.0, default_max=r.max())

        hist_z, _ = np.histogram(z, bins=n_bins_z, range=z_range)
        hist_r, _ = np.histogram(r, bins=n_bins_r, range=r_range)
        hist_z = hist_z.astype(np.float64) / N
        hist_r = hist_r.astype(np.float64) / N

        # --- Scalars ---
        bbox = pc.points.max(axis=0) - pc.points.min(axis=0)
        height = float(bbox[2])
        density = float(np.log1p(N))
        # Guard 3: numerical safety for volume
        volume_elipsoide = float((4/3) * np.pi * np.sqrt(np.prod(eigvals) + 1e-12))
        mean_radius = float(r.mean())
        std_radius = float(r.std())

        return cls(
            eigvals=eigvals,
            hist_z=hist_z,
            hist_r=hist_r,
            height=height,
            density=density,
            volume=volume_elipsoide,
            mean_radius=mean_radius,
            std_radius=std_radius,
        )

    # ---------- Helpers ----------
    @staticmethod
    def _fix_range(rng, values, default_min, default_max):
        """Normalize ranges and avoid degenerate cases."""
        if rng is None:
            rng = (default_min, default_max)
        if rng[1] - rng[0] < 1e-9:
            rng = (rng[0] - 0.5, rng[1] + 0.5)
        return rng

    # ---------- Public API ----------
    def to_vector(self) -> np.ndarray:
        """Concatenate all features into a 1D vector (for comparison/debug)."""
        return np.concatenate([
            self.eigvals,
            self.hist_z,
            self.hist_r,
            [self.height, self.density, self.volume,
             self.mean_radius, self.std_radius],
        ])

    def distance_to(self, other: "PlaceDescriptor") -> float:
        """Normalized Euclidean distance between two descriptors."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + 1e-9))

    def correlation_with(self, other: "PlaceDescriptor") -> float:
        """Pearson correlation between the descriptor vectors."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.corrcoef(v1, v2)[0, 1])

    # ---------- Visualization ----------
    def plot_histograms(self, axs=None):
        """Plot Z and radial histograms."""
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].bar(range(len(self.hist_z)), self.hist_z)
        axs[0].set_title("hist_z (heights, normalized)")
        axs[0].set_xlabel("bin"); axs[0].set_ylabel("density")
        axs[1].bar(range(len(self.hist_r)), self.hist_r)
        axs[1].set_title("hist_r (radial distance XY)")
        axs[1].set_xlabel("bin"); axs[1].set_ylabel("density")
        return axs


# ============================================================
# SDRPlaceEncoder
# ============================================================
class SDRPlaceEncoder:
    """
    Encodes a PlaceDescriptor into a unified SDR.
    Each feature has its own RDSE encoder; outputs are concatenated.
    """

    def __init__(self, feature_ranges, feature_resolutions,
                 feature_sizes=None, active_bits=21, seed=42):
        n_feat = len(feature_ranges)
        if feature_sizes is None:
            feature_sizes = [400] * n_feat

        self.encoders = []
        self.feature_ranges = list(feature_ranges)
        self.feature_resolutions = list(feature_resolutions)
        self.feature_sizes = list(feature_sizes)
        self.active_bits = active_bits
        self.seed = seed

        for i in range(n_feat):
            lo, hi = feature_ranges[i]
            resolution = feature_resolutions[i]

            n_buckets = (hi - lo) / max(resolution, 1e-12)
            if n_buckets > feature_sizes[i] / 2:
                raise ValueError(
                    f"Feature {i}: too many buckets ({n_buckets:.0f}) "
                    f"for size {feature_sizes[i]}. "
                    f"Increase resolution or size."
                )

            params = RDSE_Parameters()
            params.size       = feature_sizes[i]
            params.activeBits = active_bits
            params.resolution = resolution
            params.seed       = seed
            self.encoders.append(RDSE(params))

        self.total_size = sum(self.feature_sizes)

    def encode(self, desc_vector: np.ndarray) -> np.ndarray:
        """
        desc_vector: (D,) from PlaceDescriptor.to_vector()
        Returns: (total_size,) uint8 array of 0/1
        """
        parts = []
        for i, enc in enumerate(self.encoders):
            out = enc.encode(desc_vector[i])
            arr = out.dense.flatten() if hasattr(out, "dense") else np.asarray(out).flatten()
            parts.append(arr.astype(np.uint8))
        return np.concatenate(parts)


# ============================================================
# Utility: combined visualization
# ============================================================
def plot_pointcloud_and_descriptors(pc: PointCloud,
                                    n_bins_z: int = 10,
                                    n_bins_r: int = 10,
                                    z_range=None, r_range=None):
    """Plot top view + descriptor histograms."""
    desc = PlaceDescriptor.from_pointcloud(
        pc, n_bins_z=n_bins_z, n_bins_r=n_bins_r,
        z_range=z_range, r_range=r_range,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pc.plot_top_view(ax=axes[0], show_centroid=True)
    desc.plot_histograms(axs=(axes[1], axes[2]))
    plt.tight_layout()
    plt.show()
    return desc