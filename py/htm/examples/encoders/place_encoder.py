# place_encoder.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path


# ============================================================
# PointCloud
# ============================================================
@dataclass
class PointCloud:
    """
    Encapsula uma point cloud (N, 3) e operações geométricas básicas.
    Assume eixo Z como vertical (yaw = rotação em torno de Z).
    """
    points: np.ndarray  # (N, 3)

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(f"points deve ser (N,3), recebido {self.points.shape}")
        if len(self.points) == 0:
            raise ValueError("points vazio")
        # Garante float64 e 3 colunas
        self.points = self.points[:, :3].astype(np.float64)

    # ---------- Propriedades ----------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        """Centróide 3D."""
        return self.points.mean(axis=0)

    @property
    def centroid_xy(self) -> np.ndarray:
        """Centróide projetado em XY."""
        return self.centroid[:2]

    @property
    def centered(self) -> np.ndarray:
        """Pontos relativos ao centróide (N, 3)."""
        return self.points - self.centroid

    @property
    def xy(self) -> np.ndarray:
        """Coordenadas XY relativas ao centróide (N, 2)."""
        return self.centered[:, :2]

    @property
    def z(self) -> np.ndarray:
        """Coordenada Z relativa ao centróide (N,)."""
        return self.centered[:, 2]

    @property
    def r_xy(self) -> np.ndarray:
        """Distância radial em XY em relação ao centróide (N,)."""
        return np.linalg.norm(self.xy, axis=1)

    # ---------- Construtores ----------
    @classmethod
    def from_npy(cls, path: str | Path) -> "PointCloud":
        """Carrega de um arquivo .npy."""
        pc = np.load(str(path))
        return cls(points=pc)

    # ---------- Operações ----------
    def rotated_yaw(self, angle_rad: float) -> "PointCloud":
        """Retorna uma nova PointCloud rotacionada em torno do centróide (eixo Z)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
        centered = self.centered
        rotated = centered @ R.T + self.centroid
        return PointCloud(points=rotated)

    # ---------- Visualização ----------
    def plot_top_view(self, ax=None, show_centroid: bool = True):
        """Plota vista de topo (XY) com centróide destacado."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.points[:, 0], self.points[:, 1], s=1, alpha=0.4, label="pontos")

        if show_centroid:
            cx, cy = self.centroid_xy
            ax.scatter([cx], [cy], s=120, c="red", marker="X",
                       edgecolors="black", linewidths=1.0,
                       zorder=5, label=f"centróide ({cx:.2f}, {cy:.2f})")

        ax.set_aspect("equal")
        ax.set_title(f"Vista de topo (N={self.n_points})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)
        return ax


# ============================================================
# PlaceDescriptor
# ============================================================
@dataclass
class PlaceDescriptor:
    """
    Descritores invariantes a yaw de uma point cloud.
    Use PlaceDescriptor.from_pointcloud(pc) para construir.
    """
    eigvals: np.ndarray      # (3,) autovalores da covariância 3D (decrescente)
    hist_z: np.ndarray       # (n_bins_z,) histograma normalizado de alturas
    hist_r: np.ndarray       # (n_bins_r,) histograma normalizado de distâncias radiais
    altura: float            # z_max - z_min
    densidade: float         # log1p(N)
    volume: float            # volume da elipsoide de covariância
    raio_medio: float
    raio_std: float

    # ---------- Construtor principal ----------
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
        Calcula descritores invariantes a yaw.

        Parâmetros
        ----------
        pc : PointCloud
        n_bins_z, n_bins_r : int
        z_range, r_range : (min, max) opcional
            Se fornecidos, usa ranges fixos (recomendado para comparar nuvens).
        """
        N = pc.n_points
        z = pc.z
        r = pc.r_xy
        centered = pc.centered

        # --- Autovalores da covariância 3D (invariantes a rotação) ---
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # λ1 ≥ λ2 ≥ λ3

        # --- Histogramas ---
        z_range = cls._fix_range(z_range, z, default_min=z.min(), default_max=z.max())
        r_range = cls._fix_range(r_range, r, default_min=0.0, default_max=r.max())

        hist_z, _ = np.histogram(z, bins=n_bins_z, range=z_range)
        hist_r, _ = np.histogram(r, bins=n_bins_r, range=r_range)
        hist_z = hist_z.astype(np.float64) / N
        hist_r = hist_r.astype(np.float64) / N

        # --- Escalares ---
        bbox = pc.points.max(axis=0) - pc.points.min(axis=0)
        altura = float(bbox[2])
        densidade = float(np.log1p(N))
        volume_elipsoide = float((4 / 3) * np.pi * np.sqrt(np.prod(eigvals)))
        raio_medio = float(r.mean())
        raio_std = float(r.std())

        return cls(
            eigvals=eigvals,
            hist_z=hist_z,
            hist_r=hist_r,
            altura=altura,
            densidade=densidade,
            volume=volume_elipsoide,
            raio_medio=raio_medio,
            raio_std=raio_std,
        )

    # ---------- Helpers ----------
    @staticmethod
    def _fix_range(rng, values, default_min, default_max):
        """Normaliza ranges e evita ranges degenerados."""
        if rng is None:
            rng = (default_min, default_max)
        if rng[1] - rng[0] < 1e-9:
            rng = (rng[0] - 0.5, rng[1] + 0.5)
        return rng

    # ---------- API pública ----------
    def to_vector(self) -> np.ndarray:
        """Concatena tudo num vetor 1D (para comparação/debug)."""
        return np.concatenate([
            self.eigvals,
            self.hist_z,
            self.hist_r,
            [self.altura, self.densidade, self.volume,
             self.raio_medio, self.raio_std],
        ])

    def distance_to(self, other: "PlaceDescriptor") -> float:
        """Distância euclidiana normalizada entre dois descritores."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + 1e-9))

    def correlation_with(self, other: "PlaceDescriptor") -> float:
        """Correlação de Pearson entre os vetores."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.corrcoef(v1, v2)[0, 1])

    # ---------- Visualização ----------
    def plot_histograms(self, axs=None):
        """Plota os histogramas de Z e radial."""
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].bar(range(len(self.hist_z)), self.hist_z)
        axs[0].set_title("hist_z (alturas, normalizado)")
        axs[0].set_xlabel("bin"); axs[0].set_ylabel("densidade")
        axs[1].bar(range(len(self.hist_r)), self.hist_r)
        axs[1].set_title("hist_r (distância radial XY)")
        axs[1].set_xlabel("bin"); axs[1].set_ylabel("densidade")
        return axs


# ============================================================
# Utilitário: visualização combinada
# ============================================================
def plot_pointcloud_and_descriptors(pc: PointCloud,
                                    n_bins_z: int = 10,
                                    n_bins_r: int = 10,
                                    z_range=None, r_range=None):
    """Plota vista de topo + histogramas do descritor."""
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