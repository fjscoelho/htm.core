# visualize_match.py
"""
Plot two point clouds side by side to visually inspect whether a match
is legitimate.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud

DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")


def plot_pair(name_a, name_b):
    pc_a = PointCloud.from_npy(DATA_DIR / name_a)
    pc_b = PointCloud.from_npy(DATA_DIR / name_b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    pc_a.plot_top_view(ax=axes[0])
    axes[0].set_title(f"A: {name_a}")
    pc_b.plot_top_view(ax=axes[1])
    axes[1].set_title(f"B: {name_b}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemplo: par que casou com overlap 0.864
    plot_pair(
        "wp_0000_1786110757_821066733_pointcloud_sensor.npy",
        "wp_0004_1786110774_839766221_pointcloud_sensor.npy",
    )
    # Exemplo: par que casou com overlap 0.86
    plot_pair(
        "wp_0009_1786110789_475020577_pointcloud_sensor.npy",
        "wp_0010_1786110793_034825287_pointcloud_sensor.npy",
    )
    # Exemplo: par que casou com overlap 0.93
    plot_pair(
        "wp_0001_1786110764_807464836_pointcloud_sensor.npy",
        "wp_0097_1786111000_044628145_pointcloud_sensor.npy",
    )