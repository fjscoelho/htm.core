import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor

DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
npy_paths = sorted(DATA_DIR.glob("*.npy"))

V = np.array([
    PlaceDescriptor.from_pointcloud(PointCloud.from_npy(p)).to_vector()
    for p in npy_paths
])

# Correlation matrix
corr = np.corrcoef(V.T)

# Plot
plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Feature correlation matrix")
plt.show()

# Print strong correlations (excluding diagonal)
print("Strong correlations (|r| > 0.8):")
n = V.shape[1]
for i in range(n):
    for j in range(i+1, n):
        if abs(corr[i, j]) > 0.8:
            print(f"  feature[{i}] ↔ feature[{j}] : r = {corr[i, j]:+.3f}")