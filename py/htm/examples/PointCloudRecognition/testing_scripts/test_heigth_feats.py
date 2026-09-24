import sys
import numpy as np
from pathlib import Path

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor

DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
npy_paths = sorted(DATA_DIR.glob("*.npy"))[:50]  # amostra

# Collect features
h_vals, mh_vals, sh_vals = [], [], []
for p in npy_paths:
    desc = PlaceDescriptor.from_pointcloud(PointCloud.from_npy(p))
    h_vals.append(desc.height)
    mh_vals.append(desc.mean_height)
    sh_vals.append(desc.std_height)

h_vals  = np.array(h_vals)
mh_vals = np.array(mh_vals)
sh_vals = np.array(sh_vals)

print(f"height      : mean={h_vals.mean():.3f}, std={h_vals.std():.3f}, "
      f"CV={h_vals.std()/abs(h_vals.mean()):.3f}")
print(f"mean_height : mean={mh_vals.mean():.3f}, std={mh_vals.std():.3f}, "
      f"CV={mh_vals.std()/abs(mh_vals.mean()):.3f}")
print(f"std_height  : mean={sh_vals.mean():.3f}, std={sh_vals.std():.3f}, "
      f"CV={sh_vals.std()/abs(sh_vals.mean()):.3f}")

# Correlation between them
print(f"\nCorrelations:")
print(f"  height ↔ mean_height : {np.corrcoef(h_vals, mh_vals)[0,1]:+.3f}")
print(f"  height ↔ std_height  : {np.corrcoef(h_vals, sh_vals)[0,1]:+.3f}")
print(f"  mean_height ↔ std_height: {np.corrcoef(mh_vals, sh_vals)[0,1]:+.3f}")