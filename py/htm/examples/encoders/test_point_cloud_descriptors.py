# test_point_cloud_descriptors.py
import numpy as np
from place_encoder import (
    PointCloud, PlaceDescriptor,
    plot_pointcloud_and_descriptors,
)

def test_yaw_invariance(pc: PointCloud,
                        angles_deg=(0, 30, 45, 90, 135, 180, 270),
                        z_range=None, r_range=None,
                        plot: bool = False):
    """Compara descritores de uma nuvem rotacionada em vários ângulos."""
    desc_ref = PlaceDescriptor.from_pointcloud(pc, z_range=z_range, r_range=r_range)

    print(f"{'ângulo':>8} | {'||Δv||':>10} | {'corr':>8} | {'Δeigvals':>12}")
    print("-" * 55)

    for ang in angles_deg:
        pc_rot = pc.rotated_yaw(np.deg2rad(ang))
        desc = PlaceDescriptor.from_pointcloud(pc_rot, z_range=z_range, r_range=r_range)

        dv = desc_ref.distance_to(desc)
        corr = desc_ref.correlation_with(desc)
        deig = np.abs(desc.eigvals - desc_ref.eigvals).max()

        print(f"{ang:>7}° | {dv:>10.4f} | {corr:>8.4f} | {deig:>12.2e}")

        if plot:
            plot_pointcloud_and_descriptors(
                pc_rot, z_range=z_range, r_range=r_range,
            )


if __name__ == "__main__":
    # 1. Carrega
    pc = PointCloud.from_npy("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data/wp_0002_1786110766_170705611_pointcloud_sensor.npy")
    print(f"Point cloud: {pc.n_points} pontos")
    print(f"  centróide: {pc.centroid}")
    print(f"  centróide XY: {pc.centroid_xy}")

    # 2. Teste de invariância
    test_yaw_invariance(pc, plot=True)

    # 3. Visualização com centróide
    # desc = plot_pointcloud_and_descriptors(pc)
    desc = PlaceDescriptor.from_pointcloud(pc)

    # 4. Descritores
    print(f"\nAutovalores: {desc.eigvals}")
    print(f"Altura: {desc.altura:.3f}")
    print(f"Densidade: {desc.densidade:.3f}")
    print(f"Volume (elipsoide): {desc.volume:.3f}")
    print(f"Raio: {desc.raio_medio:.3f} ± {desc.raio_std:.3f}")
    

    