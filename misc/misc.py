import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

point_cloud = np.load("data\difficult\complete.npy")[30]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
voxel_size = 0.05
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
# o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized Point Cloud")



voxels = voxel_grid.get_voxels()
voxel_coordinates = np.array([voxel.grid_index for voxel in voxels])
print(voxel_coordinates.shape)


bounding_box = voxel_grid.get_axis_aligned_bounding_box()
min_bound = bounding_box.min_bound
max_bound = bounding_box.max_bound
grid_dimensions = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
print("Voxel Grid Dimensions:", grid_dimensions)


grid_size = (500, 500, 500)
voxel_grid_tensor = np.zeros(grid_size, dtype=np.float32)
for coord in voxel_coordinates:
    if all(coord < np.array(grid_size)):
        voxel_grid_tensor[tuple(coord)] = 1

print(np.count_nonzero(voxel_grid_tensor == 1))