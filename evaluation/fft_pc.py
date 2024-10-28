import numpy as np
import os
import open3d as o3d
import uuid
from collections import defaultdict


def array_to_tuple(arr):
    return tuple(map(tuple, arr))

def find_duplicate_indices(X):
    subarray_map = defaultdict(list)

    for i in range(X.shape[0]):
        subarray = array_to_tuple(X[i])
        subarray_map[subarray].append(i)

    duplicates = {}
    key_counter = 0
    for subarray, indices in subarray_map.items():
        if len(indices) > 1:
            duplicates[key_counter] = (indices, np.array(subarray))
            key_counter += 1

    return duplicates

comp_tr = np.load('../Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_tr.npy')
comp_te = np.load('../Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_te.npy')

duplicates = find_duplicate_indices(comp_tr)

print(len(duplicates))

arr_list = []
for k, v in duplicates.items():
    arr_list.append(v[1])














from scipy.fft import fftn, fftshift
import matplotlib.pyplot as plt

# Load or create a point cloud
point_cloud = o3d.geometry.PointCloud()
points = arr_list[4]
point_cloud.points = o3d.utility.Vector3dVector(points)

# Define voxel size
voxel_size = 0.1  # Adjust as necessary

# Create a voxel grid from the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

# o3d.visualization.draw_geometries([voxel_grid])

# Create a 3D grid to hold the voxel counts
min_bound = np.min(points, axis=0)
max_bound = np.max(points, axis=0)
grid_size = tuple(np.ceil((max_bound - min_bound) / voxel_size).astype(int))
voxel_counts = np.zeros(grid_size)

# Count points in each voxel
for point in points:
    voxel_index = ((point - min_bound) / voxel_size).astype(int)
    voxel_counts[voxel_index[0], voxel_index[1], voxel_index[2]] += 1

# Apply FFT to the voxel counts
fft_result = fftn(voxel_counts)

# Shift zero frequency component to the center
fft_result_shifted = fftshift(fft_result)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(fft_result_shifted)

# Visualize a central slice of the magnitude spectrum
central_slice = magnitude_spectrum[magnitude_spectrum.shape[0] // 2]
plt.imshow(central_slice, cmap='gray')
plt.title('Magnitude Spectrum - Central Slice')
plt.colorbar()
plt.savefig('magnitude_spectrum.png')
