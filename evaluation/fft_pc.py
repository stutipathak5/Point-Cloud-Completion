import numpy as np
import os
import open3d as o3d
import uuid
from collections import defaultdict
import torch


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

comp_tr = np.load('Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_tr.npy')
comp_te = np.load('Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_te.npy')

comp_combined = np.concatenate((comp_tr, comp_te), axis=0)

duplicates = find_duplicate_indices(comp_combined)

print(len(duplicates))

arr_list = []
for k, v in duplicates.items():
    arr_list.append(v[1])












def fft(points, voxel_size):

    from scipy.fft import fftn, fftshift
    import matplotlib.pyplot as plt

    # Load or create a point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Define voxel size
    voxel_size = voxel_size  # Adjust as necessary

    # Create a voxel grid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

    o3d.visualization.draw_geometries([voxel_grid])

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
    plt.show()




def curv_avg(points, neigh_size):

    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    from jakteristics import compute_features


    points_t = torch.tensor(points, dtype=torch.float32)
    pcd = Pointclouds(points=[points_t])
    bounding_box = pcd.get_bounding_boxes()
    diag = bounding_box[0, :, 1] - bounding_box[0, :, 0]
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / points_t.size(0)
    radius = torch.sqrt(surface_per_point * neigh_size)
    curv = (
        torch.from_numpy(
            compute_features(
                points,
                search_radius=radius,
                feature_names=["surface_variation"],
            )
        )
        .double()
        .squeeze(1)
    )

    mask = torch.isnan(curv) == False
    curv = curv[mask]
    # print("without Nan", curv.shape)
    average = curv.mean()
    return curv, average



# points = arr_list[7]
# fft(points, 0.1)


avg = 0
for points in arr_list[0:30]:

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([point_cloud])
    curv, average = curv_avg(points, 15)
    avg+=average
print(avg/30)
# tensor(0.0813, dtype=torch.float64)



directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete\02691156"
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]

avg = 0
for file_path in file_paths[0:30]:
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    curv, average = curv_avg(points, 15)
    avg+=average
print(avg/30)
# tensor(0.0770, dtype=torch.float64)