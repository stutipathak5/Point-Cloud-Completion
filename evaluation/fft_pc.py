import numpy as np
import os
import open3d as o3d
import uuid
from collections import defaultdict
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import h5py
import random


# def array_to_tuple(arr):
#     return tuple(map(tuple, arr))

# def find_duplicate_indices(X):
#     subarray_map = defaultdict(list)

#     for i in range(X.shape[0]):
#         subarray = array_to_tuple(X[i])
#         subarray_map[subarray].append(i)

#     duplicates = {}
#     key_counter = 0
#     for subarray, indices in subarray_map.items():
#         if len(indices) > 1:
#             duplicates[key_counter] = (indices, np.array(subarray))
#             key_counter += 1

#     return duplicates

# comp_tr = np.load('Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_tr.npy')
# comp_te = np.load('Point-Cloud-Autoencoder/data/final_splits/Dutch/difficult/splits/comp_te.npy')

# comp_combined = np.concatenate((comp_tr, comp_te), axis=0)

# duplicates = find_duplicate_indices(comp_combined)

# print(len(duplicates))

# arr_list = []
# for k, v in duplicates.items():
#     arr_list.append(v[1])

# with open('arr_list.pkl', 'wb') as f:
#     pickle.dump(arr_list, f)


# with open('evaluation/arr_list.pkl', 'rb') as f:
#     arr_list = pickle.load(f)

    
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


def noise(points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances_to_planes = []

    for i, point in enumerate(points):
        _, indices = nbrs.kneighbors([point])
        neighbors = points[indices[0]]

        X = neighbors[:, :2]  
        y = neighbors[:, 2]   
        plane_model = LinearRegression().fit(X, y)

        a, b = plane_model.coef_
        c, d = -1, plane_model.intercept_

        x0, y0, z0 = point
        distance = abs(a * x0 + b * y0 + c * z0 + d) / np.sqrt(a**2 + b**2 + c**2)
        distances_to_planes.append(distance)

    average_distance = np.mean(distances_to_planes)
    return average_distance


def distance(points, k, plot=False):
    # Initialize Nearest Neighbors with k=2 (k=1 is the point itself, k=2 is the nearest other point)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    nearest_neighbor_distances = np.mean(distances[:, 1:], axis=1)
    std_dev = np.std(nearest_neighbor_distances)
    
    if plot:
        # Plotting the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(nearest_neighbor_distances, bins=100, color='c', edgecolor='k', alpha=0.7)
        plt.title("Histogram of Nearest Neighbor Distances (k=1)")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.show()
    return std_dev


def voxel(points, voxel_size, plot=False):

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    voxel_size = voxel_size 

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

    o3d.visualization.draw_geometries([voxel_grid])

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    grid_size = tuple(np.ceil((max_bound - min_bound) / voxel_size).astype(int))
    voxel_counts = np.zeros(grid_size)


    for point in points:
        voxel_index = ((point - min_bound) / voxel_size).astype(int)
        voxel_counts[voxel_index[0], voxel_index[1], voxel_index[2]] += 1

    flattened_counts = voxel_counts[voxel_counts > 0]
    std_dev = np.std(flattened_counts)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(flattened_counts, bins=30, color='c', edgecolor='k', alpha=0.7)
        plt.title("Histogram of Points per Voxel")
        plt.xlabel("Number of Points in Voxel")
        plt.ylabel("Frequency")
        plt.show()
    return std_dev


def load_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hf:
        array = hf[dataset_name][:]
    return array


# points = complete[7]
# fft(points, 0.1)

no_samples = 30




"non uniformity pcn"

# parent_directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete"
# subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
# for sub_dir in subdirectories:
#     print("x")
#     directory_path = sub_dir
#     all_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
#     selected_files = random.sample(all_files, no_samples)
#     std=0
#     for s in selected_files:
#         point_cloud = o3d.io.read_point_cloud(s)
#         points = np.asarray(point_cloud.points)
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(points)
#         # o3d.visualization.draw_geometries([point_cloud])
#         std+=distance(points, 2)
#     print(sub_dir)
#     print(std/no_samples)


"non uniformity ours"

# npy_folder_path = 'data/Hungarian/easy_0.8'

# complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")
# occl = load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl")
# non_sparse = load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse")
# uni_sparse = load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse")


# np.random.shuffle(complete)
# np.random.shuffle(occl)
# np.random.shuffle(non_sparse)
# np.random.shuffle(uni_sparse)
# print(complete.shape)
# print(occl.shape)
# print(non_sparse.shape)
# print(uni_sparse.shape)


# std=0
# for i in range(no_samples):
#     points = complete[i]
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw_geometries([point_cloud])
#     std+=distance(points, 2)
# print(std/no_samples)







"curv and noise ours"

# npy_folder_path = 'data/Hungarian/easy_0.8'         

# complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")
# occl = load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl")
# non_sparse = load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse")
# uni_sparse = load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse")


# np.random.shuffle(complete)
# np.random.shuffle(occl)
# np.random.shuffle(non_sparse)
# np.random.shuffle(uni_sparse)
# print(complete.shape)
# print(occl.shape)
# print(non_sparse.shape)
# print(uni_sparse.shape)


# avg1 = 0
# avg2 = 0
# for i in range(no_samples):
#     points = complete[i]
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw_geometries([point_cloud])
#     curv, average = curv_avg(points, 15)
#     avg_noise = noise(points, 15)
#     avg1+=average
#     avg2+=avg_noise
# print(avg1/no_samples)
# print(avg2/no_samples)


"curv and noise pcn"

parent_directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete"
subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
for sub_dir in subdirectories:
    print("x")
    directory_path = sub_dir
    all_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    selected_files = random.sample(all_files, no_samples)
    avg1 = 0
    avg2 = 0
    for s in selected_files:
        point_cloud = o3d.io.read_point_cloud(s)
        points = np.asarray(point_cloud.points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([point_cloud])
        curv, average = curv_avg(points, 15)
        avg_noise = noise(points, 15)
        avg1+=average
        avg2+=avg_noise
    print(sub_dir)
    print(avg1/no_samples)
    print(avg2/no_samples)







# points = complete[7]
# voxel(points, 0.1)
# point_cloud = o3d.io.read_point_cloud(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete\02691156\1a04e3eab45ca15dd86060f189eb133.pcd")
# points = np.asarray(point_cloud.points)
# voxel(points, 0.01)