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
import torch.nn.functional as F
from ripser import Rips
import time


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



# def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
#     '''
#      seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
#     '''
#     n,c = xyz.shape

#     assert n == num_points
#     assert c == 3
#     if crop == num_points:
#         return xyz, None
        
#     INPUT = []
#     CROP = []
#     for points in xyz:
#         if isinstance(crop,list):
#             num_crop = random.randint(crop[0],crop[1])
#         else:
#             num_crop = crop

#         points = points.unsqueeze(0)

#         if fixed_points is None:       
#             center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
#         else:
#             if isinstance(fixed_points,list):
#                 fixed_point = random.sample(fixed_points,1)[0]
#             else:
#                 fixed_point = fixed_points
#             center = fixed_point.reshape(1,1,3).cuda()

#         distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

#         idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

#         if padding_zeros:
#             input_data = points.clone()
#             input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

#         else:
#             input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

#         crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

#         if isinstance(crop,list):
#             INPUT.append(fps(input_data,2048))
#             CROP.append(fps(crop_data,2048))
#         else:
#             INPUT.append(input_data)
#             CROP.append(crop_data)

#     input_data = torch.cat(INPUT,dim=0)# B N 3
#     crop_data = torch.cat(CROP,dim=0)# B M 3

#     return input_data.contiguous(), crop_data.contiguous()

# def fps(data, number):
#     '''
#         data B N 3
#         number int
#     '''
#     fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
#     fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
#     return fps_data


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



"non uniformity omni"

# parent_directory_path = r"C:\Users\spathak\Downloads\point_clouds-20241109T230701Z-002\point_clouds\ply_files\4096_ply\4096"
# subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
# # print(subdirectories)
# for sub_dir in subdirectories:
#     print("x")
#     directory_path = sub_dir
#     all_files = [os.path.join(directory_path, sub_dir) for sub_dir in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, sub_dir))]
#     if len(all_files) > 30:
#         all_files=all_files[:30]
#         no_samples = 30
#     else:
#         no_samples = len(all_files)
#     std=0
#     for s in all_files:
#         point_cloud = o3d.io.read_point_cloud(s +'\\pcd_4096.ply')
#         points = np.asarray(point_cloud.points)
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(points)
#         # o3d.visualization.draw_geometries([point_cloud])
#         std+=distance(points, 2)
#     print(sub_dir)
#     print(std/no_samples)


"non uniformity shapenet55"

# file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55\train.txt"
# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}

# for k,v in repeated_indices.items():
#     print(k)
#     if len(v) > 30:
#         std=0
#         for i in v[:30]:
#             points = np.load(os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc", lines[i]))
#             point_cloud = o3d.geometry.PointCloud()
#             point_cloud.points = o3d.utility.Vector3dVector(points)
#             std+=distance(points, 2)
#         print(std/no_samples)
#     else:
#         print("x")




"non uniformity shapenet34"

# # file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55\train.txt"
# file_path = "ShapeNet55/ShapeNet-34/train.txt"
# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}
# print(len(repeated_indices.keys()))

# for k,v in repeated_indices.items():
#     print(k)
#     if len(v) > 30:
#         std=0
#         for i in v[:30]:
#             # points = np.load(os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc", lines[i]))
#             points = np.load(os.path.join("ShapeNet55/shapenet_pc", lines[i]))
#             point_cloud = o3d.geometry.PointCloud()
#             point_cloud.points = o3d.utility.Vector3dVector(points)
#             std+=distance(points, 2)
#         print(std/no_samples)
#     else:
#         print("x")






"non uniformity mvp"

# with h5py.File('data/MVP_complete_pcs/mvp_complete_train.h5', 'r') as f:
#     complete = f['complete_pcds'][:]

# print(complete.shape)

# std=0
# for i in range(complete.shape[0]):
#     points = complete[i]
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw_geometries([point_cloud])
#     std+=distance(points, 2)
# print(std/complete.shape[0])




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

# parent_directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete"
# subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
# for sub_dir in subdirectories:
#     print("x")
#     directory_path = sub_dir
#     all_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
#     selected_files = random.sample(all_files, no_samples)
#     avg1 = 0
#     avg2 = 0
#     for s in selected_files:
#         point_cloud = o3d.io.read_point_cloud(s)
#         points = np.asarray(point_cloud.points)
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(points)
#         # o3d.visualization.draw_geometries([point_cloud])
#         curv, average = curv_avg(points, 15)
#         avg_noise = noise(points, 15)
#         avg1+=average
#         avg2+=avg_noise
#     print(sub_dir)
#     print(avg1/no_samples)
#     print(avg2/no_samples)



"curv and noise omni"

# parent_directory_path = r"C:\Users\spathak\Downloads\point_clouds-20241109T230701Z-002\point_clouds\ply_files\4096_ply\4096"
# subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
# # print(subdirectories)
# for sub_dir in subdirectories:
#     print("x")
#     directory_path = sub_dir
#     all_files = [os.path.join(directory_path, sub_dir) for sub_dir in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, sub_dir))]
#     if len(all_files) > 30:
#         all_files=all_files[:30]
#         no_samples = 30
#     else:
#         no_samples = len(all_files)
#     avg1 = 0
#     avg2 = 0
#     for s in all_files:
#         point_cloud = o3d.io.read_point_cloud(s +'\\pcd_4096.ply')
#         points = np.asarray(point_cloud.points)
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(points)
#         # o3d.visualization.draw_geometries([point_cloud])
#         curv, average = curv_avg(points, 15)
#         avg_noise = noise(points, 15)
#         avg1+=average
#         avg2+=avg_noise
#     print(sub_dir)
#     print(avg1/no_samples)
#     print(avg2/no_samples)


"curv and noise shapenet55"

# file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55\train.txt"
# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}

# for k,v in repeated_indices.items():
#     print(k)
#     if len(v) > 30:
#         avg1 = 0
#         avg2 = 0
#         for i in v[:30]:
#             points = np.load(os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc", lines[i]))
#             point_cloud = o3d.geometry.PointCloud()
#             point_cloud.points = o3d.utility.Vector3dVector(points)
#             # o3d.visualization.draw_geometries([point_cloud])
#             curv, average = curv_avg(points, 15)
#             avg_noise = noise(points, 15)
#             avg1+=average
#             avg2+=avg_noise
#         print(avg1/no_samples)
#         print(avg2/no_samples)
#     else:
#         print("x")





"curv and noise shapenet34"

# # file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55\train.txt"
# # file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-34\train.txt"
# file_path = "ShapeNet55/ShapeNet-34/train.txt"

# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}

# print(len(repeated_indices.keys()))

# for k,v in repeated_indices.items():
#     print(k)
#     if len(v) > 30:
#         avg1 = 0
#         avg2 = 0
#         for i in v[:30]:
#             # points = np.load(os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc", lines[i]))
#             points = np.load(os.path.join("ShapeNet55/shapenet_pc", lines[i]))
#             point_cloud = o3d.geometry.PointCloud()
#             point_cloud.points = o3d.utility.Vector3dVector(points)
#             # o3d.visualization.draw_geometries([point_cloud])
#             curv, average = curv_avg(points, 15)
#             avg_noise = noise(points, 15)
#             avg1+=average
#             avg2+=avg_noise
#         print(avg1/no_samples)
#         print(avg2/no_samples)
#     else:
#         print("x")




"curv and noise mvp"

# with h5py.File('data/MVP_complete_pcs/mvp_complete_train.h5', 'r') as f:
#     complete = f['complete_pcds'][:]

# print(complete.shape)


# avg1 = 0
# avg2 = 0
# for i in range(complete.shape[0]):
#     print(i)
#     points = complete[i]
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw_geometries([point_cloud])
#     curv, average = curv_avg(points.astype(np.float64), 15)
#     avg_noise = noise(points, 15)
#     avg1+=average
#     avg2+=avg_noise
# print(avg1/complete.shape[0])
# print(avg2/complete.shape[0])














"topo pcn"



# parent_directory_path = "/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code/odg/ODGNet/data/PCN/train/complete"
# # parent_directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete"

# subdirectories = [os.path.join(parent_directory_path, sub_dir) for sub_dir in os.listdir(parent_directory_path) if os.path.isdir(os.path.join(parent_directory_path, sub_dir))]
# for sub_dir in subdirectories:
#     print("x")
#     directory_path = sub_dir
#     all_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
#     selected_files = random.sample(all_files, no_samples)
#     s0 = 0
#     s1 = 0
#     for s in selected_files:
#         point_cloud = o3d.io.read_point_cloud(s)
#         data = np.asarray(point_cloud.points)
#         random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
#         data = data[random_indices]
#         rips = Rips()
#         dgms = rips.fit_transform(data)
#         H0_dgm = dgms[0]
#         H1_dgm = dgms[1]
#         # print(H0_dgm)
#         s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#         s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#         # print(s0, s1)
#     print(sub_dir)
#     avg0 = s0/no_samples
#     avg1 = s1/no_samples
#     print(avg0, avg1)



"topo shapenet55"


# file_path = r"/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/train.txt"
# # file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55\train.txt"

# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}

# # for k,v in repeated_indices.items():
# #     print(k)
# #     if len(v) > 30:
# #         s0 = 0
# #         s1 = 0
# #         for i in v[:30]:
# #             data = np.load(os.path.join("/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/shapenet_pc", lines[i]))
# #             random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
# #             data = data[random_indices]
# #             rips = Rips()
# #             dgms = rips.fit_transform(data)
# #             H0_dgm = dgms[0]
# #             H1_dgm = dgms[1]
# #             # print(H0_dgm)
# #             s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
# #             s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
# #             # print(s0, s1)
# #         avg0 = s0/no_samples
# #         avg1 = s1/no_samples
# #         print(avg0, avg1)
# #     else:
# #         print("x")


# # for selected folders only

# selected_keys = ['03624134', '04099429', '04090263', '03636649']
# new_dict = {key: repeated_indices[key] for key in selected_keys if key in repeated_indices}

# for k,v in new_dict.items():
#     print(k)
#     if len(v) > 30:
#         s0 = 0
#         s1 = 0
#         for i in v[:30]:
#             data = np.load(os.path.join("/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/shapenet_pc", lines[i]))
#             random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
#             data = data[random_indices]
#             rips = Rips()
#             dgms = rips.fit_transform(data)
#             H0_dgm = dgms[0]
#             H1_dgm = dgms[1]
#             # print(H0_dgm)
#             s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#             s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#             # print(s0, s1)
#         avg0 = s0/no_samples
#         avg1 = s1/no_samples
#         print(avg0, avg1)
#     else:
#         print("x")




"topo ours"

 
# npy_folder_path = "/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/new/Dutch" 
# # npy_folder_path = 'data/Dutch/easy_0.8'      

# complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")


# np.random.shuffle(complete)
# print(complete.shape)


# s0 = 0
# s1 = 0
# for i in range(no_samples):
#     data = complete[i]
#     rips = Rips()
#     dgms = rips.fit_transform(data)
#     H0_dgm = dgms[0]
#     H1_dgm = dgms[1]
#     # print(H0_dgm)
#     s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#     s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#     # print(s0, s1)
# avg0 = s0/no_samples
# avg1 = s1/no_samples
# print(avg0, avg1)




"topo mvp"

 

# with h5py.File('data/MVP_complete_pcs/mvp_complete_train.h5', 'r') as f:
#     complete = f['complete_pcds'][:]

# print(complete.shape)


# s0 = 0
# s1 = 0
# for i in range(complete.shape[0]):
#     print(i)
#     data = complete[i]
#     rips = Rips()
#     dgms = rips.fit_transform(data)
#     H0_dgm = dgms[0]
#     H1_dgm = dgms[1]
#     # print(H0_dgm)
#     s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#     s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#     # print(s0, s1)
# avg0 = s0/complete.shape[0]
# avg1 = s1/complete.shape[0]
# print(avg0, avg1)










"topo shapenet34"

 

# # file_path = r"/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/train.txt"
# # file_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-34\train.txt"
# file_path = "ShapeNet55/ShapeNet-34/train.txt"

# with open(file_path, 'r') as file:
#     lines = file.readlines()
# lines = [line.strip() for line in lines]
# first_8_letters = [word[:8] for word in lines]
# indices_dict = defaultdict(list)
# for idx, element in enumerate(first_8_letters):
#     indices_dict[element].append(idx)
# repeated_indices = {element: idx_list for element, idx_list in indices_dict.items() if len(idx_list) > 1}

# print(len(repeated_indices.keys()))

# start_time = time.time()
# ii=0
# # for k, v in list(repeated_indices.items())[0:5]:
# for k,v in repeated_indices.items():
#     print(ii)
#     ii+=1
#     print(k)
#     if len(v) > 30:
#         s0 = 0
#         s1 = 0
#         for i in v[:30]:
#             # print(i)
#             # data = np.load(os.path.join("/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/shapenet_pc", lines[i]))
#             # data = np.load(os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc", lines[i]))
#             data = np.load(os.path.join("ShapeNet55/shapenet_pc", lines[i]))
#             random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
#             data = data[random_indices]
#             rips = Rips()
#             dgms = rips.fit_transform(data)
#             H0_dgm = dgms[0]
#             H1_dgm = dgms[1]
#             # print(H0_dgm)
#             s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#             s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#             # print(s0, s1)
#         avg0 = s0/no_samples
#         avg1 = s1/no_samples
#         print(avg0, avg1)
#     else:
#         print("x")
# end_time = time.time()
# print(f"\nExecution time: {end_time - start_time:.2f} seconds")

# for selected folders only

# selected_keys = ['03624134', '04099429', '04090263', '03636649']
# new_dict = {key: repeated_indices[key] for key in selected_keys if key in repeated_indices}

# for k,v in new_dict.items():
#     print(k)
#     if len(v) > 30:
#         s0 = 0
#         s1 = 0
#         for i in v[:30]:
#             data = np.load(os.path.join("/home/scratch/prashant/martini-scratch2/temp/scratch2/prashant/stuti/code-stuti/ShapeNet55/shapenet_pc", lines[i]))
#             random_indices = np.random.choice(data.shape[0], size=5000, replace=False)
#             data = data[random_indices]
#             rips = Rips()
#             dgms = rips.fit_transform(data)
#             H0_dgm = dgms[0]
#             H1_dgm = dgms[1]
#             # print(H0_dgm)
#             s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
#             s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
#             # print(s0, s1)
#         avg0 = s0/no_samples
#         avg1 = s1/no_samples
#         print(avg0, avg1)
#     else:
#         print("x")








"topo KITTI cars"


folder_path = "KITTI/cars/"

pcd_files = [f for f in os.listdir(folder_path) if f.endswith(".pcd") and f.startswith("frame")]
pcd_files.sort()  
# print(pcd_files)

s0 = 0
s1 = 0
for pcd_file in pcd_files[0:200]:
    file_path = os.path.join(folder_path, pcd_file)
    print(f"Showing: {file_path}")

    pcd = o3d.io.read_point_cloud(file_path)
    # o3d.visualization.draw_geometries([pcd])
    data = np.asarray(pcd.points)
    # print(data.shape)
    rips = Rips()
    dgms = rips.fit_transform(data)
    H0_dgm = dgms[0]
    H1_dgm = dgms[1]
    # print(H0_dgm)
    s0+=np.sum(H0_dgm[1:-1][:,1] - H0_dgm[1:-1][:,0])/len(H0_dgm[1:-1])
    s1+=np.sum(H1_dgm[:,1] - H1_dgm[:,0])/len(H1_dgm)
    # print(s0, s1)
avg0 = s0/200
avg1 = s1/200
print(avg0, avg1)









# points = complete[7]
# voxel(points, 0.1)
# point_cloud = o3d.io.read_point_cloud(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete\02691156\1a04e3eab45ca15dd86060f189eb133.pcd")
# points = np.asarray(point_cloud.points)
# voxel(points, 0.01)






# points = np.load(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc\02691156-1a04e3eab45ca15dd86060f189eb133.npy")
# print(points.shape[0])
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([point_cloud])
# points= torch.from_numpy(points)
# points = points.to("cuda")

# partial, _ = seprate_point_cloud(points, points.shape[0], [int(points.shape[0] * 1/4) , int(points.shape[0] * 3/4)], fixed_points = None)
# print(partial.shape)


    
