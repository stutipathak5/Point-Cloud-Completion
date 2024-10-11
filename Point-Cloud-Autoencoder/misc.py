import numpy as np
# import h5py
import open3d as o3d
import pickle

# pc_array = np.load("data/chair_set.npy")

# print(pc_array.shape)

# for i in range(10):
#     print(pc_array[i].shape)
#     print(np.min(pc_array[i], axis=0))
#     print(np.max(pc_array[i], axis=0))
#     print(np.mean(pc_array[i], axis=0))


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load point cloud data from a NumPy file
# pc_array = np.load("data/chair_set.npy")

# # Print the shape of the point cloud array
# print(pc_array.shape)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot each point cloud on the same 3D plot
# for i in range(10):
#     point_cloud = pc_array[i]
#     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], label=f'Point Cloud {i+1}', s=1)

# # Set labels for the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Add a legend
# ax.legend()

# # Show the plot
# plt.show()



# import h5py
# import os

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d

# def load_h5(file_path, dataset_name):
#     with h5py.File(file_path, 'r') as hf:
#         array = hf[dataset_name][:]
#     return array

# npy_folder_path = '../data/Dutch/easy_0.8'

# def sample(point_cloud_array):
#     new_array = np.zeros((point_cloud_array.shape[0], 1024, 3))
#     for i in range(point_cloud_array.shape[0]):
#         indices = np.random.choice(point_cloud_array[i].shape[0], 1024, replace=False)
#         new_array[i] = point_cloud_array[i][indices, :]
#     return new_array

# complete = sample(load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete"))
# occl = sample(load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl"))
# non_sparse = sample(load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse"))
# uni_sparse = sample(load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse"))

# print(complete.shape)
# print(occl.shape)
# print(non_sparse.shape)
# print(uni_sparse.shape)


# concatenated_array = np.concatenate((complete, occl, non_sparse, uni_sparse), axis=0)


# def normalize(pc_array):
#     for i in range(pc_array.shape[0]):
#         ptcloud = pc_array[i]
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(ptcloud)
#         bbox_pcd = pcd.get_axis_aligned_bounding_box()
#         corners = bbox_pcd.get_box_points()
#         bbox = np.asarray(corners)
#         bbox[[3, 7]] = bbox[[7, 3]]
#         center = (bbox.min(0) + bbox.max(0)) / 2
#         bbox -= center
#         yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
#         rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
#         bbox = np.dot(bbox, rotation)
#         scale = bbox[3, 0] - bbox[0, 0]
#         bbox /= scale
#         ptcloud = np.dot(ptcloud - center, rotation) / scale
#         ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
#         pc_array[i] = ptcloud
#     return pc_array


# concatenated_array = normalize(concatenated_array)
# print(concatenated_array.shape)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot each point cloud on the same 3D plot
# for i in range(3):
#     point_cloud = concatenated_array[i]
#     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], label=f'Point Cloud {i+1}', s=1)

# # Set labels for the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Add a legend
# ax.legend()

# # Show the plot
# plt.show()


x = np.load(r"data/final_splits/SNCF/easy/splits/comp_tr.npy")

print(x.shape)

# indices = [6, 10, 15, 76]

# # Create a list to hold the Open3D PointCloud objects
# point_clouds = []

# # Translation offset
# offset = 10

# for i, idx in enumerate(indices):pds.pkl
#     # Convert the NumPy array to an Open3D PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(x[idx])
    
#     # Translate the point cloud along the x-axis
#     pcd.translate((i * offset, 0, 0))
    
#     # Optionally, you can set different colors for each point cloud
#     colors = np.random.rand(x[idx].shape[0], 3)  # Random colors
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     point_clouds.append(pcd)

# # Visualize the point clouds together
# o3d.visualization.draw_geometries(point_clouds)


import numpy as np
import hashlib
from topologylayer.nn import AlphaLayer
import torch

def hash_point_cloud(point_cloud):
    # Convert the point cloud to a string and hash it
    point_cloud_str = point_cloud.tostring()
    return hashlib.md5(point_cloud_str).hexdigest()

def find_duplicate_point_clouds(point_clouds):
    hash_dict = {}
    for i, pc in enumerate(point_clouds):
        pc_hash = hash_point_cloud(pc)
        if pc_hash in hash_dict:
            hash_dict[pc_hash].append(i)
        else:
            hash_dict[pc_hash] = [i]
    
    # Extract groups with more than one index
    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
    return duplicates

# Example usage
N = 10  # Number of point clouds
point_clouds = x

# Introduce some duplicates for testing
# point_clouds[1] = point_clouds[0]
# point_clouds[3] = point_clouds[2]
# point_clouds[5] = point_clouds[4]

duplicates = find_duplicate_point_clouds(point_clouds)
niters = 500
eps = 1e-3
stop_error = 1e-5
pds = [0 for _ in range(point_clouds.shape[0])]
print("Duplicate point clouds:")
for hash_val, indices in duplicates.items():
    print(f"Hash: {hash_val}, Indices: {indices}")
    layer = AlphaLayer(maxdim=1)
    pd = layer(torch.from_numpy(point_clouds[indices[0]]).float())
    # duplicates[hash_val] = (indices, pd[0])
    for i in indices:
        pds[i] = pd[0]

print(pds)

with open('pds.pkl', 'wb') as file:
    pickle.dump(pds, file)


with open('pds.pkl', 'rb') as file:
    pdss = pickle.load(file)

print(len(pdss))