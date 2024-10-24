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


def numpy_to_pcd(array, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    o3d.io.write_point_cloud(file_path, pcd)


ii = "difficult"


for i in ["Dutch", "SNCF", "Hungarian", "Chinese"]:
    os.makedirs("data_ours/" + ii + "/train/complete/" + i, exist_ok=True)
    os.makedirs("data_ours/" + ii + "/test/complete/" + i, exist_ok=True)
    os.makedirs("data_ours/" + ii + "/train/partial/" + i, exist_ok=True)
    os.makedirs("data_ours/" + ii + "/test/partial/" + i, exist_ok=True)

    comp_tr = np.load('../Point-Cloud-Autoencoder/data/final_splits/'+ i + "/" + ii + '/splits/comp_tr.npy')
    part_tr = np.load('../Point-Cloud-Autoencoder/data/final_splits/'+ i + "/" + ii + '/splits/part_tr.npy')
    comp_te = np.load('../Point-Cloud-Autoencoder/data/final_splits/'+ i + "/" + ii + '/splits/comp_te.npy')
    part_te = np.load('../Point-Cloud-Autoencoder/data/final_splits/'+ i + "/" + ii + '/splits/part_te.npy')


    print(comp_tr.shape)
    print(part_tr.shape)
    print(comp_te.shape)
    print(part_te.shape)


    duplicates = find_duplicate_indices(comp_tr)
    print(len(duplicates))
    for k, v in duplicates.items():
        unique_id = uuid.uuid4()
        numpy_to_pcd(v[1], "data_ours/" + ii + "/train/complete/" + i + "/" + str(unique_id) + ".pcd")
        os.makedirs("data_ours/" + ii + "/train/partial/" + i + "/" + str(unique_id), exist_ok=True)
        for j in v[0]:
            counter = str(j).zfill(4)
            numpy_to_pcd(part_tr[j], "data_ours/" + ii + "/train/partial/" + i + "/" + str(unique_id) + "/" + counter + ".pcd")


    duplicates = find_duplicate_indices(comp_te)
    print(len(duplicates))
    for k, v in duplicates.items():
        unique_id = uuid.uuid4()
        numpy_to_pcd(v[1], "data_ours/" + ii + "/test/complete/" + i + "/" + str(unique_id) + ".pcd")
        os.makedirs("data_ours/" + ii + "/test/partial/" + i + "/" + str(unique_id), exist_ok=True)
        for j in v[0]:
            counter = str(j).zfill(4)
            numpy_to_pcd(part_te[j], "data_ours/" + ii + "/test/partial/" + i + "/" + str(unique_id) + "/" + counter + ".pcd")

    
