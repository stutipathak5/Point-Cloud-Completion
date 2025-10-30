import open3d as o3d
import numpy as np
import os
import torch
import pytorch3d.loss as pp



def cham(pc1, pc2):
    pcd1_tensor = torch.tensor(pc1, device="cuda")
    pcd2_tensor = torch.tensor(pc2, device="cuda")

    cd_loss, _ = pp.chamfer_distance(pcd1_tensor.unsqueeze(0), pcd2_tensor.unsqueeze(0))

    return cd_loss.item()



def chamfer_distance(pcd1, pcd2):
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)

    def compute_distance(pcd_from, pcd_to_tree):
        distances = []
        for point in pcd_from.points:
            [_, idx, d] = pcd_to_tree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(d[0]))
        return np.mean(distances)

    dist1 = compute_distance(pcd1, pcd2_tree)
    dist2 = compute_distance(pcd2, pcd1_tree)
    return dist1 + dist2

def find_files_ending_with(folder, suffix):
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files

folder_path = r"C:\Users\spathak\Downloads\PCC\new_exps\upsampling\results"
up_files = find_files_ending_with(folder_path, "_up.ply")
down_files = find_files_ending_with(folder_path, "_down.ply")

# Create a dictionary to map the base name to the file paths
file_dict = {}
for file in up_files + down_files:
    base_name = os.path.basename(file).rsplit('_', 1)[0]
    if base_name not in file_dict:
        file_dict[base_name] = {}
    if file.endswith("_up.ply"):
        file_dict[base_name]['up'] = file
    elif file.endswith("_down.ply"):
        file_dict[base_name]['down'] = file

# Loop through the pairs and compute Chamfer distance
for base_name, files in file_dict.items():
    if 'up' in files and 'down' in files:
        pcd1 = o3d.io.read_point_cloud(files['down'])
        pcd2 = o3d.io.read_point_cloud(files['up'])
        
        # Compute Chamfer distance
        # cd = chamfer_distance(pcd1, pcd2)
        # print(f"Chamfer Distance for {base_name}: {cd}")
        cd2 = cham(np.asarray(pcd1.points), np.asarray(pcd2.points))
        print(f"Chamfer Distance for {base_name}: {cd2}")































