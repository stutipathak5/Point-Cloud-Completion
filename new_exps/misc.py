import os
from collections import defaultdict

def get_one_file_per_section(folder_path):
    file_dict = defaultdict(list)

    # Get all .ply files
    for file in os.listdir(folder_path):
        if file.endswith(".ply"):
            section_key = file[:7]  # First 9 characters as key
            file_dict[section_key].append(os.path.join(folder_path, file))

    # Select one file per section
    selected_files = [files[0] for files in file_dict.values()]

    return selected_files

# Example usage
folder_path = r"C:\Users\spathak\Downloads\PCC\new_exps\data\shapenet"

selected_files = get_one_file_per_section(folder_path)
print(len(selected_files))

# Print selected file paths
# for file in selected_files:
#     print(file)
# print(selected_files)

# import os

# def list_files_with_prefix(folder_path, prefix):
#     matching_files = [f for f in os.listdir(folder_path) if f.startswith(prefix)]
    
#     for file in matching_files:
#         print(os.path.join(folder_path, file))

# # Example usage
# folder_path = r"C:\Users\spathak\Downloads\PCC\new_exps\data\shapenet" # Change this to your folder path
# prefix = "04379243-"  # Change this to the 9-letter prefix you're looking for

# list_files_with_prefix(folder_path, prefix)


# C:\Users\spathak\Downloads\PCC\new_exps\data\shapenet\04379243-ffe4383cff6d000a3628187d1bb97b92.ply
# C:\Users\spathak\Downloads\PCC\new_exps\data\shapenet\02747177-e349b1d60823e089da15415f29346bc8.ply





# Extract the last part after the hyphen and before .ply
extracted_parts = [os.path.splitext(os.path.basename(file))[0] for file in selected_files]


# print(extracted_parts)
print(len(extracted_parts))


import re

def read_values_and_compute_average(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r':\s*([0-9.e+-]+)', line)
            if match:
                value = float(match.group(1))
                values.append(value)
    if values:
        average_value = sum(values) / len(values)
    else:
        average_value = 0
    return average_value

file_path = r"C:\Users\spathak\Downloads\PCC\new_exps\misc.txt"
average_value = read_values_and_compute_average(file_path)

print(f"Average value: {average_value}")





# points = np.load(r"C:\Users\spathak\Downloads\pred.npy")

# # print(points.shape)

# for i in range(points.shape[0]):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[i])
#     o3d.visualization.draw_geometries([pcd])

# # Save the point cloud as a PLY file
# o3d.io.write_point_cloud(r"C:\Users\spathak\Downloads\pred.ply", pcd)



import numpy as np
import open3d as o3d

def find_files_ending_with(folder, suffix):
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files

def pcd_to_ply(pcd_file):
    # Read the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # Save the point cloud as a PLY file
    ply_file = pcd_file.replace(".pcd", ".ply")
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Saved {ply_file}")

folder_path = r"C:\Users\spathak\Downloads\for_figure_5\for_figure_5\6172f084\input-"


pcd_files = find_files_ending_with(folder_path, ".pcd")
for pcd_file in pcd_files:
    pcd_to_ply(pcd_file)