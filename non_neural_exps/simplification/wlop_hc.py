
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Point_set_processing_3 import *
import open3d as o3d
import numpy as np
import os
import pytorch3d.loss as pp
import torch

def cham(pc1, pc2):
    pcd1_tensor = torch.tensor(pc1, device="cuda")
    pcd2_tensor = torch.tensor(pc2, device="cuda")

    cd_loss, _ = pp.chamfer_distance(pcd1_tensor.unsqueeze(0), pcd2_tensor.unsqueeze(0))

    return cd_loss.item()


def find_files_ending_with(folder, suffix):
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files

folder_path = r"C:\Users\spathak\Downloads\PCC\new_exps\upsampling\results"
suffix = "_down.ply"
down_ply_files = find_files_ending_with(folder_path, suffix)

for i in down_ply_files:

    # wlop gives simp very similar to random_simp, is good for noisy datasets

    # print("Running read_xyz_points...")
    points = Point_set_3(i)
    # print(points.size(), "points read")

    # print("Running wlop_simplify_and_regularize_point_set...")
    # wlop_point_set = Point_set_3()
    # wlop_simplify_and_regularize_point_set(points, wlop_point_set, select_percentage = 30, require_uniform_sampling = False)  # Output
    # print("Output of WLOP has", wlop_point_set.size(), "points")

    # print("Writing to ply")
    # wlop_point_set.write(i.replace("_down.ply", "_wlop_simp.ply"))


    pcd1 = o3d.io.read_point_cloud(i)
    pcd2 = o3d.io.read_point_cloud(i.replace("_down.ply", "_wlop_simp.ply"))
    cd = cham(np.asarray(pcd1.points), np.asarray(pcd2.points))

    print(f"Chamfer Distance for {os.path.basename(i).rsplit('_', 1)[0]}: {cd}")


# # hierarchy doesnt give user specified simplified pc size, so one has to adjust size and max_variation to get desired size (below done to get approx 0.03 of drag_vrip)

# print("Running read_xyz_points...")
# points = Point_set_3(i)
# print(points.size(), "points read")

# print("Running hierarchy_simplify_point_set...")
# hierarchy_simplify_point_set(points, size = 1000, maximum_variation = 0.01)
# print(points.size(), "point(s) remaining,", points.garbage_size(),
#       "point(s) removed")

# print("Writing to ply")
# points.write(i.replace("_down.ply", "_hc_simp.ply"))