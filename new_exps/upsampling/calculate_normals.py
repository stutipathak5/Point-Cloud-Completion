import open3d as o3d
import os
import glob


# def calculate_normals_and_save(ply_file):
#     pcd = o3d.io.read_point_cloud(ply_file)
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
#     o3d.io.write_point_cloud(ply_file, pcd)
#     print(f"Calculated normals and saved {ply_file} successfully!")
# ply_dir = r"C:\Users\spathak\Downloads\shapenet"
# ply_files = glob.glob(os.path.join(ply_dir, "*.ply"))
# for ply_file in ply_files:
#     calculate_normals_and_save(ply_file)


def calculate_normals_and_save(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Calculated normals and saved {ply_file} successfully!")

ply_dir = r"C:\Users\spathak\Downloads\PCC\new_exps\data\realpc"
for root, _, files in os.walk(ply_dir):
    for filename in files:
        if filename.endswith(".ply"):
            ply_file = os.path.join(root, filename)
            calculate_normals_and_save(ply_file)