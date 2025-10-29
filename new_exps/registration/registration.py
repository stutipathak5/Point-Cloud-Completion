import open3d as o3d
import numpy as np
import copy



def remove_farthest_points(pcd, n):
    points = np.asarray(pcd.points)
    viewpoint = np.random.rand(3) * (points.max(axis=0) - points.min(axis=0)) + points.min(axis=0)
    distances = np.linalg.norm(points - viewpoint, axis=1)
    farthest_indices = np.argsort(distances)[-n:]
    mask = np.ones(len(points), dtype=bool)
    mask[farthest_indices] = False
    filtered_points = points[mask]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return new_pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 0]) # source is black in color
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    # draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def remove_noise(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def scale_point_cloud_to_01(pcd):
    points = np.asarray(pcd.points)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    scale_factors = max_vals - min_vals
    scale_factors = np.maximum(scale_factors, 1e-6)
    scaled_points = (points - min_vals) / scale_factors
    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)

    return scaled_pcd


def sample(num_samples, pcd):
    points = np.asarray(pcd.points)
    sampled_indices = np.random.choice(points.shape[0], num_samples, replace=False)
    points = points[sampled_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def random_translation(point_cloud, max_translation=1.0):
    # Random translation: Uniform random shift along each axis
    translation = np.random.uniform(-max_translation, max_translation, 3)
    point_cloud.translate(translation)
    return point_cloud

def random_rotation(point_cloud):
    # Random rotation: Random axis and angle for rotation
    angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
    axis = np.random.uniform(-1, 1, 3)  # Random axis for rotation
    axis /= np.linalg.norm(axis)  # Normalize to get a unit vector
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(angle * axis)
    point_cloud.rotate(R, center=(0, 0, 0))  # Rotate around the origin
    return point_cloud

def random_scaling(point_cloud, scale_range=(0.5, 2.0)):
    # Random scaling: Uniform random scaling factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    point_cloud.scale(scale_factor, center=(0, 0, 0))  # Scale around the origin
    return point_cloud

def random_transform_point_cloud(point_cloud):
    # Apply random transformations
    point_cloud = random_translation(point_cloud)
    point_cloud = random_rotation(point_cloud)
    point_cloud = random_scaling(point_cloud)
    return point_cloud




voxel_size = 0.01



"shapenet"

points = np.load(r"registration\temp_data\1a04e3eab45ca15dd86060f189eb133.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = sample(5000, pcd)
print(len(pcd.points))

pcd1 = sample(2500, pcd)
pcd2 = sample(2500, pcd)
# pcd1, pcd2 = pcd, pcd

source = remove_farthest_points(pcd1, 1)
print(len(source.points))

target = remove_farthest_points(pcd2, 1)
print(len(target.points))


target = random_transform_point_cloud(target)
source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([target_temp, source_temp])





source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
result_icp = refine_registration(source, target, result_ransac, voxel_size)
print(result_ransac)
print(result_icp)
source.transform(result_icp.transformation)
o3d.visualization.draw_geometries([source + target])





"ours"


pcd = o3d.io.read_point_cloud(r"C:\Users\spathak\Downloads\2e5bd22a-477e-4a13-8262-48d68656f288.pcd")
pcd = scale_point_cloud_to_01(pcd)

print(len(pcd.points))

pcd1 = sample(2500, pcd)
pcd2 = sample(2500, pcd)
# pcd1, pcd2 = pcd, pcd

source = remove_farthest_points(pcd1, 1)
print(len(source.points))

target = remove_farthest_points(pcd2, 1)
print(len(target.points))


target = random_transform_point_cloud(target)
source_temp = copy.deepcopy(source)
target_temp = copy.deepcopy(target)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([target_temp, source_temp])


source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
result_icp = refine_registration(source, target, result_ransac, voxel_size)
print(result_ransac)
print(result_icp)
source.transform(result_icp.transformation)
o3d.visualization.draw_geometries([source + target])










import open3d as o3d
import numpy as np

# # Load PCD file
# pcd = o3d.io.read_point_cloud(r"C:\Users\spathak\Downloads\2e5bd22a-477e-4a13-8262-48d68656f288.pcd")

# # Save as PLY file
# o3d.io.write_point_cloud(r"C:\Users\spathak\Downloads\2e5bd22a-477e-4a13-8262-48d68656f288.ply", pcd)
# print("Converted PCD to PLY successfully!")




# points = np.load(r"C:\Users\spathak\Downloads\shapenet_pc\02691156-10155655850468db78d106ce0a280f87.npy")  
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points) 

# pcd = sample(5000, pcd)
# o3d.io.write_point_cloud(r"C:\Users\spathak\Downloads\shapenet\02691156-10155655850468db78d106ce0a280f87.ply", pcd)
# print("Converted NPY to PLY successfully!")


import open3d as o3d
import numpy as np
import os
import glob

# def convert_npy_to_ply(npy_file, ply_file):
#     points = np.load(npy_file)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.io.write_point_cloud(ply_file, pcd)
#     print(f"Converted {npy_file} to {ply_file} successfully!")

# # Directory paths
# npy_dir = r"C:\Users\spathak\Downloads\shapenet_pc"
# ply_dir = r"C:\Users\spathak\Downloads\shapenet"

# # Ensure the output directory exists
# os.makedirs(ply_dir, exist_ok=True)

# # Get all .npy files in the input directory
# npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))

# # Convert each .npy file to .ply
# for npy_file in npy_files:
#     file_name = os.path.splitext(os.path.basename(npy_file))[0]
#     ply_file = os.path.join(ply_dir, f"{file_name}.ply")
#     convert_npy_to_ply(npy_file, ply_file)


# pcd = o3d.io.read_point_cloud(r"C:\Users\spathak\Downloads\shapenet\02958343-9757fd5be93ee0fc82b157e7120744ea.ply")

# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
# o3d.visualization.draw_geometries([pcd],
#                                   point_show_normal=True)
# print(np.asarray(pcd.normals).shape)




















