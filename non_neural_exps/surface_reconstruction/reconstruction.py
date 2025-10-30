import open3d as o3d
import numpy as np
import pyvista as pv

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

import os

def find_files_ending_with(folder, suffix):
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files




def get_ss(mesh, i):

    # Convert to points and faces
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Convert faces to PyVista-compatible format (each face starts with the number of vertices, i.e., 3 for triangles)
    faces_pyvista = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    # Create PyVista mesh
    pyvista_mesh = pv.PolyData(points, faces_pyvista)

    # Create plotter
    plotter = pv.Plotter(off_screen=True)

    # Add mesh with grey color
    plotter.add_mesh(pyvista_mesh, color='grey')

    # Compute bounding box
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2

    # Compute bounding box size and camera distance
    bounding_box_size = np.linalg.norm(max_bound - min_bound)
    camera_distance = 2 * bounding_box_size  # You can adjust this multiplier if needed

    # Define multiple camera views
    camera_views = [
        {"name": "front", "position": center + [camera_distance, 0, 0], "focal_point": center, "viewup": [0, 0, 1]},
        {"name": "back", "position": center - [camera_distance, 0, 0], "focal_point": center, "viewup": [0, 0, 1]},
        {"name": "top", "position": center + [0, camera_distance, 0], "focal_point": center, "viewup": [0, 0, 1]},
        {"name": "bottom", "position": center - [0, camera_distance, 0], "focal_point": center, "viewup": [0, 0, -1]},
        {"name": "left", "position": center - [0, 0, camera_distance], "focal_point": center, "viewup": [0, 1, 0]},
        {"name": "right", "position": center + [0, 0, camera_distance], "focal_point": center, "viewup": [0, 1, 0]},
    ]


    mesh_name = os.path.basename(i).split('_')[0]
    output_folder = os.path.join("ss/0.15", mesh_name)
    os.makedirs(output_folder, exist_ok=True)


    # Loop through views and save screenshots
    for view in camera_views:
        plotter.camera_position = (view["position"], view["focal_point"], view["viewup"])
        plotter.camera.clipping_range = (0.01, 4 * camera_distance)  # Extend clipping range
        plotter.render()

        screenshot_filename = os.path.join(output_folder, f"{view['name']}.png")
        plotter.screenshot(screenshot_filename, window_size=[2560, 1440])
        # print(f"Saved {screenshot_filename}")

    plotter.close()






folder_path = r"C:\Users\spathak\Downloads\PCC\new_exps\upsampling\results"
suffix = "_down.ply"
down_ply_files = find_files_ending_with(folder_path, suffix)
# print(len(down_ply_files))
# down_ply_files.remove(r"C:\Users\spathak\Downloads\PCC\new_exps\upsampling\results\116fbb5a-a290-466b-85dd-fc904fc9c726_down.ply")
# down_ply_files.remove(r"C:\Users\spathak\Downloads\PCC\new_exps\upsampling\results\1a18a945-db8c-4a1c-9a51-7a699e7daacf_down.ply")
# print(len(down_ply_files))


for i in down_ply_files:
    pcd = o3d.io.read_point_cloud(i)

    # o3d.visualization.draw_geometries([pcd])
    # 0.05, 0.1, 0.15
    alpha = 0.15
    # print(f"alpha={alpha:.3f}")

    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        
        pcd1 = mesh.sample_points_uniformly(number_of_points=5000)


        from scipy.spatial.distance import directed_hausdorff

        def hausdorff_distance(pc1, pc2):
            """Compute Hausdorff Distance between two point clouds."""
            h1 = directed_hausdorff(pc1, pc2)[0]
            h2 = directed_hausdorff(pc2, pc1)[0]
            havg = (h1+h2)/2
            return h1, h2, havg

        # Compute Hausdorff Distance
        h1, h2, havg = hausdorff_distance(np.asarray(pcd.points), np.asarray(pcd1.points))
        # print("Hausdorff Distance:", h1, h2, havg)
        print(f"Hausdorff Distance for {os.path.basename(i).split('_')[0]}: {havg}")
    except Exception as e:
        print(f"Error for {os.path.basename(i).split('_')[0]}")
        continue
    get_ss(mesh, i)








# for i in down_ply_files:
#     pcd = o3d.io.read_point_cloud(i)

#     radii = [0.05, 0.1, 0.2, 0.4]

#     # o3d.visualization.draw_geometries([pcd])


#     try:

#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
#         mesh.compute_vertex_normals()
#         o3d.visualization.draw_geometries([pcd, mesh])
#         pcd1 = mesh.sample_points_uniformly(number_of_points=5000)


#         from scipy.spatial.distance import directed_hausdorff

#         def hausdorff_distance(pc1, pc2):
#             """Compute Hausdorff Distance between two point clouds."""
#             h1 = directed_hausdorff(pc1, pc2)[0]
#             h2 = directed_hausdorff(pc2, pc1)[0]
#             havg = (h1+h2)/2
#             return h1, h2, havg

#         # Compute Hausdorff Distance
#         h1, h2, havg = hausdorff_distance(np.asarray(pcd.points), np.asarray(pcd1.points))
#         # print("Hausdorff Distance:", h1, h2, havg)
#         print(f"Hausdorff Distance for {os.path.basename(i).split('_')[0]}: {havg}")
#     except Exception as e:
#         print(f"Error for {os.path.basename(i).split('_')[0]}")
#         continue

