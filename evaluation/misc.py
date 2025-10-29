import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


point_cloud = o3d.io.read_point_cloud(r"//datanasop3mech/ProjectData/3_phd/Stuti/PCC&PSS/Code/ODGNet/data/PCN/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133/07.pcd")

colors = np.zeros((len(point_cloud.points), 3))  
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud with larger points
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)
opt = vis.get_render_option()
opt.point_size = 20.0  # Adjust the point size to make them appear more circular
vis.run()
vis.destroy_window()


points = np.asarray(point_cloud.points)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o') 

ax.set_axis_off()

plt.show()