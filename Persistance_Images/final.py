import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips

# Path to the PCD file
pcd_file_path = r"C:\Users\spathak\Downloads\PCC\CVPR submission\animations\9\1028af04-1078-4aaf-add3-4b645c2b183a.ply"

# Read the PCD file
pcd = o3d.io.read_point_cloud(pcd_file_path)

# Convert to NumPy array
points = np.asarray(pcd.points)


points=np.load(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc\03797390-ef24c302911bcde6ea6ff2182dd34668.npy")

data = points

print(data.shape)

downsampled_indices = np.random.choice(points.shape[0], 5000, replace=False)
downsampled_points = points[downsampled_indices]
data = downsampled_points

rips = Rips()
dgms = rips.fit_transform(data)
H0_dgm = dgms[0]
H1_dgm = dgms[1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', s=0.1)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(data[:,0], data[:,1], s=4)
plt.title("Scatter plot of noisy data with some circles")
plt.show()

plt.subplot(122)
rips.plot(dgms, legend=False, show=False)
plt.title("Persistence diagram of $H_0$ and $H_1$")
plt.show()
