import numpy as np
import os
import open3d as o3d

def list_folders_in_directory(directory_path):
    items = os.listdir(directory_path)
    folders = [os.path.join(directory_path, item) for item in items if os.path.isdir(os.path.join(directory_path, item))]
    return folders

def list_files_in_folders(folders):
    all_files = []
    for folder in folders:
        items = os.listdir(folder)
        files = [os.path.join(folder, item) for item in items if os.path.isfile(os.path.join(folder, item))]
        all_files.extend(files)
    return all_files


directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\complete"  # Replace with your folder path
folders = list_folders_in_directory(directory_path)
files = list_files_in_folders(folders)
# print(len(files))
# print(files)

final_array = np.zeros((len(files)*8, 5000, 3))                    # each complete .pcd has 8 corresponding partial .pcd
final_part_array = np.zeros((len(files)*8, 500, 3))
j=0
for i in files:
    components = i.split(os.sep)
    last_two_components = os.path.join(components[-2], components[-1][:-4])
    new_path = os.path.join(r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\PCN\train\partial", last_two_components)
    files_2 = list_files_in_folders([new_path])
    pcd = o3d.io.read_point_cloud(i)
    np_array = np.asarray(pcd.points)
    if np_array.shape[0] > 5000:
        indices = np.random.choice(np_array.shape[0], 5000, replace=False)  # each complete pcd is 16k points
        downsampled_np_array = np_array[indices]
    else:
        downsampled_np_array = np_array
    for ii in range(8):
        final_array[j*8+ii,:,:] = downsampled_np_array
        part = o3d.io.read_point_cloud(files_2[ii])
        part_array = np.asarray(part.points)
        if part_array.shape[0] > 500:
            indices = np.random.choice(part_array.shape[0], 500, replace=False)       # each partial pcd is around 1k points
            downsampled_part_array = part_array[indices]
        else:
            downsampled_part_array = part_array
        final_part_array[j*8+ii,:,:] = downsampled_part_array
    j+=1
        
print(final_array.shape)
print(final_part_array.shape)

# np.save("comp_tr.npy", final_array)
# np.save("part_tr.npy", final_part_array)

for i in range(final_array.shape[0]):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(final_array[i])
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(final_part_array[i])

    pcd2.translate((1, 0, 0))
    
    o3d.visualization.draw_geometries([pcd1, pcd2])




