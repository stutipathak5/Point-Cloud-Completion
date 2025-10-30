import torch
import pytorch3d
import plyfile
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import os
import  math
import time
import open3d as o3d


UP_RATIO = 2
UP_THRESHOLD = 0.93  # the smaller the threshold, the more points insert at each iteration, i.e., faster but less evenly distribution
KNN_NUM = 15 # the larger, the better, but the slower; assume no outlier
EDGE_SENSITIVITY = 4  # if need more points sampled around the edge, use EDGE_SENSITIVITY = 1, 2, 3, 4...


def read_ply(file):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)


    points = points.transpose(1, 0)
    sampled_indices = np.random.choice(points.shape[0], 5000, replace=False)
    points = points[sampled_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Extract (x, y, z)
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:])  # Extract (nx, ny, nz)
    o3d.io.write_point_cloud("C:/Users/spathak/Downloads/PCC/new_exps/upsampling/results/"+ pc_name + "_down.ply", pcd)


    return points


def save_ply(filename, points, colors=None, normals=None, binary=True):
    """
    save 3D/2D points to ply file
    Args:
        points (numpy array): (N,2or3)
        colors (numpy uint8 array): (N, 3or4)
    """
    assert(points.ndim == 2)
    if points.shape[-1] == 2:
        points = np.concatenate(
            [points, np.zeros_like(points)[:, :1]], axis=-1)


    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr


    if normals is not None:
        assert(normals.ndim == 2)
        if normals.shape[-1] == 2:
            normals = np.concatenate(
                [normals, np.zeros_like(normals)[:, :1]], axis=-1)
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr


    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr


    vertex_all = np.empty(num_vertex, dtype=desc)


    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]


    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]


    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]


    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=(not binary))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)




def return_nn(args):
    pass


def normalize_normal_tensors(normals):
    normals = normals.squeeze(0).cpu().numpy()
    for i in range(len(normals)):
        norm = np.linalg.norm(normals[i])
        normals[i] = normals[i]/norm


    normals = torch.from_numpy(normals).unsqueeze(0).cuda()
    return  normals




def base_point_selection(query, query_normal, neighbor_pt, neighbor_pt_normals, edge_sensitivity = torch.tensor(EDGE_SENSITIVITY).cuda()):
    query = query.squeeze(0)
    query_normal = query_normal.squeeze(0)
    neighbor_pt = neighbor_pt.squeeze(0)
    neighbor_pt_normals = neighbor_pt_normals.squeeze(0)


    best_dist2 = torch.tensor(-10).cuda()
    v = query


    for i in range(len(neighbor_pt)):
        t = neighbor_pt[i]
        tm = neighbor_pt_normals[i]
        vm = query_normal


        diff_v_t = t - v
        mid_point = v + (diff_v_t * 0.5)


        dot_product = torch.pow( (2.0 - torch.dot(vm, tm.to(torch.float64)) ), edge_sensitivity)
        


        diff_t_mid = mid_point - t
        project_t = diff_t_mid * tm
        # min_dist2 = diff_t_mid.squared_length() - project_t * project_t
        min_dist2 = torch.norm(diff_t_mid, 2) - torch.dot(project_t, project_t)


        for j in range(len(neighbor_pt)):
            diff_s_mid = mid_point - neighbor_pt[j]
            project_s = diff_s_mid * neighbor_pt_normals[j]
            proj_min2 = torch.norm(diff_s_mid, 2) - torch.dot(project_s, project_s)


            if (proj_min2 < min_dist2):
                min_dist2 = proj_min2


        min_dist2 *= dot_product


        if min_dist2 > best_dist2:
            best_dist2 = min_dist2
            output_base_index = i


    return  best_dist2, neighbor_pt[i], neighbor_pt_normals[i]






def compute_neighbor_points_and_normals(points, normals):
    pt_num = len(points[0])
    knn_k = KNN_NUM


    # obtain normals from index
    dist, idx, nn = knn_points(points, points, K=(knn_k+1), return_sorted=True)


    k_nearest_neighbors = torch.empty(1, pt_num, knn_k, 3).cuda()
    k_nearest_neighbors_normals = torch.empty(1, pt_num, knn_k, 3).cuda()
    for i in range(len(points.squeeze(0))):
        for k in range(knn_k):
            k_nearest_neighbors_normals[0, i, k,:] = normals[0, idx[0, i, k+1], :]
            k_nearest_neighbors[0, i, k, :] = points[0, idx[0, i, k+1], :]




    return k_nearest_neighbors, k_nearest_neighbors_normals




def upsample(infile, output_file):
    knn_k = KNN_NUM
    loaded = read_ply(infile)


    input_num = len(loaded)
    target_num = input_num * UP_RATIO
    print(input_num, target_num)


    loaded = torch.from_numpy(loaded).unsqueeze(0).cuda()


    points, normals = torch.split(loaded, (3,3), dim=-1)
    normals = normalize_normal_tensors(normals)




    k_nearest_neighbors, k_nearest_neighbors_normals = compute_neighbor_points_and_normals(points, normals)




    # define empirical parameters
    sharpness_angle = 45
    cos_sigma = sharpness_angle / 180.0 * math.pi
    sharpness_bandwidth = pow(max(1e-8, 1.0 - cos_sigma), 2) # only for projection
    edge_sensitivity = torch.tensor(EDGE_SENSITIVITY).cuda()


    sum_density = torch.tensor(0.0).cuda()
    count_density = torch.tensor(1).cuda()
    max_iter_time = 20


    density_pass_threshold = 0.0


    rich_point_set = points.clone()
    rich_point_set_normals = normals.clone()


    # estimate density threshold for the first time
    for iter in range(max_iter_time):
        start_time = time.time()
        count_new_insert = 0
        current_size = len(rich_point_set.squeeze(0))
        is_dense_enough = torch.zeros(current_size, dtype=torch.bool)


        print("\n iteration: %s, current size %s" % (iter, current_size))


        # recompute neighborhood every iteration?
        if iter > 0:
            k_nearest_neighbors, k_nearest_neighbors_normals = compute_neighbor_points_and_normals(rich_point_set, rich_point_set_normals)


            elapsed_time = time.time() - start_time
            print(time.strftime("neighborhood time: %H:%M:%S", time.gmtime(elapsed_time)))
            start_time = time.time()




        # do one pass to estimate the stopping threshold
        if iter == 0:
            for i in range(len(points.squeeze(0))):
                density2, _, _ = base_point_selection(points[0, i], normals[0, i], k_nearest_neighbors[0, i], k_nearest_neighbors_normals[0, i], edge_sensitivity)


                if(density2 < 0):
                    continue


                sum_density += torch.sqrt(density2)
                count_density = count_density + 1


        density_pass_threshold = (sum_density/ count_density) * torch.tensor(UP_THRESHOLD).cuda()
        density_pass_threshold2 = density_pass_threshold * density_pass_threshold




        elapsed_time = time.time() - start_time
        print(time.strftime("estimate threshold time: %H:%M:%S", time.gmtime(elapsed_time)))
        start_time = time.time()




        # insert new points until all the points' density pass the threshold
        for loop in range(1): ## to-do: dynamic neighborhood or update the neighborhood for new points, more efficient
            # print("outer iteration: %s, inner loop: %s, current size %s" % (iter, loop, len(rich_point_set.squeeze(0))))
            count_new_insert = 0
            for i in range(len(rich_point_set.squeeze(0))):
                if is_dense_enough[i]:
                    continue


                if len(rich_point_set.squeeze(0)) >= target_num:
                    continue


                father_v = rich_point_set[:, i]
                father_vn = rich_point_set_normals[:, i]


                density2, mother_v, mother_vn = base_point_selection(father_v, father_vn, k_nearest_neighbors[0, i],
                                                            k_nearest_neighbors_normals[0, i], edge_sensitivity)


                if density2 < density_pass_threshold2:
                    is_dense_enough[i] = True
                    continue


                count_new_insert += 1


                new_v = father_v + ((mother_v - father_v) * torch.tensor(0.5).cuda())
                new_vn = (mother_vn + father_vn) * torch.tensor(0.5).cuda()
                # is_dense_enough = torch.cat((is_dense_enough, torch.tensor(1))) ## not so good solution
                is_dense_enough = torch.cat((is_dense_enough, torch.ones(1, dtype=torch.bool)), dim=0)  ## to-do: dynamic neighborhood or update the neighborhood for new points


                rich_point_set = torch.cat((rich_point_set, new_v.unsqueeze(0)), dim=1)
                rich_point_set_normals = torch.cat((rich_point_set_normals, new_vn.unsqueeze(0)), dim=1)




        if (count_new_insert == 0) or len(rich_point_set.squeeze(0)) >= target_num:
            break


        density_pass_threshold2 * 0.96


        elapsed_time = time.time() - start_time
        print(time.strftime("one loop insertion time: %H:%M:%S", time.gmtime(elapsed_time)))


    print("finish point number: " + str(len(rich_point_set.squeeze(0))))
    save_ply(output_file, rich_point_set.squeeze(0).cpu().numpy(), normals=rich_point_set_normals.squeeze(0).cpu().numpy())

# xx = ['03001627-1006be65e7bc937e9141f9b58470d646', '03046257-10312d9d07db5d9a4159aeb47682f2cb', '03085013-13c3acba1f43836a3123e2af297efed8', '03207941-1418f648e32f1921df3a1b0d597ce76e', '03211117-1063cfe209bdaeb340ff33d80c1d7d1e', '03261776-1757fe64e76a9630fc176230c2f2d294', '03325088-103f0eb370d64fe532149015b3e64c', '03337140-109a19840a8fbc774c3aee8e9d3a6ffa', '03467517-10158b6f14c2700871e863c034067817', '03513137-10dee7587a785cbfe22dbe8dd678fde8', '03593526-10af6bdfd126209faaf0ad030fc37d94', '03624134-102982a2159226c2cc34b900bb2492e', '03636649-101d0e7dbd07d8247dfd6bf7196ba84d', '03642806-10f18b49ae496b0109eaabd919821b8', '03691459-101354f9d8dede686f7b08d9de913afe', '03710193-10e1051cbe10626e30a706157956b491', '03759954-109f1779a6993782f6426d6e3b63c5ce', '03761084-112624e40e96e5654a44340bf227e40', '03790512-103e3c517cd759657395d58407f193ba', '03797390-1038e4eac0e18dcce02ae6d2a21d494a', '03928116-13394ca47c89f91525a3aaf903a41c90', '03938244-121f2fa5ec7a7db47df4feb40f08ca17', '03948459-10640377f4eb9ecdadceecd3bc8bde14', '03991062-10433e5bd8fa2a337b00c7b93209c459', '04004475-11ff4341475b3731c39a00da16746b95', '04074963-1941c37c6db30e481ef53acb6e05e27a', '04099429-15474cf9caa757a528eba1f0b7744e9', '04225987-10b81426bea7f98b5e38a8a610bb08f5', '04256520-1037fd31d12178d396f164a988ef37cc', '04330267-108f01e1b4fbab56557991c690a01e0', '04379243-ffe4383cff6d000a3628187d1bb97b92', '04401088-1049bf1611874c9cf0c2cf8583536651', '04460130-10312d9d07db5d9a4159aeb47682f2cb', '04468005-108baba96ac2f9be5f73a346378dad81', '04530566-10212c1a94915e146fc883a34ed13b89']

# for i in xx:
#     print(i)


if __name__ == '__main__':
    # file_name = "0000001400_000_part_out_EDGE0"
    pc_name = "02992529-1101db09207b39c244f01fc4278d10c1"
    input_path = "C:/Users/spathak/Downloads/PCC/new_exps/data/shapenet/" + pc_name + ".ply"
    output_path = "C:/Users/spathak/Downloads/PCC/new_exps/upsampling/results/"+ pc_name + "_up.ply"

    # pc_name = "116fbb5a-a290-466b-85dd-fc904fc9c726"
    # input_path = "C:/Users/spathak/Downloads/PCC/new_exps/data/realpc/hung_1/" + pc_name + ".ply"
    # output_path = "C:/Users/spathak/Downloads/PCC/new_exps/upsampling/results/"+ pc_name + "_up.ply"

    start_time = time.time()

    upsample(input_path, output_path)


    elapsed_time = time.time() - start_time
    print(time.strftime("Total time: %H:%M:%S", time.gmtime(elapsed_time)))


    import sys
