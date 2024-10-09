from topologylayer.nn import AlphaLayer
import h5py
import os
import torch
import os
from sinkhorn import sinkhorn
import time

def load_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hf:
        array = hf[dataset_name][:]
    return array

npy_folder_path = '../data/Dutch/easy_0.8'

complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")
occl = load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl")
non_sparse = load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse")
uni_sparse = load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse")

print(complete.shape)
print(occl.shape)
print(non_sparse.shape)
print(uni_sparse.shape)

data = complete[11]

print(data.shape)


layer = AlphaLayer(maxdim=2)
# x = your_point_cloud
s=time.time()
pd = layer(torch.from_numpy(data).float())
print(time.time()-s)
print(pd[0][0])

loss, corrs_1_to_2, corrs_2_to_1 = sinkhorn(pd[0][1], pd[0][1], p=2, verbose=True)

print(loss)