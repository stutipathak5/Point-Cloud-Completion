# %%
import numpy as np
import time
import utils
import matplotlib.pyplot as plt
import torch
import model
import torch.optim as optim
from Dataloaders import GetDataLoaders, GetDataLoaders_Catenary

# %%
batch_size = 32
output_folder = "output/catenary_occl_topo1/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model

# %%

















import h5py
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def load_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hf:
        array = hf[dataset_name][:]
    return array

npy_folder_path = '../data/Dutch/easy_0.8'

def sample(point_cloud_array):
    new_array = np.zeros((point_cloud_array.shape[0], 1024, 3))
    for i in range(point_cloud_array.shape[0]):
        indices = np.random.choice(point_cloud_array[i].shape[0], 1024, replace=False)
        new_array[i] = point_cloud_array[i][indices, :]
    return new_array

complete = sample(load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete"))
occl = sample(load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl"))
non_sparse = sample(load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse"))
uni_sparse = sample(load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse"))

print(complete.shape)
print(occl.shape)
print(non_sparse.shape)
print(uni_sparse.shape)


def normalize(pc_array):
    for i in range(pc_array.shape[0]):
        ptcloud = pc_array[i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud)
        bbox_pcd = pcd.get_axis_aligned_bounding_box()
        corners = bbox_pcd.get_box_points()
        bbox = np.asarray(corners)
        bbox[[3, 7]] = bbox[[7, 3]]
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        pc_array[i] = ptcloud
    return pc_array


complete = normalize(complete)
occl = normalize(occl)
non_sparse = normalize(non_sparse)
uni_sparse = normalize(uni_sparse)





# load dataset from numpy array and divide 90%-10% randomly for train and test sets

train_loader, test_loader = GetDataLoaders_Catenary(complete, occl, non_sparse, uni_sparse)
print(list(train_loader)[0][0].size(), list(train_loader)[0][1].size(), list(train_loader)[0][2].size(), list(train_loader)[0][3].size())
# Assuming all models have the same size, get the point size from the first model
point_size = len(train_loader.dataset[0][0])
print(point_size)

# till here

# %%
net = model.PointCloudAE(point_size,latent_size)

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)

# %%
from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

optimizer = optim.Adam(net.parameters(), lr=0.0005)




from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths

pdfn_pcd = LevelSetLayer2D(size=(1024, 3),  sublevel=False)   # what is the significance of the size
topfn = PartialSumBarcodeLengths(dim=1, skip=1)
topfn2 = SumBarcodeLengths(dim=0)
topo = True



# %%
def train_epoch():
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        incomplete_data = data[1].to(device)
        complete_data = data[0].to(device)
        optimizer.zero_grad()
        output = net(incomplete_data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        cham_loss, _ = chamfer_distance(complete_data, output) 
        if topo == True:
            loss_topo = 0
            for  i in range(complete_data.size()[0]):            
                dgminfo_pcd = pdfn_pcd(complete_data[i])    
                loss_topo += topfn(dgminfo_pcd) + topfn2(dgminfo_pcd)
            loss = cham_loss + loss_topo
        else:
            loss = cham_loss 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/i

# %%
def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        incomplete_data = data[1].to(device)
        complete_data = data[0].to(device)
        output = net(incomplete_data.permute(0,2,1))
        loss, _ = chamfer_distance(complete_data, output) 
        
    return loss.item(), output.cpu()

# %%
def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for i, data in enumerate(test_loader):
            loss, output = test_batch(data)
            epoch_loss += loss

    return epoch_loss/i

# %%
if(save_results):
    utils.clear_folder(output_folder)

# %%
train_loss_list = []  
test_loss_list = []  

for i in range(1001) :
    print(i)

    startTime = time.time()
    
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    # plot train/test loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
    plt.legend()

    if(save_results): # save all outputs to the save folder

        # write the text output to file
        with open(output_folder + "prints.txt","a") as file: 
            file.write(writeString)

        # update the loss graph
        plt.savefig(output_folder + "loss.png")
        plt.close()

        # save input/output as image file
        if(i%50==0):
            test_samples = next(iter(test_loader))
            loss , test_output = test_batch(test_samples)
            utils.plotPCbatch_comp(test_samples[0], test_samples[1], test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i))) # complete_gt, occl_input, complete_output

    else : # display all outputs
        
        test_samples = next(iter(test_loader))
        loss , test_output = test_batch(test_samples)
        utils.plotPCbatch(test_samples,test_output)

        print(writeString)

        plt.show()

        



