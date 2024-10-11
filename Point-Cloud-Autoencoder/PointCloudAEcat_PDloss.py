# %%
# import point_cloud_utils as pcu
# import polyscope as ps
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from sinkhorn import sinkhorn
from scipy.spatial.transform import Rotation
import time, pdb
from tqdm import trange

import matplotlib.pyplot as plt

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
import h5py
import os
import argparse

from mpl_toolkits.mplot3d import Axes3D
from topologylayer.nn import AlphaLayer
from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from sinkhorn import sinkhorn
import model
import utils
import pickle


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',         type=str,   default='',            help='size of minibatch used during training')
parser.add_argument('--batch_size',   type=int,   default=512,           help='size of minibatch used during training')
parser.add_argument('--log',          type=str,   default=512,           help='size of minibatch used during training')

args = parser.parse_args()

def load_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hf:
        array = hf[dataset_name][:]
    return array

# npy_folder_path = '../data/Dutch/easy_0.8'
npy_folder_path = args.data

def sample(point_cloud_array):
    new_array = np.zeros((point_cloud_array.shape[0], 1024, 3))
    for i in range(point_cloud_array.shape[0]):
        indices = np.random.choice(point_cloud_array[i].shape[0], 1024, replace=False)
        new_array[i] = point_cloud_array[i][indices, :]
    return new_array



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




def train_epoch(epoch, static_persistence):


    
    epoch_loss = 0
    
    niters = 500
    eps = 1e-3
    stop_error = 1e-5
    k=0
    for data in (train_loader):
        
        # #pdb.set_trace()
        incomplete_data = data[1].to(device)
        complete_data = data[2].to(device)
        persistence =  static_persistence[k*args.batch_size: (k+1)*args.batch_size]
        k+=1 
        
        

        optimizer.zero_grad()
        output = net(incomplete_data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        # #pdb.set_trace()
        cham_loss, _ = chamfer_distance(complete_data, output) 

        loss = cham_loss



        if wasserstein == True and epoch % 10 ==0 :

            loss_pd = 0
            for i in trange(complete_data.size()[0]):   

                layer = AlphaLayer(maxdim=1)

                pd_pred = layer(output[i])
                pd_comp = persistence[i]    #add 
                # pd_comp = pd_comp.cuda()
                
                # #pdb.set_trace()
                loss_h0, corrs_1_to_2, corrs_2_to_1 = sinkhorn(pd_pred[0][0][1:], pd_comp[0][1:].cuda(),  p=2, eps=eps, max_iters=niters, stop_thresh=stop_error, verbose=False)
                loss_h1, corrs_1_to_2, corrs_2_to_1 = sinkhorn(pd_pred[0][1], pd_comp[1].cuda(), p=2,  eps=eps, max_iters=niters, stop_thresh=stop_error, verbose=False)
            
                loss_pd += loss_h0 + loss_h1
            

            loss_pd =   loss_pd/complete_data.size()[0]
            loss    +=  loss_pd
        


        if emd == True:
            emd_loss = torch.Tensor([emd_samples(output.cpu(), complete_data.cpu(), bins = 40)])
        


        if topo == True:
            loss_topo = 0
            for i in range(complete_data.size()[0]):            
                dgminfo_pcd = pdfn_pcd(complete_data[i])    
                loss_topo += topfn(dgminfo_pcd) + topfn2(dgminfo_pcd)
            loss_topo = loss_topo/complete_data.size()[0]
            if emd == True:
                loss = emd_loss + loss_topo
            else:
                loss = cham_loss + 0.000001*loss_topo
        



        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/k





# %%
def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        incomplete_data = data[1].to(device)
        complete_data = data[2].to(device)
        # #pdb.set_trace()
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

    return epoch_loss/(i+1)




















# %%
batch_size = 32
output_folder = args.log # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model

# %%




class CatenaryLoader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, partial, complete, static_persistence):
        super(CatenaryLoader, self).__init__()

        self.partial  = partial
        self.complete = complete
        self.static_persistence = static_persistence
    

    def __len__(self):
        return self.complete.shape[0]

    def __getitem__(self, index):
        
        return index, self.partial[index], self.complete[index], self.static_persistence[index]






class CatenaryLoaderTest(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, partial, complete):
        super(CatenaryLoaderTest, self).__init__()

        self.partial  = partial
        self.complete = complete
    

    def __len__(self):
        return self.complete.shape[0]

    def __getitem__(self, index):
        
        return index, self.partial[index], self.complete[index]





#-------------------------------------------------------------------------------------------------------------------
#Dataloaders creation


dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_tr.npy'))).astype(np.float32)[::16]   #8668, 4000, 3
static_lidar = normalize(np.load(os.path.join(args.data, 'comp_tr.npy'))).astype(np.float32)[::16]

# pdb.set_trace()

with open(os.path.join(args.data, 'pds.pkl'), 'rb') as file:
    pds = pickle.load(file)


static_persistence = pds[15::16]       #add - assuming that  laoding gives as list --> the [3::4] gives every 4th element


# static_persistence =[]

# for i in static_persistence1:
#     if not isinstance(i[], tuple):
#         tup = tuple(i)
#         static_persistence.append(tup)
# #pdb.set_trace()





# static_persistence = np.ndarray(static_persistence, dtype = tuple)


data_train = CatenaryLoaderTest(dynamic_lidar, static_lidar)

# data_train = list(zip(dynamic_lidar, static_lidar, static_persistence))

train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                shuffle=False, num_workers=4, drop_last=True)




dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_te.npy')))
static_lidar  = normalize(np.load(os.path.join(args.data, 'comp_te.npy')))



data_test = CatenaryLoaderTest(dynamic_lidar, static_lidar)
test_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                shuffle=True, num_workers=4, drop_last=True)


input_size = dynamic_lidar.shape[1]
output_size = static_lidar.shape[1]
#-------------------------------------------------------------------------------------------------------------------


# %%
net = model.PointCloudAE(input_size, output_size, latent_size)

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)

# %%


optimizer = optim.Adam(net.parameters(), lr=0.0005)






pdfn_pcd = LevelSetLayer2D(size=(1024, 3),  sublevel=False)   # what is the significance of the size
topfn = PartialSumBarcodeLengths(dim=1, skip=1)
# topfn = SumBarcodeLengths(dim=1)
topfn2 = SumBarcodeLengths(dim=0)
topo = False

# from pyemd import emd_samples
emd = False

wasserstein = True
"xxxx"
# %%


# %%
if(save_results):
    utils.clear_folder(output_folder)

# %%
train_loss_list = []  
test_loss_list = []  





for i in range(501) :
    print('Epoch ', i)

    startTime = time.time()
    
    train_loss = train_epoch(i, static_persistence) #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    #pdb.set_trace()
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    print(writeString)
    
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
            # test_samples = [list(test_loader)[10][0], list(test_loader)[10][1]]
            loss , test_output = test_batch(test_samples)
            utils.plotPCbatch_comp(test_samples[2], test_samples[1], test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i))) # complete_gt, occl_input, complete_output

    else : # display all outputs
        
        test_samples = next(iter(test_loader))
        loss , test_output = test_batch(test_samples)
        utils.plotPCbatch(test_samples,test_output)

        print(writeString)

        plt.show()