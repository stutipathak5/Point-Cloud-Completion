import numpy as np
import time
import utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import model
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pdb
from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from torchsummary import summary

torch.manual_seed(0)

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',         type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--batch_size',   type=int,   default=512,           help='size of minibatch used during training')
parser.add_argument('--log',          type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--topo',         action= "store_true",             help='size of minibatch used during training')

args = parser.parse_args()




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





def train_epoch():

    
    
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        loss = 0
        
        # pdb.set_trace()
        incomplete_data = data[1].to(device)
        complete_data = data[2].to(device)
        optimizer.zero_grad()
        output,z = net(incomplete_data.permute(0,2,1)) # takes input as 3,4000
        
        cham_loss, _ = chamfer_distance(complete_data, output.permute(0,2,1))

        #-------------------------------------------------------
        pdb.set_trace()

        # expand z dimesion to 1 at end
        pi_z = pi_z.unsqueeze(2)

        pi_x = VietorisRipsComplex(complete_data)   #[bs,5000,3]
        pi_z = VietorisRipsComplex(z)               #[bs,512,1]  to be 


        topo_loss = self.loss([complete_data, pi_x], [z, pi_z])
        #-------------------------------------------------------

        loss = cham_loss + 10*topo_loss

        loss.backward()
        optimizer.step()
        # print('Batch : ', i, ' Loss : ', loss.item())    
        epoch_loss += cham_loss.item()
    
    return epoch_loss/(i+1)





# %%
def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        incomplete_data = data[1].to(device)
        complete_data = data[2].to(device)
        # pdb.set_trace()
        output = net(incomplete_data.permute(0,2,1))
        # loss, _ = chamfer_distance(complete_data, output.permute(0,2,1)) 
        # pdb.set_trace()
        loss, _ = chamfer_distance(complete_data, output.permute(0,2,1)) 
        # loss, _ = chamfer_distance(complete_data, output) 
        
    return loss.item(), output.cpu()

# %%
def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for i, data in enumerate(test_loader):
            loss, output = test_batch(data)
            epoch_loss += loss

    return epoch_loss/i



def append_to_log_file(log_file_path, text):

  with open(log_file_path, 'a') as f:
    f.write(text + '\n')



















# %%

os.makedirs(os.path.join('log', args.log), exist_ok=True)
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model

# %%




class CatenaryLoader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, partial, complete):
        super(CatenaryLoader, self).__init__()

        self.partial  = partial
        self.complete = complete
    

    def __len__(self):
        return self.complete.shape[0]

    def __getitem__(self, index):
        
        return index, self.partial[index], self.complete[index]





#-------------------------------------------------------------------------------------------------------------------
#Dataloaders creation


dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_tr.npy'))).astype(np.float32)[:]   #8668, 4000, 3
static_lidar = normalize(np.load(os.path.join(args.data, 'comp_tr.npy'))).astype(np.float32)[:]
# pdb.set_trace()


data_train = CatenaryLoader(dynamic_lidar, static_lidar)
train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,shuffle=True, num_workers=4, drop_last=True)




dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_te.npy'))).astype(np.float32)
static_lidar  = normalize(np.load(os.path.join(args.data, 'comp_te.npy'))).astype(np.float32)



data_test = CatenaryLoader(dynamic_lidar, static_lidar)
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size,
                shuffle=True, num_workers=4, drop_last=True)


input_size = dynamic_lidar.shape[1]
output_size = static_lidar.shape[1]
#-------------------------------------------------------------------------------------------------------------------


# %%
latent_size = 128
net = model.PointCloudAE(input_size, output_size, latent_size)

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)

# %%


optimizer     = optim.Adam(net.parameters(), lr=0.0005)
# scheduler   = CosineAnnealingLR(optimizer, 400, eta_min=0.0005)
output_folder = os.path.join('log', args.log)





# %%
train_loss_list = []  
test_loss_list = []  


summary(net, (3, input_size), device='cuda')
# exit(0)





for i in range(1001) :
    # print('Epoch ', i)

    startTime = time.time()
    
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    append_to_log_file(os.path.join(args.log), writeString)
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