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
from torch_topological.nn import SignatureLoss
from torch_topological.nn import SummaryStatisticLoss  # Importing SummaryStatisticLoss
from torch_topological.nn import WassersteinDistance


torch.manual_seed(0)

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex,AlphaComplex

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',         type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--batch_size',   type=int,   default=512,           help='size of minibatch used during training')
parser.add_argument('--log',          type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--topo',         action= "store_true",             help='size of minibatch used during training')

parser.add_argument('-s', '--statistic',choices=['persistent_entropy','polynomial_function','total_persistence',], default='polynomial_function', help='Name of summary statistic to use for the loss')

    

parser.add_argument('-p',type=float,default=2.0,help='Outer exponent for summary statistic loss calculation')

parser.add_argument('-q',type=float,default=2.0,help='Inner exponent for summary statistic loss calculation. Will ''only be used for certain summary statistics.')



args = parser.parse_args()




def normalize(pc_array):
    is_tensor = torch.is_tensor(pc_array)
    if is_tensor:
        pc_array = pc_array.cpu().numpy()
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
        if abs(scale)<1e-6:
            #print("warning, too small. setting to 1e-6")
            scale = 1e-6
        
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        pc_array[i] = ptcloud
    if is_tensor:
        pc_array = torch.from_numpy(pc_array).float()
    return pc_array




top_loss = SignatureLoss(p=1)
wasserstein_dist = WassersteinDistance(p=2) 



def train_epoch():

   
    alpha_complex = AlphaComplex()

    loss_fn_entropy = SummaryStatisticLoss(
        summary_statistic='persistent_entropy',
        p=2,
        q=2
    )

    loss_fn_poly = SummaryStatisticLoss(
        summary_statistic='polynomial_function',
        p=2,
        q=2
    )

    loss_fn_persis = SummaryStatisticLoss(
        summary_statistic='total_persistence',
        p=2,
        q=2
    )


    
    
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        loss = 0
        
        # pdb.set_trace()
        incomplete_data = normalize(data[1]).to(device)
        complete_data = normalize(data[2]).to(device)
        optimizer.zero_grad()
        output= net(incomplete_data.permute(0,2,1)) # takes input as 3,4000
        print(f"Output shape: {output.shape}")  # Debugging line
        print(f"Complete data shape: {complete_data.shape}")  # Debugging line
        # Ensure output and complete_data are in the shape (N, P, D)
        if output.dim() == 3 and output.shape[1] == 3 and output.shape[2] == 5000:
            # Permute output to (N, P, D)
            output = output.permute(0, 2, 1)  # Change shape to (N, P, D)
        elif output.dim() == 2:
            # If output has shape (N, D), you may need to add a dimension
            output = output.unsqueeze(1)  # Change shape to (N, 1, D)
        else:
            raise ValueError(f"Unexpected output dimensions: {output.dim()}")  # Raise an error for unexpected dimensions

        # Ensure complete_data is also in the shape (N, P, D)
        if complete_data.dim() == 2:
            complete_data = complete_data.unsqueeze(1)  # Change shape to (N, 1, D) if needed
        elif complete_data.dim() == 3:
            # If complete_data is already in the correct shape, do nothing
            pass
        else:
            raise ValueError(f"Unexpected complete_data dimensions: {complete_data.dim()}")  # Raise an error for unexpected dimensions

        # Now call chamfer_distance
        
        
        cham_loss, _ = chamfer_distance(complete_data, output)

        #-------------------------------------------------------
        

        print("shape of output = ", output.shape)
        print("shape of complete data = ", complete_data.shape)
        #output = output[:,1000:,:]
        #complete_data = complete_data[:,1000:,:]

        vr = VietorisRipsComplex(dim=2)
        print("1")

        pi_source = vr(output) #[bs,5000,3]
        print("2")
        pi_target = vr(complete_data)  #[bs,512,1]  to be 
        print("3")

        pi_source_diagram = pi_source.persistence()
        pi_target_diagram = pi_target.persistence()
        print("4")

        # Convert diagrams to tensors (birth-death pairs)
        # Assuming persistence() outputs something convertible to tensors
        pi_source_tensor = torch.tensor(pi_source_diagram, dtype=torch.float32)
        pi_target_tensor = torch.tensor(pi_target_diagram, dtype=torch.float32)

       
      

        print("has progressed beyond lines 153 and 154")
        

        # Now pass the extracted points to the loss function
    
    # ... existing code ...
        # Check the type of pi_source and pi_target
        print(type(pi_source), type(pi_target))  # Debugging line to check types
        topo_loss1 = loss_fn_entropy(pi_source_tensor, pi_target_tensor)
        #topo_loss2 = loss = loss_fn_poly(pi_source, pi_target)
        # topo_loss3 = loss = total_persistence(pi_source, pi_target)
        #wass_loss = wasserstein_dist(pi_source,pi_target)
        #-------------------------------------------------------

        #loss = cham_loss + topo_loss1
        loss = cham_loss + topo_loss1
        # loss = cham_loss + topo_loss3
        # loss = cham_loss + topo_loss1 + topo_loss2 + topo_loss3

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
train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,shuffle=True, num_workers=1, drop_last=True)




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





for i in range(100) :
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
            print("Reconstructed Output Shape before permute:", test_output.shape)
            test_output = test_output.permute(0,2,1)
            print("Complete Ground Truth Shape:", test_samples[2].shape)
            print("Incomplete Input Shape:", test_samples[1].shape)
            print("Reconstructed Output Shape after permute:", test_output.shape)
            utils.plotPCbatch_comp(test_samples[2], test_samples[1], test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i))) # complete_gt, occl_input, complete_output

    else : # display all outputs
        
        test_samples = next(iter(test_loader))
        loss , test_output = test_batch(test_samples)
        utils.plotPCbatch(test_samples,test_output)

        print(writeString)

        plt.show()
