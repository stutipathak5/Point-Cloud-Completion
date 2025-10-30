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
from utils import *
from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from torchsummary import summary
import torch.nn.functional as F
from tqdm import trange


torch.manual_seed(0)

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',         type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--batch_size',   type=int,   default=512,          help='size of minibatch used during training')
parser.add_argument('--log',          type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--topo',         action= "store_true",             help='size of minibatch used during training')
parser.add_argument('--cont',         type=str,   default='',           help='size of minibatch used during training')

args = parser.parse_args()


triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))


def train_epoch():
   
    
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        loss = 0
        
        # pdb.set_trace()
        incomplete_data = data[1].cuda()
        complete_data = data[2].cuda()

        
        
        sparse_seed  = incomplete_data[:,::4]
        very_sparse_seed  = incomplete_data[:,::8]
        very_very_sparse_seed  = incomplete_data[:,::16]
        very_very_very_parse_seed  = incomplete_data[:,::32]
        very_very_very_very_parse_seed  = incomplete_data[:,::64]
        very_very_very_very_very_parse_seed  = incomplete_data[:,::128]
        

       
        
        #-----------------------------------------------------------------------------------------------
 
        optimizer.zero_grad()
        output_incom,z1   = net(incomplete_data.permute(0,2,1)) 
        output_compl,z2   = net(complete_data.permute(0,2,1)) 
        # output_dense,z3   = net(dense_seed.permute(0,2,1)) 
        output_spars,z4   = net(sparse_seed.permute(0,2,1)) 
        output_very_spars,z5   = net(very_sparse_seed.permute(0,2,1)) 
        output_very_very_spars,z5   = net(very_very_sparse_seed.permute(0,2,1)) 
        output_very_very_very_spars,z5   = net(very_very_very_parse_seed.permute(0,2,1)) 
        output_very_very_very_very_spars,z6   = net(very_very_very_very_parse_seed.permute(0,2,1)) 
        # output_very_very_very_very_very_spars,z7   = net(very_very_very_very_very_parse_seed.permute(0,2,1)) 
        
        
        # output_negative,z6 = net(data[3].cuda().permute(0,2,1))

        # pdb.set_trace()
        cham_loss_incom, _ = chamfer_distance(complete_data, output_incom.permute(0,2,1))
        cham_loss_compl, _ = chamfer_distance(complete_data, output_compl.permute(0,2,1))

        cham_loss_sparse, _ = chamfer_distance(complete_data, output_spars.permute(0,2,1))
        cham_loss_very_sparse, _ = chamfer_distance(complete_data, output_very_spars.permute(0,2,1))
        cham_loss_very_very_sparse, _ = chamfer_distance(complete_data, output_very_very_spars.permute(0,2,1))
        cham_loss_very_very__very_sparse, _ = chamfer_distance(complete_data, output_very_very_very_spars.permute(0,2,1))
        cham_loss_very_very__very_very_sparse, _ = chamfer_distance(complete_data, output_very_very_very_very_spars.permute(0,2,1))
        # cham_loss_very_very__very_very_very_sparse, _ = chamfer_distance(complete_data, output_very_very_very_very_very_spars.permute(0,2,1))

        #-----------------------------------------------------------------------------------------------


        sparse_seed_comp  = complete_data[:,::4]
        very_sparse_seed_comp  = complete_data[:,::8]
        very_very_sparse_seed_comp  = complete_data[:,::16]
        very_very_very_parse_seed_comp  = complete_data[:,::32]
        very_very_very_very_parse_seed_comp  = complete_data[:,::64]
        # very_very_very_very_very_parse_seed_comp  = complete_data[:,::128]

        # output_incom,z1   = net(incomplete_data.permute(0,2,1)) 
        # output_dense,z3   = net(dense_seed.permute(0,2,1)) 
        output_spars_comp,z4   = net(sparse_seed_comp.permute(0,2,1)) 
        output_very_spars_comp,z5   = net(very_sparse_seed_comp.permute(0,2,1)) 
        output_very_very_spars_comp,z5   = net(very_very_sparse_seed_comp.permute(0,2,1)) 
        output_very_very_very_spars_comp,z5   = net(very_very_very_parse_seed_comp.permute(0,2,1)) 
        output_very_very_very_very_spars_comp,z6   = net(very_very_very_very_parse_seed_comp.permute(0,2,1)) 
        # output_very_very_very_very_very_spars_comp,z7   = net(very_very_very_very_very_parse_seed_comp.permute(0,2,1)) 
        

        cham_loss_sparse_comp, _ = chamfer_distance(complete_data, output_spars_comp.permute(0,2,1))
        cham_loss_very_sparse_comp, _ = chamfer_distance(complete_data, output_very_spars_comp.permute(0,2,1))
        cham_loss_very_very_sparse_comp, _ = chamfer_distance(complete_data, output_very_very_spars_comp.permute(0,2,1))
        cham_loss_very_very__very_sparse_comp, _ = chamfer_distance(complete_data, output_very_very_very_very_spars_comp.permute(0,2,1))
        cham_loss_very_very__very_very_sparse_comp, _ = chamfer_distance(complete_data, output_very_very_very_very_spars_comp.permute(0,2,1))
        

        # contrastive0 = triplet_loss(z2,z1,z6)
        # contrastive1 = triplet_loss(z2,z3,z6)
        # contrastive2 = triplet_loss(z2,z4,z6)
        # contrastive3 = triplet_loss(z2,z5,z6)

        # contrastive = (contrastive0 +contrastive1 + contrastive2+contrastive3)/ 4
        cham_loss = (cham_loss_compl+ cham_loss_incom + cham_loss_sparse + cham_loss_very_sparse + cham_loss_very_very_sparse + cham_loss_very_very__very_sparse+cham_loss_very_very__very_very_sparse)/7
        cham_loss_comp = (cham_loss_sparse_comp + cham_loss_very_sparse_comp + cham_loss_very_very_sparse_comp + cham_loss_very_very__very_sparse_comp + cham_loss_very_very__very_very_sparse_comp)/5

        
        loss = cham_loss + cham_loss_comp# + contrastive
        loss.backward()
        optimizer.step()
        # print('Batch : ', i, ' Loss : ', loss.item())    
        epoch_loss += loss.item()
    
    return epoch_loss/i






def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        incomplete_data = data[1].cuda()
        complete_data = data[2].cuda()
        # pdb.set_trace()
        output,_  = net(incomplete_data.permute(0,2,1))
        loss, _ = chamfer_distance(complete_data, output.permute(0,2,1)) 
        
    return loss.item(), output.cpu()



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






#-------------------------------------------------------------------------------------------------------------------
#Dataloaders creation



dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_tr.npy'))).astype(np.float32)[:]   #8668, 4000, 3
static_lidar = normalize(np.load(os.path.join(args.data, 'comp_tr.npy'))).astype(np.float32)[:]
# pdb.set_trace()

samples = dynamic_lidar.shape[0]

data_train = CatenaryLoader(dynamic_lidar[:int(samples*0.8)], static_lidar[:int(samples*0.8)])
train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                shuffle=True, num_workers=4, drop_last=True)







# dynamic_lidar = normalize(np.load(os.path.join(args.data, 'part_te.npy'))).astype(np.float32)[:]   #8668, 4000, 3
# static_lidar = normalize(np.load(os.path.join(args.data, 'comp_te.npy'))).astype(np.float32)[:]



data_test = CatenaryLoader(dynamic_lidar[int(samples*0.8):], static_lidar[int(samples*0.8):])
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size,
                shuffle=False, num_workers=4, drop_last=False)


input_size = dynamic_lidar.shape[1]
output_size = static_lidar.shape[1]
#-------------------------------------------------------------------------------------------------------------------



os.makedirs(os.path.join('log', args.log), exist_ok=True)
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model




net = model.PointCloudAE(input_size, output_size, latent_size).cuda()


# %%


optimizer     = optim.Adam(net.parameters(), lr=0.001)
# scheduler   = CosineAnnealingLR(optimizer, 400, eta_min=0.0005)
output_folder = os.path.join('log', args.log)






# %%
if(save_results):
    utils.clear_folder(output_folder)

# %%
train_loss_list = []  
test_loss_list = []  


summary(net, (3, 1000), device='cuda')
# exit(0)
start = 0

if args.cont != '':
    weight =torch.load(args.cont)
    net.load_state_dict(weight['state_dict'])
    optimizer.load_state_dict(weight['optimizer'])
    start = weight['epoch']



best_test = np.inf
for i in trange(start,1501) :
    # print('Epoch ', i)

    startTime = time.time()
    
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)

    test_loss = test_epoch() # test with test set
    if test_loss < best_test:
        print('Best index till now', i)

        state = {'epoch': i + 1, 'state_dict': net.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(state, f'log/{args.log}/ckpt_best_{str(i)}.t7')
        best_test = test_loss
        
    if (i%15==0):
        state = {'epoch': i + 1, 'state_dict': net.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(state, f'log/{args.log}/ckpt_{str(i)}.t7')
    
    
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    append_to_log_file(os.path.join(args.log), writeString)
    print(writeString)
    
    # plot train/test loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
    plt.legend()

    if(save_results): 

        # write the text output to file
        with open(output_folder + "prints.txt","a") as file: 
            file.write(writeString)

        # update the loss graph
        plt.savefig(output_folder + "loss.png")
        plt.close()

        # save input/output as image file
    #     if(i%50==0):
    #         test_samples = next(iter(test_loader))
    #         # test_samples = [list(test_loader)[10][0], list(test_loader)[10][1]]
    #         loss , test_output = test_batch(test_samples)
    #         utils.plotPCbatch_comp_new(test_samples[2], test_samples[1], test_output.permute(0,2,1),output_folder+'/'+str(i), show=False, save=True, name = str(i)) # complete_gt, occl_input, complete_output

    # else : # display all outputs
        
    #     test_samples = next(iter(test_loader))
    #     loss , test_output = test_batch(test_samples)
    #     utils.plotPCbatch(test_samples,test_output)

    #     print(writeString)

    #     plt.show()