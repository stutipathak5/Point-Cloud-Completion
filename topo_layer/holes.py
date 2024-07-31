# Example from README
from topologylayer.nn import AlphaLayer, BarcodePolyFeature
import torch, numpy as np, matplotlib.pyplot as plt
import argparse
# from utils512 import *
import os
# random pointcloud

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data_file', type=str)
parser.add_argument('--results_folder', type=str)
args = parser.parse_args()



def save(x,k, folder):
    y = x.detach().numpy()
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ax[0].scatter(data[:,0], data[:,1], s=1, color = 'blue')
    ax[0].set_title("Before")
    ax[1].scatter(y[:,0], y[:,1], s=1, color='red')
    ax[1].set_title("After")
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, left=False)
    plt.savefig('results/images/' +args.results_folder + '/holes-'+ str(k) + '.png',format='png' ,dpi=1200 )


def save_topo(x,k, folder):
    y = x.detach().numpy()
    # plt.figure()
    # plt.scatter(y[:,0], y[:,1], s=1, color='red')
    # # plt.xlim([-0.15, 0.15])
    # # plt.ylim([-0.20, 0.20])
    # plt.savefig('images/' +args.folder + '/holes-'+ str(k) + '.svg',format='svg' ,dpi=1200 )


    fig, ax = plt.subplots(ncols=1, figsize=(2,4))
    # ax[0].scatter(data[:,0], data[:,1], s=1, color = 'blue')
    # ax[0].set_title("Before")
    ax.scatter(y[:,0], y[:,1], s=1, color='red')
    # ax[0].set_title("After")
    for i in range(1):
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(bottom=False, left=False)
    plt.savefig('results/images/' +args.results_folder + '/holes-'+ str(k) + '.png',format='png' ,dpi=1200 )
    



if not os.path.exists('results/images/'+str(args.results_folder)):
	os.makedirs('results/images/'+str(args.results_folder))



np.random.seed(0)
data = np.load(args.data_file)[4]
print(data.shape)

# optimization to increase size of holes
layer = AlphaLayer(maxdim=1)
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
f1 = BarcodePolyFeature(0,2,0)  # dm, p ,q

k=0
optimizer = torch.optim.Adam([x], lr=1e-4)
for i in range(1000):
    optimizer.zero_grad()
    loss = f1(layer(x))
    save_topo(x, k, args.results_folder)
    k+=1
    loss.backward() 
    optimizer.step()

# save figure
