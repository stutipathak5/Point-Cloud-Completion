import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split
import shutil
import open3d as o3d
import numpy as np
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)




import torch

def inject_gaussian_noise(point_clouds, mean, variance):
  """Injects Gaussian noise to each point cloud in the given PyTorch tensor.

  Args:
    point_clouds: A PyTorch tensor of shape (64, 2000, 3) representing 64 point clouds.
    mean: The mean of the Gaussian distribution.
    variance: The variance of the Gaussian distribution.

  Returns:
    A PyTorch tensor of shape (64, 2000, 3) with Gaussian noise injected into each point cloud.
  """

  # Create a noise tensor with the same shape as the point clouds
  noise = torch.randn_like(point_clouds) * torch.sqrt(torch.tensor(variance)) + mean

  # Add the noise to each point cloud
  noised_point_clouds = point_clouds + noise

  return noised_point_clouds





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





class CatenaryLoader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, partial, complete):
        super(CatenaryLoader, self).__init__()

        self.partial  = partial
        self.complete = complete
    

    def __len__(self):
        return self.complete.shape[0]-1

    def __getitem__(self, index):
        
        return index, self.partial[index], self.complete[index], self.partial[index + 1]








def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=9 , sizex = 12, sizey=3):
    
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

        # ax.set_xlim3d(0.25, 0.75)
        # ax.set_ylim3d(0.25, 0.75)
        # ax.set_zlim3d(0.25, 0.75)

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
    

def plotPCbatch_comp(pcArray1, pcArray2, pcArray3, show = True, save = False, name=None, fig_count=9 , sizex = 20, sizey=15):
    
    start = 0
    end = 9
    pc1 = pcArray1[start:end]
    pc2 = pcArray2[start:end]
    pc3 = pcArray3[start:end]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*3):

        ax = fig.add_subplot(3,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        elif (fig_count<=i<fig_count*2):
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc3[i-2*fig_count,:,0], pc3[i-2*fig_count,:,2], pc3[i-2*fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

        # ax.set_xlim3d(0.25, 0.75)
        # ax.set_ylim3d(0.25, 0.75)
        # ax.set_zlim3d(0.25, 0.75)

        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig