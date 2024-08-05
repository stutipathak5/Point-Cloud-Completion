import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
import open3d as o3d







def voxelize_point_cloud(point_cloud, voxel_size, grid_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)   
    voxels = voxel_grid.get_voxels()
    voxel_coordinates = np.array([voxel.grid_index for voxel in voxels])

    # bounding_box = voxel_grid.get_axis_aligned_bounding_box()
    # min_bound = bounding_box.min_bound
    # max_bound = bounding_box.max_bound
    # grid_dimensions = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    # print("Voxel Grid Dimensions:", grid_dimensions)

    voxel_grid_tensor = np.zeros(grid_size, dtype=np.float32)
    for coord in voxel_coordinates:
        if all(coord < np.array(grid_size)):
            voxel_grid_tensor[tuple(coord)] = 1
            
    return voxel_grid_tensor


class CatenaryDataset(Dataset):
    def __init__(self, gt_path, partial_path, voxel_size, grid_size, transform=None):
        self.gt_path = gt_path
        self.partial_path = partial_path
        self.gt_data = np.load(gt_path)
        self.partial_data = np.load(partial_path)
        self.transform = transform
        self.voxel_size = voxel_size
        self.grid_size = grid_size

    def __len__(self):
        return self.gt_data.shape[0]

    # FIX:this will take only the last partial pc
    def __getitem__(self, idx):
        for i in range(self.partial_data.shape[1]):
            gt_pc = self.gt_data[idx, :, :]
            partial_pc = self.partial_data[idx, i, :, :]
            gt_grid = voxelize_point_cloud(gt_pc, self.voxel_size, self.grid_size)
            partial_grid = voxelize_point_cloud(partial_pc, self.voxel_size, self.grid_size)

        if self.transform:
            gt_pc = self.transform(gt_pc)
            partial_pc = self.transform(partial_pc)

        # return torch.from_numpy(gt_pc).float(), torch.from_numpy(partial_pc).float(),\
        #         torch.from_numpy(gt_grid).float(), torch.from_numpy(partial_grid).float()
        return torch.from_numpy(gt_grid).float().unsqueeze(0)
    

















def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
