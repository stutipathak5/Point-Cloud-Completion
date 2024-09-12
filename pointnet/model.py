import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import open3d as o3d









def load_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hf:
        array = hf[dataset_name][:]
    return array


class CatenaryDataset(Dataset):
    def __init__(self, npy_folder_path, transform=None):
        self.npy_folder_path = npy_folder_path
        self.complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")
        self.occl = load_h5(os.path.join(npy_folder_path, "occl.h5"), "occl")
        self.non_sparse = load_h5(os.path.join(npy_folder_path, "non_sparse.h5"), "non_sparse")
        self.uni_sparse = load_h5(os.path.join(npy_folder_path, "uni_sparse.h5"), "uni_sparse")
        self.transform = transform

    def __len__(self):
        return self.occl.shape[0]

    def __getitem__(self, idx):

        no_partial_per_complete = int(self.occl.shape[0]/self.complete.shape[0])
        complete_pc = self.complete[idx // no_partial_per_complete, :, :]
        occl_pc = self.occl[idx, :, :]
        non_sparse_pc = self.non_sparse[idx, :, :]
        uni_sparse_pc = self.uni_sparse[idx // no_partial_per_complete, :, :]
        complete_pc = normalize_point_cloud(complete_pc)
        occl_pc = normalize_point_cloud(occl_pc)
        non_sparse_pc = normalize_point_cloud(non_sparse_pc)
        uni_sparse_pc = normalize_point_cloud(uni_sparse_pc)


        if self.transform:
            complete_pc = self.transform(complete_pc)
            occl_pc = self.transform(occl_pc)
            non_sparse_pc = self.transform(non_sparse_pc)
            uni_sparse_pc = self.transform(uni_sparse_pc)

        sampled_indices = torch.randperm(5000)[:1024]
        complete_pc = complete_pc[sampled_indices, :]
        # occl_pc = occl_pc[sampled_indices, :]
        # non_sparse_pc = non_sparse_pc[sampled_indices, :]
        # uni_sparse_pc = uni_sparse_pc[sampled_indices, :]


        return torch.from_numpy(complete_pc).float().transpose(0, 1) , torch.from_numpy(occl_pc).float().transpose(0, 1) ,\
                torch.from_numpy(non_sparse_pc).float().transpose(0, 1) , torch.from_numpy(uni_sparse_pc).float().transpose(0, 1)
    




class CatenaryDataset_only_comp(Dataset):
    def __init__(self, npy_folder_path, transform=None):
        self.npy_folder_path = npy_folder_path
        self.complete = load_h5(os.path.join(npy_folder_path, "complete.h5"), "complete")

    def __len__(self):
        return self.complete.shape[0]

    def __getitem__(self, idx):
        complete_pc = self.complete[idx, :, :]
        complete_pc = normalize_point_cloud(complete_pc)
        sampled_indices = torch.randperm(5000)[:1024]
        complete_pc = complete_pc[sampled_indices, :]


        return torch.from_numpy(complete_pc).float().transpose(0, 1)



# Normalize and center point clouds (mean centering and scaling to unit sphere)
def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)  # Compute centroid
    pc = pc - centroid  # Center to the origin
    max_distance = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # Compute the furthest distance
    pc = pc / max_distance  # Scale to unit sphere
    return pc

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]  # Max pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class PointNetDecoder(nn.Module):
    def __init__(self):
        super(PointNetDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 1024)  # Assuming output is a point cloud with 1024 points
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 3, 1024)  # Reshape to (batch_size, 3, num_points)
        return x



class PointNetAutoencoder(nn.Module):
    def __init__(self):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointNetEncoder()
        self.decoder = PointNetDecoder()
        
    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed_points = self.decoder(encoded_features)
        return reconstructed_points

# Visualization function using open3d (side by side)
def visualize_point_clouds_side_by_side(input_pc, output_pc, translation_offset=1.5):
    input_o3d = o3d.geometry.PointCloud()
    output_o3d = o3d.geometry.PointCloud()

    input_o3d.points = o3d.utility.Vector3dVector(input_pc)
    output_o3d.points = o3d.utility.Vector3dVector(output_pc)

    input_o3d.paint_uniform_color([1, 0, 0])   # Red for input
    output_o3d.paint_uniform_color([0, 1, 0])  # Green for output

    output_o3d.translate([translation_offset, 0, 0])

    o3d.visualization.draw_geometries([input_o3d, output_o3d])

# Visualization after all epochs
def visualize_saved_point_clouds_side_by_side():
    final_inputs = np.load("pointnet/final_inputs.npy", allow_pickle=True)
    final_outputs = np.load("pointnet/final_outputs.npy", allow_pickle=True)

    # Visualize side-by-side input and output point clouds
    for input_pc, output_pc in zip(final_inputs, final_outputs):
        visualize_point_clouds_side_by_side(input_pc.transpose(), output_pc.transpose())

# Training loop
def train(model, dataloader, optimizer, criterion, epochs, device):
    model.train()

    final_inputs = []
    final_outputs = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in dataloader:
            # batch = x[0]                  # for CatenaryDataset
            batch = x                   # for CatenaryDataset_only_comp
            inputs = batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(inputs.size(), outputs.size())
            loss = criterion(outputs, inputs)  # Compare reconstruction to input
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader)}")

        # Save inputs and outputs after the last epoch for visualization
        if epoch == epochs - 1:
            final_inputs = [pc.detach().cpu().numpy() for pc in inputs]
            final_outputs = [output.detach().cpu().numpy() for output in outputs]

    # Save the final inputs and outputs as numpy arrays
    np.save("pointnet/final_inputs.npy", final_inputs)
    np.save("pointnet/final_outputs.npy", final_outputs)

# Main function to set up dataset, model, optimizer, and train the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CatenaryDataset_only_comp(npy_folder_path = 'data\Dutch\easy_0.8', transform = None)


    print(len(dataset))
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


    # print(list(train_loader)[0][0].size(), list(train_loader)[0][1].size(), list(train_loader)[0][2].size(), list(train_loader)[0][3].size())   # for CatenaryDataset                 
    print(list(train_loader)[0].size())                                                                                                         # for CatenaryDataset_only_comp
    dataloader = train_loader
    # Initialize the model, optimizer, and loss function
    model = PointNetAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    epochs = 100
    train(model, dataloader, optimizer, criterion, epochs, device)

    # After training, visualize the input-output point clouds side by side
    visualize_saved_point_clouds_side_by_side()

if __name__ == "__main__":
    main()
