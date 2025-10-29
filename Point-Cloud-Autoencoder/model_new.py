import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d

'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

class PointCloudAE(nn.Module):

    #input is 3,4000 ouput is 3,5000
    def __init__(self, input_size, output_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,512)
        self.dec2 = nn.Linear(512,512)
        self.dec3 = nn.Linear(512,1024)
        
        self.dec4 = nn.Linear(1024,self.output_size*3)

    def encoder(self, x): 
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = self.bn5(self.conv5(x4))
        x6 = torch.max(x5, 2, keepdim=True)[0]
        x7 = x6.view(-1, self.latent_size)
        return x7
    
    def decoder(self, x):
        x1 = F.relu(self.dec1(x))
        x2 = F.relu(self.dec2(x1))
        x3 = F.relu(self.dec3(x2))
        x4 = self.dec4(x3)
        return x4.view(-1, 3,self.output_size)#, x1,x2,x3
    
    def forward(self, x):
        x = self.encoder(x)
        z=x
        x = self.decoder(x)
        return x,z
    