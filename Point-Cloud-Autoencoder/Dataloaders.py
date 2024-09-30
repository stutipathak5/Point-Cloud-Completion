import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

class ReadDataset(Dataset):
    def __init__(self,  source):
     
        self.data = torch.from_numpy(source).float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(npArray, batch_size, train_set_percentage = 0.9, shuffle=True, num_workers=0, pin_memory=True):
    
    
    pc = ReadDataset(npArray)

    train_set, test_set = RandomSplit(pc, train_set_percentage)

    train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    
    return train_loader, test_loader










class ReadDataset_Catenary(Dataset):
    def __init__(self, complete, occl, non_sparse, uni_sparse):
        self.complete = torch.from_numpy(complete).float()
        self.occl = torch.from_numpy(occl).float()
        self.non_sparse = torch.from_numpy(non_sparse).float()
        self.uni_sparse = torch.from_numpy(uni_sparse).float()

    def __len__(self):
        return len(self.occl)

    def __getitem__(self, idx):

        no_partial_per_complete = int(self.occl.shape[0]/self.complete.shape[0])
        complete_pc = self.complete[idx // no_partial_per_complete, :, :]
        occl_pc = self.occl[idx, :, :]
        non_sparse_pc = self.non_sparse[idx, :, :]
        uni_sparse_pc = self.uni_sparse[idx // no_partial_per_complete, :, :]

        return complete_pc, occl_pc, non_sparse_pc, uni_sparse_pc


def GetDataLoaders_Catenary(complete, occl, non_sparse, uni_sparse):

    dataset = ReadDataset_Catenary(complete, occl, non_sparse, uni_sparse)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
