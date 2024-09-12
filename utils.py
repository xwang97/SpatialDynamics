import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, data, locations):
        self.data = data
        self.locations = locations

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = {
            'data': torch.Tensor(self.data[idx]),
            'location': torch.Tensor(self.locations[idx])
        }
        return sample


def read_data(file1, file2, SEQ_LEN, dim_inputs):
    df = pd.read_csv(file1)
    # Extract features from the DataFrame and reshape each row into a matrix
    # Each matrix is T*D, which represents a sample
    data = [row.values.reshape(SEQ_LEN, dim_inputs) for _, row in df.iterrows()]
    data = np.stack(data, axis=0)
    # Read the locations
    df = pd.read_csv(file2)
    locations = df.values
    print(data.shape)
    return data, locations


def visualize(gt, pre):
    plt.plot(gt)
    plt.plot(pre)
    plt.show()
    return