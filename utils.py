import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random


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
    if file2 is not None:
        df = pd.read_csv(file2)
        locations = df.values
        print(data.shape)
        return data, locations
    else:
        return data


def visualize(gt, pre):
    plt.plot(gt)
    plt.plot(pre)
    plt.show()
    return

def save_and_plot_simu(gt, pre, filename, switch='rate'):
    """
    Save the prediction results on simulated data to csv files for further plotting
    in R, and show the plots by python in jupyter notebook. 
    """
    # Create a DataFrame to hold time, ground truth, and prediction
    df = pd.DataFrame({
        'Time': range(len(gt)-1),
        'Ground_Truth': gt[:-1],
    })
    pre_df = pd.DataFrame(pre.T, columns=[f'sample_{i}' for i in range(pre.shape[0])])
    df = pd.concat([df, pre_df], axis=1)
    # Save the DataFrame as a CSV file
    df.to_csv(f'{filename}.csv', index=False)

    plt.plot(gt, c=(178/255, 60/255, 60/255, 255/255))
    plt.plot(pre[1], c=(111/255, 167/255, 182/255, 255/255))
    plt.xlabel('Time')
    if switch == 'rate':
        plt.ylabel('Rate')
    else:
        plt.ylabel('On/off status')
    plt.show()
    return


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # if using GPU
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disables some optimizations for reproducibility