import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from models import Model
from statsmodels.tsa.stattools import grangercausalitytests
# from training import test


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
        # print(data.shape)
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


def visulize_cell(cell):
    # visulize how the spatial distribution of molecules changes over time
    time_steps = cell.shape[0]  # Number of time steps 
    fig, axes = plt.subplots(1, time_steps, figsize=(5 * time_steps, 5))  # 1 row, N columns

    for time_step, row in enumerate(cell):
        x_coords = []
        y_coords = []
        
        for distance, molecule_count in enumerate(row):
            # Generate coordinates for each molecule at this distance
            angles = np.random.uniform(0, 2 * np.pi, int(molecule_count))
            r = (distance + 0.1)*3  # Distances are indexed from 0, shift by 1 to avoid r=0
            
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            
            x_coords.extend(x)
            y_coords.extend(y)
        
        # Plot on the corresponding subplot
        ax = axes[time_step]
        ax.scatter(x_coords, y_coords, alpha=0.6)
        ax.scatter(0, 0, color='red', s=100, marker='o')  # Add red center point
        ax.set_title(f'Time Step {time_step + 1}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio square

    plt.tight_layout()  # Adjusts spacing between subplots for clarity
    plt.show()


def heuristic_alpha(file):
    """
    Heuristic way to determine the alpha parameter (weight of the status loss) for the training.
    Input: 
        file: the path to the csv file containing the molecules per gene data
    Output:
        alpha: the hyperparameter for the status loss
    """
    trans = pd.read_csv(file)
    grouped_trans_sizes = trans.groupby('cell_id').size()
    avg = grouped_trans_sizes.mean()
    alpha = avg / 50
    return alpha

def heuristic_alpha2(data):
    """
    Heuristic way to determine the alpha parameter (weight of the status loss) for the training.
    Input: 
        file: the path to the csv file containing the training samples
    Output:
        alpha: the hyperparameter for the status loss
    """
    sum_per_cell = data.sum(axis=-1)
    alpha = np.median(sum_per_cell) / 100
    return alpha

def easy_test(data_folder, model_folder, gene, SEQ_LEN, dim_inputs, hidden_size, latent_size):
    from training import test
    # load test data
    data_path = data_folder + gene + '_data.csv'
    locs_path = data_folder + gene + '_locs.csv'
    data, locs = read_data(data_path, locs_path, SEQ_LEN, dim_inputs)
    test_data = data
    test_locs = locs
    # load the trained model and start test
    model_path = model_folder + gene + '_model.pth'
    net = Model(dim_inputs, hidden_size, latent_size, SEQ_LEN)
    net.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        prediction, generation, trans_status, loss_recon = test(test_data, test_locs, net)
    return prediction, generation, trans_status, loss_recon

def granger_causality(velos, maxlag=2):
    """
    Run granger causality test on the velocity data.
    Input:
        velos: a dataframe containing the velocity data, first half of the columns are transcription rates of the 
        tf, second half are the transcription rates of the target gene.
        maxlag: the maximum lag to test
    Output:
        pvalues: a list of pvalues for the granger causality test of the input samples
        random_pvalues: a list of pvalues for the granger test of the randomly shuffled samples
    """
    pvalues = []
    random_pvalues = []
    for index, row in velos.iterrows():
        data = row.values
        sample1 = data[:len(data)//2]
        sample2 = data[len(data)//2:]
        if len(np.unique(sample1)) <= 3 or len(np.unique(sample2)) <= 3:
            pvalues.append(None)
            random_pvalues.append(None)
            continue
        sample1 += np.random.normal(0, 1e-6, len(sample1))  # add a small noise to avoid the error of granger test
        sample2 += np.random.normal(0, 1e-6, len(sample2))
        results = grangercausalitytests(np.column_stack((sample1, sample2)), maxlag=maxlag, verbose=False)
        pvalue = results[2][0]['lrtest'][1]
        pvalues.append(pvalue)

        np.random.shuffle(sample1)
        np.random.shuffle(sample2)
        results = grangercausalitytests(np.column_stack((sample1, sample2)), maxlag=maxlag, verbose=False)
        pvalue = results[2][0]['lrtest'][1]
        random_pvalues.append(pvalue)
    return pvalues, random_pvalues