import torch
from utils import read_data
import random
import numpy as np
import argparse
import os
from training import train


# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set parameters
SEQ_LEN = 20
dim_inputs = 10
hidden_size = 100
latent_size = 100

batch_size = 1024
base_lr = 0.01
lr_step = 10
num_epochs = 100


if __name__ == '__main__':
    # load parameters from command line
    parser = argparse.ArgumentParser(description='Training starts')
    parser.add_argument('-data_path', type=str, help='filename for loading sequencing data')
    parser.add_argument('-locs_path', type=str, help='filename for loading cell locations')
    parser.add_argument('-save_model_path', type=str, default='../model_params/', help='folder to save the trained model')
    parser.add_argument('-seq_len', type=int, default=20, help='sequence length of each sample')
    parser.add_argument('-dim_features', type=int, default=10, help='dimension of feature vector in each time step')
    args = parser.parse_args()
    # data loading and training
    data, locs = read_data(args.data_path, args.locs_path, args.seq_len, args.dim_features)
    train_data = data
    train_locs = locs
    net = train(train_data, train_locs, batch_size, base_lr, lr_step, num_epochs, hidden_size, latent_size, SEQ_LEN)
    # save the trained model
    gene = os.path.basename(args.data_path).split('_')[0]
    torch.save(net.state_dict(), args.save_model_path + gene + '_model.pth')