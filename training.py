import torch
import torch.nn as nn
from models import Model, smoothness_loss
from utils import MyDataset
from torch.utils.data import DataLoader
import math


def train(data, locs, batch_size, base_lr, lr_step, num_epochs, hidden_size, latent_size, seq_len):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make data loader
    data = torch.from_numpy(data).float().to(device)
    locs = torch.from_numpy(locs).float().to(device)
    dataset = MyDataset(data, locs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # network initialization
    dim_inputs = data.shape[2]
    net = Model(dim_inputs, hidden_size, latent_size, seq_len)
    net.to(device)
    mse = nn.MSELoss()
    # start training
    for epoch in range(num_epochs):
        learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        recon = torch.FloatTensor([0])
        variation = torch.FloatTensor([0])
        status = torch.FloatTensor([0])
        smooth = torch.FloatTensor([0])
        for batch in data_loader:
            data = batch['data']
            locs = batch['location']
            prediction, generation, trans_status, totals_pre, totals_post = net(data, locs)
            loss_recon = mse(data[:, 1:], prediction)
            loss_var = mse(totals_pre[:, 1:], totals_post[:, :-1])
            loss_status = torch.mean(torch.sum(trans_status, dim=1))
            loss_smooth = smoothness_loss(trans_status)
            loss_center = mse(data[:, 1:, 0], prediction[:, :, 0])
            loss = loss_recon + 0.5*loss_var + 0.8*loss_center
            # loss = loss_recon + 0.0005*loss_status
            # loss = loss_recon + 0.00 * loss_var + 0.005 * loss_status + 0.1 * loss_smooth  # GATA3
            # loss = loss_recon + 0.00 * loss_var + 0.01 * loss_status + 0.1 * loss_smooth # FOXA1
            # loss = loss_recon + 0.00 * loss_var + 0.0000 * loss_status + 0.00 * loss_smooth # ZEB1

            recon += loss_recon.cpu()
            variation += loss_var.cpu()
            status += loss_status.cpu()
            smooth += loss_smooth.cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        recon /= len(data_loader)
        variation /= len(data_loader)
        status /= len(data_loader)
        smooth /= len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {recon.item():.4f}, Loss2: {status.item():.4f}, Loss3: {smooth.item():.4f}')
    return net


def test(sample, loc, net):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test
    net.eval()
    sample = torch.from_numpy(sample).float().to(device)
    loc = torch.from_numpy(loc).float().to(device)
    prediction, generation, trans_status, _, _ = net(sample, loc)
    return prediction, generation, trans_status