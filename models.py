import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryThreshold(torch.autograd.Function):
    """
    Used to generate the transcription on-off states
    """
    @staticmethod
    def forward(ctx, input_tensor, threshold=0):
        ctx.save_for_backward(input_tensor)
        return torch.where(input_tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None

def smoothness_loss(output):
    """
    Compute smoothness loss based on the squared differences between adjacent elements.
    Parameters:
        output (Tensor): Output tensor from the neural network.
    Returns:
        smoothness_loss (Tensor): Smoothness loss tensor.
    """
    # Compute squared differences between adjacent elements along the time dimension
    squared_diff = torch.pow(output[:, :-1] - output[:, 1:], 2)
    # Compute mean of squared differences
    smoothness_loss = torch.mean(squared_diff)
    return smoothness_loss

class Model(nn.Module):
    def __init__(self, dim_inputs, hidden_size, latent_size, seq_len):
        super(Model, self).__init__()
        self.dim_inputs = dim_inputs
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len = seq_len

        self.rnn_cell = nn.LSTMCell(dim_inputs, self.hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)  # next step mean distance of the existing molecules
        self.fc_var = nn.Linear(hidden_size, latent_size) # next step log variance of the existing melocules
        self.velo = nn.Linear(hidden_size, latent_size)   # latent generation rate at center
        self.generator = nn.Sequential(                   # non-negative output generation rate
            nn.Linear(latent_size, 1),
            nn.ReLU()
        )
        self.switch = nn.Linear(hidden_size, latent_size) # latent transcription state  
        self.on = nn.Sequential(                          # whether the transcription is on/off
            nn.Linear(latent_size, 1)
            # nn.Sigmoid()
        )
        self.next_state = nn.Linear(latent_size, dim_inputs) # predict state of the next step
        self.location_embedding = nn.Linear(2, self.hidden_size) # embedding of the physical location
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Log variance of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, input, h, c):
        h, c = self.rnn_cell(input, (h, c))
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        velo = self.velo(h)
        switch = self.switch(h)
        return h, c, mu, logvar, velo, switch
    
    def decode(self, x, z, velo, switch):
        # state = self.next_state(z)
        delta_state = self.next_state(z)             # change of the state
        state = torch.zeros_like(x).to(device)
        state[:, 1:] = x[:, 1:] + delta_state[:, 1:] # skip connection
        state = F.relu(state).clamp(min=0)
        totals_pre = torch.sum(state, 1)
        generate = self.generator(velo)              # number of generated molecules
        # generate = F.relu(generate)
        central = torch.zeros((z.size()[0], self.dim_inputs), requires_grad=False)
        central[:, 0] += 1
        central = central.to(device)
        on_off = BinaryThreshold.apply(self.on(switch)) # the transcription on-off state
        state += generate * central * on_off            # at the cell center, on state will generate new molecules
        totals_post = torch.sum(state, 1)
        return state, generate, on_off, totals_pre, totals_post
    
    def forward(self, data, locs):
        # h = Variable(torch.zeros((data.size()[0], self.hidden_size)))
        h = self.location_embedding(locs).to(device)
        c = Variable(torch.zeros((data.size()[0], self.hidden_size))).to(device)
        states = []
        generations = []
        trans_status = []
        totals_no_gen = [torch.zeros(data.size()[0]).to(device)]
        totals_gen = [torch.zeros(data.size()[0]).to(device)]
        x = data[:, 0, :]
        for i in range(1, self.seq_len):
            h, c, mu, logvar, velo, switch = self.encode(x, h, c)
            z = self.reparameterize(mu, logvar)
            x, generate, on_off, totals_pre, totals_post = self.decode(x, z, velo, switch)
            states.append(x)
            generations.append(generate)
            trans_status.append(on_off)
            totals_no_gen.append(totals_pre)
            totals_gen.append(totals_post)
        states_tensor = torch.stack(states, dim=0).permute(1, 0, 2)             # states of each time step
        generations_tensor = torch.stack(generations, dim=0).permute(1, 0, 2)   # generation rates
        trans_status_tensor = torch.stack(trans_status, dim=0).permute(1, 0, 2) # on-off status
        totals_no_gen = torch.stack(totals_no_gen, dim=0).permute(1, 0)
        totals_gen = torch.stack(totals_gen, dim=0).permute(1, 0)
        return states_tensor, generations_tensor, trans_status_tensor, totals_no_gen, totals_gen

