import math
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def quat_loss(self, recon_x, x):
        return torch.sum(1 - torch.sum(recon_x * x, dim=1))

    def reconstruction_loss(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def kl_divergence_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_shape))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def train_epoch(self, epoch):
        self.train()
        train_loss = 0
        rec_loss_g = 0
        kl_loss_g = 0
        true_kl_loss_g = 0

        for batch_idx, (data,) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            recon_batch, z, mu, logvar = self.forward(data)
            loss, rec_loss, kl_loss, true_kl_loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            rec_loss_g += rec_loss.item()
            kl_loss_g += kl_loss.item()
            true_kl_loss_g += true_kl_loss.item()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset)
        rec_loss_g /= len(self.train_loader.dataset)
        kl_loss_g /= len(self.train_loader.dataset)
        true_kl_loss_g /= len(self.train_loader.dataset)
        return train_loss, rec_loss_g, kl_loss_g, true_kl_loss_g

    def fit(self, X, y, n_epochs, batch_size, lr=1e-3, beta=1):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta

        train_data = torch.utils.data.TensorDataset(X, y)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        train_losses = []
        for epoch in tqdm(range(1, self.n_epochs + 1), desc='Epochs'):
            train_losses.append(self.train_epoch(epoch))
            recon_data, _, _, _ = self.forward(X, y)
            mse = F.mse_loss(recon_data, X).item()

        return train_losses, mse

    def sample(self, n_samples):
        with torch.no_grad():
            latents = torch.randn((n_samples, self.latent_shape))
            return self.decode(latents), latents

class GraspModel(BaseModel):
    def __init__(self, in_dim=21, hid_dim=128, lat_dim=2):
        super().__init__()
        self.input_shape = in_dim
        self.hidden_shape = hid_dim
        self.latent_shape = lat_dim

        ## Encoder
        self.enc_fc1 = nn.Linear(self.input_shape, self.hidden_shape)
        self.enc_fc2 = nn.Linear(self.hidden_shape, self.hidden_shape)
        self.enc_fc31 = nn.Linear(self.hidden_shape, self.latent_shape)
        self.enc_fc32 = nn.Linear(self.hidden_shape, self.latent_shape)
        
        ## Decoder
        self.dec_fc1 = nn.Linear(self.latent_shape, self.hidden_shape)
        self.dec_fc2 = nn.Linear(self.hidden_shape, self.hidden_shape)
        self.dec_fc3 = nn.Linear(self.hidden_shape, self.input_shape)

    def encode(self, x):
        h1 = F.relu(self.enc_fc1(x))
        h2 = F.relu(self.enc_fc2(h1))
        return self.enc_fc31(h2), self.enc_fc32(h2)

    def decode(self, z):
        leaky_relu = torch.nn.LeakyReLU(0.5)
        relu = torch.nn.ReLU()

        h1 = leaky_relu(self.dec_fc1(z))
        h2 = leaky_relu(self.dec_fc2(h1))
        out = relu(self.dec_fc3(h2))

        return out

    def loss(self, recon_x, x, mu, logvar):
        rec_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        return rec_loss + self.beta * kl_loss, rec_loss, self.beta * kl_loss, kl_loss

    def fit(self, X, n_epochs, batch_size, lr=1e-3, beta=1):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = beta * frange_cycle_cosine(0.0, 1.0, self.n_epochs, 6)

        train_data = torch.utils.data.TensorDataset(X)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        train_losses = []
        for epoch in tqdm(range(1, self.n_epochs + 1), desc='Epochs'):
            self.beta = self.betas[epoch - 1]
            train_losses.append(self.train_epoch(epoch))
            recon_data, _, _, _ = self.forward(X)
            mse = F.mse_loss(recon_data, X).item()

        recon_data, _, _, _ = self.forward(X)
        mse = F.mse_loss(recon_data, X).item()
        print (f'MSE: {mse}')
        return train_losses, mse

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    

def load_model(filename, model_type):
    print (f'Loading model from: {filename}')
    model_data = torch.load(filename)
    print (f"Model description: {model_data['desc']}")
    layers = model_data['layers']
    model = model_type(layers[0], layers[1], layers[2])
    model.load_state_dict(model_data['state_dict'])
    return model

