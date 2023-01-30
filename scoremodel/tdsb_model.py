import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import functools
import utils
import math

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__() # inherits from pytorch nn class
        # Randomly sample weights during initialisation. 
        # Weights are fixed during optimisation and are not trainable
        # Scaled tensor of random numbers with size (embed_dim//2)
        # Stored as parameter subclass
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """Fully connected layer that reshapes outputs to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """Dense nn layer output musr have same dimensions as input data: 
             For 1D data: [batchsize, channels, (dummy)sequence_length]
             For 2D data: [batchsize, channels, (dummy)height_in_pixels, (dummy)width_in_pixels]
        """
        return self.dense(x)[..., None]

class ScoreNet(nn.Module):
    """A time-dependent score-based U-net model"""
    def __init__(self, marginal_prob_std, output_channels=[32, 64, 128, 256], embed_dim=64):
        """Initialise a time-dependent score-based network
        
        Args:
        marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel p_{0t}(x(t) | x(0))
        output_channels: The number of output_channels for feature maps of each resolution
        embed_dim: The dimensionality of Gaussian random feature embeddings
        """
        super().__init__()
        input_channels = 1
        kernel_size = 5
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv1d(input_channels, output_channels[0], kernel_size, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, output_channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=output_channels[0])
        self.conv2 = nn.Conv1d(output_channels[0], output_channels[1], kernel_size, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, output_channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=output_channels[1])
        self.conv3 = nn.Conv1d(output_channels[1], output_channels[2], kernel_size, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, output_channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=output_channels[2])
        self.conv4 = nn.Conv1d(output_channels[2], output_channels[3], kernel_size, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, output_channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=output_channels[3])
        #Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose1d(output_channels[3], output_channels[2], kernel_size, stride=2, bias=False, output_padding=1)    
        self.dense5 = Dense(embed_dim, output_channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=output_channels[2])
        self.tconv3 = nn.ConvTranspose1d(output_channels[2]+output_channels[2], output_channels[1], kernel_size, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, output_channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=output_channels[1])
        self.tconv2 = nn.ConvTranspose1d(output_channels[1]+output_channels[1], output_channels[0], kernel_size, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, output_channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=output_channels[0])
        self.tconv1 = nn.ConvTranspose1d(output_channels[0]+output_channels[0], 1, kernel_size, stride=1)
        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
    
    """Define the forward pass for the network"""
    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        # Incorporate information on parameter 't' (noise scale)
        h1 += self.dense1(embed)
        # Group normalisation
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
         # Rescale the output of the U-net (see tutorial comments)
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

"""Set up the SDE"""
def marginal_prob_std(t, sigma):
    """ 
    Choosing the SDE: 
        dx = sigma^t dw
        t in [0,1]
    Compute the standard deviation of: p_{0t}(x(t) | x(0))
        Args:
    t: A vector of time steps taken as random numbers sampled from uniform distribution [0,1)
    sigma: The sigma in our SDE which we set in the code
  
    Returns:
        The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def loss_fn(model, x, epoch, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models
    Uses the weighted sum of Denoising Score matching objectives
    Denoising score matching
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    - Perturbation avoids need to calculate trace of Jacobian of model output

    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model
        x: A mini-batch of training data
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
        eps: A tolerance value for numerical stability
    """
    # Create a tensor (length=batch_size) of time/noise steps randomly sampled from a uniform distribution over [0,1)
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    
    # Create tensor (length=batch_size) with dimensions of input but with values randomly sampled from a Gaussian with mean 0 and variance 1
    z = torch.randn_like(x)
    
    # Create tensor of the standard deviations of the transition kernel for each time/noise step
    std = marginal_prob_std(random_t)

    # Perturb original data by adding Gaussian noise proportional to the size of the perturbation 
    # Strictly, proportional to the std of the Gaussian transition kernel of the time-dependent SDE
    # Careful with the dimesnions of the standard deviation object
    perturbed_x = x + z * std[:, None, None]
    
    # Evaluate the model on the perturbed data
    score = model(perturbed_x, random_t)

    # Evaluate the loss
    loss = torch.mean( torch.sum( (score * std[:, None, None] + z)**2, dim=(1,2,3) ) )
    return loss

def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    n_epochs = 5
    batch_size = 3
    lr = 1e-4
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    # converts all parameters to float64
    score_model = score_model.to(torch.double)
    optimiser = Adam(score_model.parameters(),lr=lr)

    # load .hdf5 dataset
    filename = '../datasets/dataset_1_photons_1.hdf5'
    train_ds = utils.custom_dataset(filename,'train', 0.5)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(0,n_epochs):
        avg_loss = 0.
        num_items = 0
        batch_ = 0
        # loop over batches
        for shower_, energy_ in train_dl:
            batch_+=1
            if batch_ % 100 == 0:
                print('Batch: {:3f}'.format(batch_))
            shower_ = shower_.to(device)
            # Add dummy channel dimension as functional 
            # Conv1d expects 1st input argument w. dimensions: [batchsize, channels, sequence_length]
            shower_ = torch.unsqueeze(shower_, dim=1)
            loss = loss_fn(score_model, shower_, epoch, marginal_prob_std_fn)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            avg_loss+=loss.item()*shower_.shape[0]
            num_items+=shower_.shape[0]
        print('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save(score_model.state_dict(), 'ckpt.pth')
            
    
    
    



if __name__=='__main__':
    start = time.time()
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))

