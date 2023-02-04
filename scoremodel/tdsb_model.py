import time, torch, functools, utils, math, sys, random
sys.path.insert(1, '/Users/joshuha.thomas-wilsker/Documents/work/machine_learning/score_based_models/calo_challenge/code')
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from HighLevelFeatures import HighLevelFeatures as HLF

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__() # inherits from pytorch nn class
        # Randomly sample weights during initialisation. 
        # Weights are fixed during optimisation and are not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # tensor of times multiplied by matrix of weights
        # size of matrix 1/2 embedding dimension
        # reduces dimensionality of continuous time
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
        kernel_size = 7
        # Create time embedding (small NN with fixed weights)
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
    def forward(self, x, t, e):
        #print('Models input: %s %s %s ' % (x.shape, t.shape, e.shape) )
        
        # Obtain the Gaussian random feature embedding for t
        # (projected into 64 dimensional embedding space)
        embed = self.act(self.embed(t))
        # Obtain the Gaussian random feature embedding for energy
        embed_e_ = self.act(self.embed(e))
        # Encoding path
        # kernel slides over input and outputs 32 channels each with 364 output (depends on kernel size)
        h1 = self.conv1(x)
        # Incorporate information on parameter 't' (noise scale)
        # pass 64 dim time embedding through dense layer creates 32 dim output (# examples, 32 channels, 1 time/noise-scale)
        # The same noise-scale used for all pixels and spans across the channels
        h1 += self.dense1(embed)
        h1 += self.dense1(embed_e_)
        # Group normalisation
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 += self.dense2(embed_e_)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 += self.dense3(embed_e_)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 += self.dense4(embed_e_)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h += self.dense5(embed_e_)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h += self.dense6(embed_e_)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h += self.dense7(embed_e_)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
         # Rescale the output of the U-net (see tutorial comments)
        #h = h / self.marginal_prob_std(t)[:, None, None, None]
        h = h / self.marginal_prob_std(t)[:, None, None]
        #print('Models output: ', h.shape)
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

def diffusion_coeff(t, sigma=25.0):
    """Compute the diffusion coefficient of our SDE
    Args:
        t: A vector of time steps
        sigma: from the SDE
    Returns:
    Vector of diffusion coefficients
    """
    return torch.tensor(sigma**t, device=device)

def loss_fn(model, x, energy_, epoch, marginal_prob_std, eps=1e-5):
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
    # Tensor (length=batch_size) of time/noise steps randomly sampled from a uniform distribution over [0,1)
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # Tensor of energies (remove all dimensions=1 to match time dimensions)
    energy_ = energy_.squeeze()

    # Concatenate time and energy tensors into one embedding
    #embedded_info_ = torch.concat((random_t, energy_),-1)
    
    # Tensor (length=batch_size) with dimensions of input but with values randomly sampled from a Gaussian with mean 0 and variance 1
    z = torch.randn_like(x)
    
    # Tensor of the standard deviations of the transition kernel for each time/noise step
    std = marginal_prob_std(random_t)

    # Perturb original data by adding Gaussian noise proportional to the size of the perturbation 
    # Strictly, proportional to the std of the Gaussian transition kernel of the time-dependent SDE
    # Careful with the dimesnions of the standard deviation object
    perturbed_x = x + z * std[:, None, None]
    
    # Evaluate the model on the perturbed data
    score = model(perturbed_x, random_t, energy_)

    # Evaluate the loss
    loss = torch.mean( torch.sum( (score * std[:, None, None] + z)**2, dim=(1,2,3) ) )
    return loss

def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_energies, batch_size=64, snr=0.16, device='cuda', eps=1e-3):
    ''' Generate samples from score based models with Predictor-Corrector method
        Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the std of the perturbation kernel
        diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

        Returns:
            samples
    '''
    num_steps=500
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 368, device=device) * marginal_prob_std(t)[:,None,None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0]-time_steps[1]
    x = init_x
    with torch.no_grad():
         for time_step in time_steps:
            print(f"Sampler step: {time_step:.4f}")
            batch_time_step = torch.ones(batch_size,device=device) * time_step
            #sampled_energies.to(device)
            # Sneaky bug fix (matrix multiplication in GaussianFourier projection doesnt like float64s)
            sampled_energies = sampled_energies.to(torch.float32)
            
            # Corrector step (Langevin MCMC)
            # First calculate Langevin step size
            grad = score_model(x, batch_time_step, sampled_energies)
            grad_norm = torch.norm( grad.reshape(grad.shape[0], -1), dim=-1 ).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            
            # Implement iteration rule
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Euler-Maruyama predictor step
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None] * torch.randn_like(x)

    # Do not include noise in last step
    return x_mean

def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filename = '../datasets/dataset_1_photons_1.hdf5'

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    n_epochs = 20
    batch_size = 100
    lr = 1e-4
    training_switch = 0
    sampling_switch = 1

    # load .hdf5 dataset
    train_ds = utils.custom_dataset(filename,'train', 0.5)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    
    HLF_1_photons = HLF('photon', filename='../code/binning_dataset_1_photons.xml')
    HLF_1_photons.CalculateFeatures(train_ds.showers_[:])
    for energy in [256., 1024., 1048576.]:
        savename = 'training_showers_'+str(energy)+'.png'
        voxel_dim = train_ds.showers_.shape[1]
        energy_idx = np.tile( train_ds.energies_[:] == energy, voxel_dim)
        reshaped_showers_ = train_ds.showers_[energy_idx].reshape(-1, voxel_dim)
        _ = HLF_1_photons.DrawAverageShower(reshaped_showers_, filename=savename, title='Av. photon shower @ E = {} MeV'.format(energy))

    if training_switch:
        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        score_model = score_model.to(device)
        # converts all parameters to float64
        score_model = score_model.to(torch.double)
        optimiser = Adam(score_model.parameters(),lr=lr)

        for epoch in range(0,n_epochs):
            avg_loss = 0.
            num_items = 0
            batch_ = 0
            print('Epoch: {:1f}'.format(epoch))
            # loop over batches
            for shower_, energy_ in train_dl:
                batch_+=1
                if batch_ % 1000 == 0:
                    print(f"Batch: {batch_:.1f}")
                shower_ = shower_.to(device)
                energy_ = energy_.to(device)
                # Add dummy channel dimension as functional 
                # Conv1d expects 1st input argument w. dimensions: [batchsize, channels, sequence_length]
                shower_ = torch.unsqueeze(shower_, dim=1)
                loss = loss_fn(score_model, shower_, energy_,epoch, marginal_prob_std_fn)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                avg_loss+=loss.item()*shower_.shape[0]
                num_items+=shower_.shape[0]
            print(f'Average Loss: {(avg_loss/num_items):.3f}')
            # Save checkpoint file after each epoch
            torch.save(score_model.state_dict(), 'ckpt_tmp.pth')

    if sampling_switch:
        sample_batch_size = 100
        score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
        load_name = 'test_model/ckpt.pth'
        score_model.load_state_dict(torch.load(load_name, map_location=device))
        sampled_energies = utils.uniform_energy_sampler(filename, sample_batch_size).energy_samples
        sampled_energies = sorted(train_ds.energies_[:])
        sampled_energies = random.sample(sampled_energies, sample_batch_size)
        sampled_energies = torch.tensor(sampled_energies) # Converting tensor from list of ndarrays is very slow (should convert to single ndarray first)
        sampled_energies = torch.squeeze(sampled_energies)
        sampler = pc_sampler
        samples = sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_energies, sample_batch_size, device=device)
        samples = torch.squeeze(samples)
        sampled_energies = torch.unsqueeze(sampled_energies, dim=1)
        
        for energy in [256., 1024., 1048576.]:
            savename = 'genshowers_'+str(energy)+'.png'
            voxel_dim = samples.shape[1]
            energy_idx = np.tile( sampled_energies[:] == energy, voxel_dim)
            reshaped_samples = samples[energy_idx].reshape(-1, voxel_dim)
            _ = HLF_1_photons.DrawAverageShower(reshaped_samples, filename=savename, title='Av. photon shower @ E = {} MeV'.format(energy))

        
            
    
    
    



if __name__=='__main__':
    start = time.time()
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))

