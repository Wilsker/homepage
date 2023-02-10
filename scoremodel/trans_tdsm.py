import time, functools, utils, math, sys, random
import torch#, torch_geometric
sys.path.insert(1, '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/calochall_homepage/code')
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from HighLevelFeatures import HighLevelFeatures as HLF


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        super().__init__()

        # batch_first=True because normally in NLP the batch dimension would be the second dimension
        # In everything(?) else it is the first dimension so this flag is set to true to match other conventions
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.fc1_cls = nn.Linear(embed_dim, hidden)
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.act_dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        residual = x.clone()
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x_cls = self.act(self.fc1_cls(x_cls))
        x_cls = self.act_dropout(x_cls)
        x_cls = self.fc2(x_cls)

        x = x + x_cls
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = x + residual
        return x


class Gen(nn.Module):
    def __init__(self,n_dim,l_dim_gen,hidden_gen,num_layers_gen,heads_gen,dropout_gen,**kwargs):
        super().__init__()
        # Embedding layer increases dimensionality:
        #   size of input (n_dim) features -> size of output (l_dim_gen)
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        # Encoder is a series of 'Blocks'
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=l_dim_gen,
                    num_heads=heads_gen,
                    hidden=hidden_gen,
                    dropout=dropout_gen,
                )
                for i in range(num_layers_gen)
            ]
        )
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)
        self.act = nn.GELU()

    def forward(self,x,mask=None):
        x = self.embbed(x)

        # 'class' token (mean field)
        x_cls = self.cls_token.expand(x.size(0), 1, -1)

        for layer in self.encoder:
            x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)

        return self.out(x)

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

def loss_fn(model, x, marginal_prob_std , eps=1e-5):
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
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None]
    model_output = model(perturbed_x)
    cloud_loss = torch.sum( (model_output*std + z)**2, dim=(1,2))
    return cloud_loss

def main():
    print('torch version: ', torch.__version__)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    
    filename = '/eos/user/t/tihsu/SWAN_projects/homepage/datasets/graph/dataset_1_photons_1_graph_0.pt'
    loaded_file = torch.load(filename)
    point_clouds = loaded_file[0]
    print(f'Loading {len(point_clouds)} point clouds from file {filename}')
    energies = loaded_file[1]
    nclouds = 1
    batch_size = 5
    lr = 1e-4
    n_epochs = 5
    
    # Size of the last dimension of the input must match the input to the embedding layer
    # First arg = number of features
    model=Gen(4,20,128,3,1,0)
    #for para_ in model.parameters():
    #    print('model parameters: ', para_)
    optimiser = Adam(model.parameters(),lr=lr)
    cumulative_epoch_loss = 0.
    cumulative_num_clouds = 0
    # Load all clouds in data
    point_clouds_loader = DataLoader(point_clouds,batch_size=nclouds,shuffle=False)
    cloud_iter_ = iter(point_clouds_loader)
    for epoch in range(0,n_epochs):
        print(f"epoch: {epoch}")
        
        batch_losses = []


        # Batch loop
        for i in range(0, len(point_clouds), batch_size):
            print(f"Batch: {i}")

            if cumulative_num_clouds+batch_size > len(point_clouds):
                batch_size = len(point_clouds)-cumulative_num_clouds
            
            # Loop over clouds in batch
            for cloud_ in range(0,batch_size):
                print(f'cloud_: {cloud_}')
                cloud_data = next(cloud_iter_)
                input_data = torch.unsqueeze(cloud_data.x, 0)
                # Calculate loss for cloud
                cloud_loss = loss_fn(model, input_data, marginal_prob_std_fn)
                batch_losses.append( cloud_loss )
            
            # Get mean of per cloud losses for batch loss
            batch_loss_tensor = torch.cat(batch_losses)
            batch_mean_loss = torch.mean(batch_loss_tensor)
            
            # Zero any gradients from previous steps
            optimiser.zero_grad()

            # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
            batch_mean_loss.backward(retain_graph=True)

            # Update value of x += -lr * x.grad
            optimiser.step()

            # add the batch mean loss * size of batch to cumulative loss
            cumulative_epoch_loss+=batch_mean_loss.item()*batch_size

            # add the batch size just used to the total number of clouds
            cumulative_num_clouds+=batch_size

        print(f'Sanity check: {cumulative_num_clouds}, {len(point_clouds)}')
        print(f'Average Loss for the epoch: {(cumulative_epoch_loss/cumulative_num_clouds):.3f}')
        # Save checkpoint file after each epoch
        torch.save(model.state_dict(), 'ckpt_tmp.pth')


if __name__=='__main__':
    start = time.time()
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))