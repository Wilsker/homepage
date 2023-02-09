import time, torch, torch_geometric, functools, utils, math, sys, random
sys.path.insert(1, '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/calochall_homepage/code')
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
#from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, to_dense_batch
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
        print('x_cls: ', x_cls.shape)

        for layer in self.encoder:
            x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)

        return self.out(x)

def main():
    print('torch version: ', torch.__version__)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    
    #filename = '/eos/user/t/tihsu/SWAN_projects/homepage/datasets/dataset_1_photons_1_graph.pt'
    filename = '/eos/user/t/tihsu/SWAN_projects/homepage/datasets/dataset_1_photons_1_graph.pt'
    loaded_file = torch.load(filename)
    point_clouds = loaded_file[0]
    energies = loaded_file[1]
    print('clouds: ', type(point_clouds[0]))
    #batch_size = len(point_clouds)

    #num_epochs = 20
    #for epoch in range(0,num_epochs):
    #    print(f'Epoch: {epoch:1f}')
    
    point_clouds_loader = DataLoader(point_clouds,batch_size=1,shuffle=False)
    for batch in point_clouds_loader:
        batch = batch.to(device)
        print(f'x: {batch.x.shape}')
        print(f'x: {type(batch.x)}')
        print(f'batch: {type(batch.batch)}')
        new_batch = torch.unsqueeze(batch.x, 0)
        print(new_batch.size(dim=-1))
        # Size of the last dimension of the input
        # must match the input to the embedding layer
        model=Gen( new_batch.size(dim=-1) ,20,128,3,1,0)
        print(model(new_batch))

    
    '''for cloud in point_clouds:
        # Node feature matrix cloud.x
        print('cloud shape [points, features]: ', cloud.x.shape)

        cloud_loader = DataLoader(cloud)
        #z=torch.zeros(10,3,25)
        
        n_points = cloud.x.size(dim=1)
        print('n_points: ' , n_points)
        
        # Size of the last dimension of the input
        # must match the input to the embedding layer
        model=Gen(n_points,20,128,3,1,0)
        model(cloud.x)
        #print(model(z))'''


if __name__=='__main__':
    start = time.time()
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))