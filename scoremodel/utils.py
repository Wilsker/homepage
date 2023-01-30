import h5py
import math
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self,file_name, train_test, train_test_split):
        file_ = h5py.File(file_name, 'r')
        n_examples = math.floor(file_['incident_energies'].shape[0]*train_test_split)
        self.showers_ = file_['showers']
        self.energies_ = file_['incident_energies']
        '''if train_test=='train':
            self.showers_ = file_['showers'][:n_examples]
            self.energies_ = file_['incident_energies'][:n_examples]
        elif train_test=='test':
            self.showers_ = file_['showers'][n_examples:]
            self.energies_ = file_['incident_energies'][n_examples:]'''
    
    def __len__(self):
        return len(self.showers_)
    
    def __getitem__(self, idx):
        '''Return item requested by idx (this returns a single sample)
        Pytorch DataLoader class will use this method to make an iterable for train/test/val loops'''
        shower_ = self.showers_[idx]
        energy_ = self.energies_[idx]
        return shower_, energy_

    

