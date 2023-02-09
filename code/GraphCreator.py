from HighLevelFeatures import HighLevelFeatures as HLF
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import optparse


if __name__ == "__main__":

  usage = 'usage: %prog [options]'
  parser = optparse.OptionParser(usage)
  parser.add_option('-p','--particle', dest='particle', help = 'type of particle', default='photon', type='string')
  parser.add_option('-x','--xml', dest='binning_xml',   help = 'xml with binning information', default='binning_dataset_1_photons.xml', type='string')
  parser.add_option('-d','--dataset', dest='dataset',   help = 'input dataset', default = '/eos/user/t/tihsu/SWAN_projects/homepage/datasets/dataset_1_photons_1.hdf5', type='string')
  parser.add_option('-f','--from', dest='From', help = 'Start index', default = 0, type = int)
  parser.add_option('-t','--to', dest='To', help = 'End index', default = 5000, type = int)
  parser.add_option('-g','--tag', dest = 'tag', help = 'Store file tag', default = 0, type = int)
  parser.add_option('--store_geometric', dest = 'store_geometric', action = "store_true")

  (args,opt) = parser.parse_args()

  HLF_obj = HLF(args.particle, filename = args.binning_xml)
  data_file = h5py.File(args.dataset, 'r')
  data = [HLF_obj.Get_Graphic(data_file["showers"][args.From:args.To],args.store_geometric), data_file["incident_energies"][args.From:args.To]]
  if args.store_geometric:
    torch.save(data, args.dataset.replace('.hdf5', '_graph_'+str(args.tag)+'.pt'))
  else:
    torch.save(data, args.dataset.replace('.hdf5', '_tensor_'+str(args.tag)+'.pt'))

