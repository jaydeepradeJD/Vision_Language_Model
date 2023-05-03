import os
import numpy as np
import torch
# filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/PDBs_seq_emds/A0A2K3DMP5_tr.pdb/A0A2K3DMP5_tr_esm2_t33.txt'
# data = np.loadtxt(filepath)
# tensor = torch.from_numpy(data)

# print(data.shape)
# print(tensor.size())

with open('/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/val_samples.txt', 'r') as f:
	dir_list = f.readlines()
	dirs = [d.strip() for d in dir_list]

for i in range(5):
	print(dirs[i].split('/')[-1])
	print(dirs[i].split('/')[-1].split('.')[0])
	