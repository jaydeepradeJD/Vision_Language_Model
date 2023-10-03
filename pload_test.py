from data import ProteinTransformDataset
from data import ProteinDataset
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg

def main(cfg):
	train_dataset = ProteinTransformDataset(cfg, 'train', cfg.CONST.N_VIEWS_RENDERING)
	test_dataset = ProteinDataset(cfg, 'train', cfg.CONST.N_VIEWS_RENDERING, big_dataset=True)
	

	train_data_loader = DataLoader(train_dataset, batch_size=cfg.CONST.BATCH_SIZE, shuffle=True, num_workers=cfg.CONST.NUM_WORKER,
													drop_last=True)
	test_data_loader = DataLoader(test_dataset, batch_size=cfg.CONST.BATCH_SIZE, shuffle=True, num_workers=cfg.CONST.NUM_WORKER,
													drop_last=True)

	# print(train_dataset.dirs)
	next(iter(train_data_loader))

	# print(train_dataset.pre_array)
	# print("*************")
	# print(train_dataset.after_array)
	


	#expected length 128 * n_views

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Vision Language Models')
	parser.add_argument('--save_dir', default='./logs/',
						type=str,help='path to directory for storing the checkpoints etc.')
	parser.add_argument('-b','--batch_size', default=32, type=int,
						help='Batch size')
	parser.add_argument('-ep','--n_epochs', default=32, type=int,
						help='Number of epochs')
	parser.add_argument('-g','--gpu', default=1, type=int,
						help='num gpus')
	parser.add_argument('--num_workers', default=1, type=int,
						help='num workers for data module.')
	parser.add_argument('-d', '--debug', action='store_true',
						help='fast_dev_run argument')
	parser.add_argument('--weights', dest='weights',
						help='Initialize network from the weights file', default=None)
	parser.add_argument('--n_views', dest='n_views_rendering',
						help='number of views used', default=1, type=int)
	parser.add_argument('--loss', dest='loss',
						help='Loss Function', default='bce', type=str)
	parser.add_argument('--lr', dest='lr',
						help='Learning Rate', default=None, type=float)
	parser.add_argument('--l2_penalty', dest='l2_penalty',
						help='L2 penalty Weight decay', default=None, type=float)
	parser.add_argument('--optim', dest='optim',
						help='Optimizer/Training Policy', default='adam', type=str)
	parser.add_argument('--rep', dest='rep',
						help='Protein representation', default='surface_with_inout', type=str)
	parser.add_argument('--gray', dest='gray',
						help='If the input images are grayscale', action='store_true')
	parser.add_argument('--inp_size', dest='inp_size',
						help='input image resolution ', default=224, type=int)
	parser.add_argument('--name', dest='name',
						help='Experiment Name', default=None, type=str)
	parser.add_argument('--proj_name', dest='proj_name',
						help='Project Name', default=None, type=str)
	parser.add_argument('-ae', '--ae', action='store_true',
						help='If training AutoEncoder')
	parser.add_argument('-bg', '--bg', action='store_true',
						help='Use images with added background')
	parser.add_argument('--ae_weights', dest='ae_weights',
						help='Initialize Encoder network from the weights file', default=None)
	parser.add_argument('-gn', '--gn', action='store_true',
						help='Use Group Normalization')
	parser.add_argument('-nodes','--num_nodes', default=None, type=int,
						help='Number of nodes for training')
	parser.add_argument('-OnlySeq', '--OnlySeq', action='store_true',
						help='If training AutoEncoder')
	parser.add_argument('-bigData', '--bigData', action='store_true',
						help='Use Bige dataset of 543K samples')
	parser.add_argument('--dropout', dest='dropout',
						help='Dropout rate', default=None, type=float)
	parser.add_argument('-transformer', '--transformer', action='store_true',
						help='Use transformer and specify number of transformer blocks')
	parser.add_argument('--num_blocks', dest='num_blocks',
						help='Number Transformer blocks', default=None, type=int)
	parser.add_argument('--num_heads', dest='num_heads',
						help='Number of heads in multi-head attention', default=1, type=int)
	parser.add_argument('--test', dest='test',
						help='Perform Inferencing', action='store_true')
	parser.add_argument('--num_test_samples', dest='num_test_samples',
						help='Number of samples to perform test ons', default=1, type=int)
	parser.add_argument('--disc', dest='disc',
						help='Use Discriminator', action='store_true')
	parser.add_argument('--num_samples', dest='num_samples',
						help='Numer of samples to use for training', default='128', type=str)
	
	args = parser.parse_args()

	if args.n_views_rendering is not None:
		cfg.CONST.N_VIEWS_RENDERING = args.n_views_rendering

	if args.num_samples is not None:
		cfg.DATASET.NUM_SAMPLES = args.num_samples

	if args.batch_size is not None:
		cfg.CONST.BATCH_SIZE = args.batch_size

	if args.num_workers is not None:
		cfg.CONST.NUM_WORKER  = args.num_workers

	main(cfg)