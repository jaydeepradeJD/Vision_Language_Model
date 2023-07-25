import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import ProteinDataset, SequenceDataset, ProteinAutoEncoderDataset
from VL_trainer import Model, OnlySeqModel, AutoEncoder, PretrainedAE_Model, AutoEncoder_old
from config import cfg
import utils.data_transforms
from test import test

def main(cfg):
	if not os.path.exists(cfg.DIR.OUT_PATH):
		os.makedirs(cfg.DIR.OUT_PATH)
		# Set up data augmentation
	
	IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
	
	if not cfg.DATASET.AUTOENCODER:
		train_transforms = utils.data_transforms.Compose([
			utils.data_transforms.Resize(IMG_SIZE),
			utils.data_transforms.ToTensor(),
			])
		val_transforms = utils.data_transforms.Compose([
			utils.data_transforms.Resize(IMG_SIZE),
			utils.data_transforms.ToTensor(),
			]) 
	
	if cfg.DATASET.AUTOENCODER:
		train_transforms = utils.data_transforms.Compose([
			utils.data_transforms.ResizeV2(IMG_SIZE),
			utils.data_transforms.ToTensorV2(),
			])
		val_transforms = utils.data_transforms.Compose([
			utils.data_transforms.ResizeV2(IMG_SIZE),
			utils.data_transforms.ToTensorV2(),
			]) 


	# test_transforms = utils.data_transforms.Compose([
	# 	utils.data_transforms.Resize(IMG_SIZE),
	# 	utils.data_transforms.ToTensor(),
	# 	])
	# Dataset
	if not cfg.DATASET.AUTOENCODER and not cfg.TRAIN.ONLYSEQ:
		train_dataset = ProteinDataset('train', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, train_transforms, grayscale=cfg.DATASET.GRAYSCALE, big_dataset=cfg.DATASET.BIGDATA)
		val_dataset = ProteinDataset('val', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, val_transforms, grayscale=cfg.DATASET.GRAYSCALE, big_dataset=cfg.DATASET.BIGDATA)
		# test_dataset = utils.data_loaders.ProteinDataset('test', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, test_transforms, grayscale=cfg.DATASET.GRAYSCALE)
		# test_dataset = None

	if cfg.DATASET.AUTOENCODER:
		train_dataset = ProteinAutoEncoderDataset('train', train_transforms, background=cfg.DATASET.BACKGROUND)
		val_dataset = ProteinAutoEncoderDataset('val', val_transforms, background=cfg.DATASET.BACKGROUND)
		# test_dataset = ProteinAutoEncoderDataset('test', test_transforms)
	
	if cfg.TRAIN.ONLYSEQ:
		train_dataset = SequenceDataset('train', cfg.CONST.REP, big_dataset=cfg.DATASET.BIGDATA)
		val_dataset = SequenceDataset('val', cfg.CONST.REP, big_dataset=cfg.DATASET.BIGDATA)
	
	if cfg.TEST.IS_TEST:
		indices = np.arange(cfg.TEST.NUM_SAMPLES) #random.sample(range(len(train_dataset)), cfg.CONST.BATCH_SIZE*10)
		train_dataset = torch.utils.data.Subset(train_dataset, indices)
		val_dataset = torch.utils.data.Subset(val_dataset, indices)

	# Set up Dataloader
	train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
													batch_size=cfg.CONST.BATCH_SIZE,
													num_workers=cfg.CONST.NUM_WORKER,
													shuffle=True,
													drop_last=True)
	val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
												  batch_size=cfg.CONST.BATCH_SIZE,
												  num_workers=cfg.CONST.NUM_WORKER,
												  shuffle=False)
	# if test_dataset is not None:
	# 	test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
	# 											  batch_size=1,
	# 											  num_workers=1,
	# 											  shuffle=False)

	# Initiate the Model
	
	if cfg.DATASET.AUTOENCODER:
		model = AutoEncoder(cfg)

	if (not cfg.DATASET.AUTOENCODER) and (not cfg.TRAIN.ONLYSEQ):
		if cfg.DIR.AE_WEIGHTS is not None:
			ae = AutoEncoder_old.load_from_checkpoint(cfg.DIR.AE_WEIGHTS, cfg=cfg)
			ae.eval()
			model = PretrainedAE_Model(cfg, pretrained_model=ae)
		else:
			model = Model(cfg)
	
	if cfg.TRAIN.ONLYSEQ:
		model = OnlySeqModel(cfg)

	if not cfg.TEST.IS_TEST:		
		# Initiate the trainer
		logger = pl.loggers.TensorBoardLogger(cfg.DIR.OUT_PATH, name=cfg.DIR.EXPERIMENT_NAME)

		wandb_logger = pl.loggers.WandbLogger(name=cfg.DIR.EXPERIMENT_NAME,
											project=cfg.DIR.PROJECT_NAME, dir=cfg.DIR.OUT_PATH)

		checkpoint = ModelCheckpoint(monitor='val/loss',
									dirpath=logger.log_dir, 
									filename='{epoch}-{step}',
									mode='min', 
									save_last=True)

		trainer = pl.Trainer(devices=cfg.TRAIN.GPU, 
							num_nodes=cfg.CONST.NODES,
							accelerator='gpu', 
							strategy='ddp',
							callbacks=[checkpoint],
							logger=[logger, wandb_logger], 
							max_epochs=cfg.CONST.NUM_EPOCHS, 
							default_root_dir=cfg.DIR.OUT_PATH, 
							fast_dev_run=cfg.TRAIN.DEBUG)

		# Training
		
		trainer.fit(model, train_data_loader, val_data_loader, ckpt_path=cfg.DIR.WEIGHTS)

	if cfg.TEST.IS_TEST:
		test(model, train_data_loader, val_data_loader, cfg.DIR.WEIGHTS, cfg.DIR.OUT_PATH)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Vision Language Models')
	parser.add_argument('--save_dir', default='./logs/',
						type=str,help='path to directory for storing the checkpoints etc.')
	parser.add_argument('-b','--batch_size', default=32, type=int,
						help='Batch size')
	parser.add_argument('-ep','--n_epochs', default=100, type=int,
						help='Number of epochs')
	parser.add_argument('-g','--gpu', default=1, type=int,
						help='num gpus')
	parser.add_argument('--num_workers', default=8, type=int,
						help='num workers for data module.')
	parser.add_argument('-d', '--debug', action='store_true',
						help='fast_dev_run argument')
	parser.add_argument('--weights', dest='weights',
						help='Initialize network from the weights file', default=None)
	parser.add_argument('--n_views', dest='n_views_rendering',
						help='number of views used', default=5, type=int)
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
	
	
	args = parser.parse_args()
		
	if args.save_dir is not None:
		cfg.DIR.OUT_PATH  = args.save_dir
	if args.debug:
		cfg.TRAIN.DEBUG = True

	if args.gpu is not None:
		cfg.TRAIN.GPU = args.gpu

	if args.num_workers is not None:
		cfg.CONST.NUM_WORKER  = args.num_workers

	if args.batch_size is not None:
		cfg.CONST.BATCH_SIZE = args.batch_size
	
	if args.n_epochs is not None:
		cfg.CONST.NUM_EPOCHS = args.n_epochs

	if args.n_views_rendering is not None:
		cfg.CONST.N_VIEWS_RENDERING = args.n_views_rendering
	
	if args.rep is not None:
		cfg.CONST.REP = args.rep
	if args.rep == 'surface_with_inout_fixed_views':
		cfg.CONST.N_VIEWS_RENDERING = 6


	if args.optim is not None:
		cfg.TRAIN.OPTIM = args.optim

	if args.l2_penalty is not None:
		cfg.TRAIN.L2_PENALTY = args.l2_penalty
	
	if args.lr is not None:
		cfg.TRAIN.LEARNING_RATE = args.lr

	if args.gray:
		cfg.DATASET.GRAYSCALE = True
	if args.loss is not None:
		cfg.TRAIN.LOSS = args.loss

	if args.inp_size is not None:
		cfg.CONST.IMG_W = args.inp_size
		cfg.CONST.IMG_H = args.inp_size
	if args.weights is not None:
		cfg.DIR.WEIGHTS = args.weights
	if args.ae_weights is not None:
		cfg.DIR.AE_WEIGHTS = args.ae_weights
		
	if args.name is not None:
		cfg.DIR.EXPERIMENT_NAME = args.name
	if args.proj_name is not None:
		cfg.DIR.PROJECT_NAME = args.proj_name
	
	if args.ae:
		cfg.DATASET.AUTOENCODER = True

	if args.bg:
		cfg.DATASET.BACKGROUND = True
	if args.gn:
		cfg.NETWORK.GROUP_NORM = True
	if args.num_nodes is not None:
		cfg.CONST.NODES = args.num_nodes
	if args.OnlySeq:
		cfg.TRAIN.ONLYSEQ = True
	if args.bigData:
		cfg.DATASET.BIGDATA = True
	if args.dropout is not None:
		cfg.NETWORK.DROPOUT = args.dropout
	
	if args.transformer:
		cfg.NETWORK.TRANSFORMER = True
		cfg.NETWORK.TRANSFORMER_NUM_BLOCKS = args.num_blocks
		cfg.NETWORK.TRANSFORMER_NUM_HEADS = args.num_heads
	
	if args.test:
		cfg.TEST.IS_TEST = True
		cfg.TEST.NUM_SAMPLES = args.num_test_samples
	main(cfg)