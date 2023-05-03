import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import ProteinDataset, SequenceDataset, ProteinAutoEncoderDataset
from VL_trainer import Model, OnlySeqModel, AutoEncoder, PretrainedAE_Model
from config import cfg
import utils.data_transforms

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
	if not cfg.DATASET.AUTOENCODER:
		train_dataset = ProteinDataset('train', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, train_transforms, grayscale=cfg.DATASET.GRAYSCALE)
		val_dataset = ProteinDataset('val', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, val_transforms, grayscale=cfg.DATASET.GRAYSCALE)
		# test_dataset = utils.data_loaders.ProteinDataset('test', cfg.CONST.N_VIEWS_RENDERING, cfg.CONST.REP, test_transforms, grayscale=cfg.DATASET.GRAYSCALE)
		# test_dataset = None

	if cfg.DATASET.AUTOENCODER:
		train_dataset = ProteinAutoEncoderDataset('train', train_transforms, background=cfg.DATASET.BACKGROUND)
		val_dataset = ProteinAutoEncoderDataset('val', val_transforms, background=cfg.DATASET.BACKGROUND)
		# test_dataset = ProteinAutoEncoderDataset('test', test_transforms)
	

	# train_dataset = SequenceDataset('train', cfg.CONST.REP)
	# val_dataset = SequenceDataset('val', cfg.CONST.REP)
	

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

	if not cfg.DATASET.AUTOENCODER:
		if cfg.DIR.AE_WEIGHTS is not None:
			ae = AutoEncoder.load_from_checkpoint(cfg.DIR.AE_WEIGHTS, cfg=cfg)
			ae.eval()
			model = PretrainedAE_Model(cfg, pretrained_model=ae)
		else:
			model = Model(cfg)
	# model = OnlySeqModel(cfg)
	
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
						 accelerator='gpu', 
						 # strategy='ddp',
						 callbacks=[checkpoint],
						 logger=[logger, wandb_logger], 
						 max_epochs=cfg.CONST.NUM_EPOCHS, 
						 default_root_dir=cfg.DIR.OUT_PATH, 
						 fast_dev_run=cfg.TRAIN.DEBUG)

	# Training
	
	trainer.fit(model, train_data_loader, val_data_loader, ckpt_path=cfg.DIR.WEIGHTS)

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


	if args.gray:
		cfg.DATASET.GRAYSCALE = True

	if args.loss is not None:
	   cfg.TRAIN.LOSS = args.loss
	
	if args.optim is not None:
		cfg.TRAIN.OPTIM = args.optim

	if args.l2_penalty is not None:
		cfg.TRAIN.L2_PENALTY = args.l2_penalty
	
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
	main(cfg)