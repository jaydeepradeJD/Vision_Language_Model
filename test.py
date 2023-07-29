import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import ProteinDataset
from VL_trainer import Model, OnlySeqModel
from config import cfg
import utils.data_transforms
from utils.plotting import visMC

def test(model, train_data_loader, val_data_loader, weight_path=None, save_dir=None):
	save_dir = save_dir + '/' + weight_path.split('/')[-3] + '/viz_results'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	print(cfg)
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# model = Model.load_from_checkpoint('/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/tensorboard_logs/version_4/last.ckpt',
	# 									cfg=cfg)
	# model = OnlySeqModel.load_from_checkpoint(weight_path,
	# 									cfg=cfg)
	model = model.load_from_checkpoint(weight_path,
										cfg=cfg)

	model = model.to(device)
	model.eval()

	train_ious = []
	for idx, sample in enumerate(train_data_loader):
		seq_emd, target = sample
		# imgs, seq_emd, target = sample
		# imgs = imgs.to(device).unsqueeze(dim=0)
		# imgs = imgs.to(device)
		seq_emd = seq_emd.to(device)
		# seq_emd = seq_emd.view(-1, 20, 4, 4, 4)
		
		target = target.to(device)
		predicted = model(seq_emd)
		# iou = compute_iou(target, predicted)
		iou2 = compute_iou_v2(target, predicted)
		train_ious.append(iou2)
		# print(iou,'--'*5 , iou2)
		predicted = predicted.cpu().detach().squeeze().numpy()
		target = target.cpu().detach().squeeze().numpy()
		
		#visMC(target, predicted, idx, path=save_dir+'/train')

	val_ious = []
	for idx, sample in enumerate(val_data_loader):
		seq_emd, target = sample
		
		# imgs, seq_emd, target = sample
		# imgs = imgs.to(device).unsqueeze(dim=0)
		# imgs = imgs.to(device)
		seq_emd = seq_emd.to(device)
		# seq_emd = seq_emd.view(-1, 20, 4, 4, 4)
		
		target = target.to(device)
		predicted = model(seq_emd)
		# predicted = predicted.cpu().detach().squeeze().numpy()
		# target = target.cpu().detach().squeeze().numpy()
		# iou = compute_iou(target, predicted)
		iou2 = compute_iou_v2(target, predicted)
		val_ious.append(iou2)
		# print(iou,'--'*5 , iou2)
		predicted = predicted.cpu().detach().squeeze().numpy()
		target = target.cpu().detach().squeeze().numpy()
		
		#visMC(target, predicted, idx, path=save_dir+'/val')

	print('train_iou = ', np.mean(train_ious), '#'*5, 'val_iou = ', np.mean(val_ious))


def compute_iou(target, predicted, th=0.5):
	# From Pix2Vox++ code
	th = 0.5
	_volume = torch.ge(predicted, th).float()
	intersection = torch.sum(_volume.mul(target)).float()
	union = torch.sum(torch.ge(_volume.add(target), 1)).float()
	iou = (intersection / union).item()
	return iou

def compute_iou_v2(target, predicted, th=0.5):
	# Modified to take a mean of IoUs over each sample IoU
	th = 0.5
	_volume = torch.ge(predicted, th).float()
	intersection = torch.sum(_volume.mul(target), dim=(1,2,3)).float()
	union = torch.sum(torch.ge(_volume.add(target), 1), dim=(1,2,3)).float()
	iou = (intersection / union)
	iou = torch.mean(iou).item()
	return iou

if __name__ == '__main__':

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
	

	# Set up Dataloader
	
	indices = np.arange(10) #random.sample(range(len(train_dataset)), cfg.CONST.BATCH_SIZE*10)
	train_subdataset = torch.utils.data.Subset(train_dataset, indices)
	val_subdataset = torch.utils.data.Subset(val_dataset, indices)

	train_data_loader = torch.utils.data.DataLoader(dataset=train_subdataset,
												  batch_size=cfg.CONST.BATCH_SIZE,
												  num_workers=cfg.CONST.NUM_WORKER,
												  shuffle=False)
	
	val_data_loader = torch.utils.data.DataLoader(dataset=val_subdataset,
												  batch_size=cfg.CONST.BATCH_SIZE,
												  num_workers=cfg.CONST.NUM_WORKER,
												  shuffle=False)

	log_dir = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/Inference'
	weight_path = '/work/mech-ai-scratch/jmsrdgrs/vision_language_model/Vision_Language_Model/logs/transformer_16_4/version_0/last.ckpt'
	test(train_data_loader, val_data_loader, weight_path, log_dir)
	
	# # experiments = os.listdir(log_dir)
	# # experiments = ['Only_SeqEmbeddings', 'Only_SeqEmbeddings_lr_0.0003', 
	# # 'Only_SeqEmbeddings_lr_0.0003_MSE', 'Only_SeqEmbeddings_lr_0.0003_BCE_10xMSE' ]
	
	# experiments = ['Only_SeqEmbeddings_lr_0.0003_less_params']
	# # experiments = ['Only_SeqEmbeddings_lr_0.0003_less_params_v2', 'Only_SeqEmbeddings_lr_0.0003_less_params_v2_L1loss',
	# # 'Only_SeqEmbeddings_lr_0.0003_less_params_v2_L1loss_l2_0.003', 'Only_SeqEmbeddings_lr_0.0003_less_params_v2_l2_0.001']
	
	# # experiments = ['Only_SeqEmbeddings_lr_0.0003_more_params_2fc']
	
	# for e in experiments:
	# 	weight_path = os.path.join(log_dir, e, 'version_0', 'last.ckpt')
	# 	if e == 'tensorboard_logs':
	# 		continue
	# 		weight_path = os.path.join(log_dir, e, 'version_4', 'last.ckpt')
	# 	print('#'*10, ' '*5, e, ' '*5, '#'*10)
	# 	test(train_data_loader, val_data_loader, weight_path)