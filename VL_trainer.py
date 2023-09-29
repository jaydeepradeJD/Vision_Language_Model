import pytorch_lightning as pl
import torch
from models.encoder import Encoder, Matrix_Encoder
from models.decoder import Decoder, Matrix_Decoder, OnlySeq_Decoder
from models.autoencoder import Encoder as AE_encoder
from models.autoencoder import Decoder as AE_decoder

from models.autoencoder_old import Encoder as AE_encoder_old
from models.autoencoder_old import Decoder as AE_decoder_old
from models.unet import UNet3D
from models.mlp import MLP


class Model(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(Model, self).__init__()
		self.cfg = cfg
		self.encoder = Encoder(self.cfg)
		self.decoder = Decoder(self.cfg)
		self.mlp = MLP(self.cfg)

		self.loss = torch.nn.BCELoss()
		
	def forward(self, imgs, seq_emd):
		img_features = self.encoder(imgs)
		# batch_size,320(64x5),4,4  or batch_size,160(32*5),4,4
		emd_features = self.mlp(seq_emd)
		# batch_size,1280
		emd_features = emd_features.view(-1, 80, 4, 4)
		combined_features = torch.cat([img_features, emd_features], dim=1)
		predicted = self.decoder(combined_features)

		return predicted

	def training_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE
		weight_decay = self.cfg.TRAIN.L2_PENALTY
		
		opt = torch.optim.Adam(list(self.encoder.parameters())+
			list(self.decoder.parameters())+
			list(self.mlp.parameters()), lr, weight_decay=weight_decay)
			  
		return opt


class PretrainedAE_Model(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg, pretrained_model=None):
		super(PretrainedAE_Model, self).__init__()
		self.cfg = cfg
		self.encoder = pretrained_model.encoder
		self.decoder = Decoder(self.cfg)
		self.mlp = MLP(self.cfg)

		self.loss = torch.nn.BCELoss()
		
	def encode_img(self, rendering_images):
		# print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
		rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
		rendering_images = torch.split(rendering_images, 1, dim=0)
		image_features = []

		for img in rendering_images:
			encoded_features = self.encoder(img.squeeze(dim=0))
			image_features.append(encoded_features)

		# image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
		image_features = torch.cat(image_features, dim=1) 
		# batch_size x 160 (32x5) x 4 x 4 
		# print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7]) / torch.Size([batch_size, n_views, 512, 4, 4]) / torch.Size([batch_size, n_views, 256, 4, 4])
		return image_features

	def forward(self, imgs, seq_emd):
		img_features = self.encode_img(imgs)
		# batch_size,1280,4,4
		emd_features = self.mlp(seq_emd)
		# batch_size,1280
		emd_features = emd_features.view(-1, 80, 4, 4)
		combined_features = torch.cat([img_features, emd_features], dim=1)
		predicted = self.decoder(combined_features)

		return predicted

	def training_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE
		opt = torch.optim.Adam(list(self.decoder.parameters())+
			list(self.mlp.parameters()), lr)
			  
		return opt


class OnlySeqModel(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(OnlySeqModel, self).__init__()
		self.decoder = OnlySeq_Decoder(cfg)
		if cfg.NETWORK.DISCRIMINATOR:
			self.disc = UNet3D(cfg)
		self.cfg = cfg
		self.loss = torch.nn.BCELoss()
		# self.loss = torch.nn.L1Loss()
		# self.loss2 = torch.nn.MSELoss()
		
		
	def forward(self, seq_emd, target):
		
		# batch_size,1280
		# emd_features = seq_emd.view(-1, 20, 4, 4, 4)
		
		predicted = self.decoder(seq_emd)
		if self.cfg.NETWORK.DISCRIMINATOR:
			reconstructed = self.disc(predicted)
			loss1 = self.loss(predicted, target)
			loss2 = self.loss(reconstructed, target)
			total_loss = loss1 + loss2
			return predicted, reconstructed, loss1, loss2, total_loss
		else:
			loss = self.loss(predicted, target)
			return predicted, loss
		
	def training_step(self, batch, batch_idx):
		seq_emd, target = batch 
		
		if self.cfg.NETWORK.DISCRIMINATOR:
			predicted, reconstructed, loss1, loss2, loss = self.forward(seq_emd, target)
			self.log_dict({'train/loss1':loss1, 'train/loss2':loss2, 'train/total_loss':loss})
		else:
			predicted, loss = self.forward(seq_emd, target)
			self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		seq_emd, target = batch 
		
		if self.cfg.NETWORK.DISCRIMINATOR:
			predicted, reconstructed, loss1, loss2, loss = self.forward(seq_emd, target)
			self.log_dict({'val/loss1':loss1, 'val/loss2':loss2, 'val/total_loss':loss})
		else:
			predicted, loss = self.forward(seq_emd, target)
			self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE #0.001 #0.0003
		weight_decay = self.cfg.TRAIN.L2_PENALTY
		# opt = torch.optim.Adam(list(self.decoder.parameters()), lr)
		opt = torch.optim.Adam(list(self.decoder.parameters()), lr, weight_decay=weight_decay)
			  
		return opt

class AutoEncoder(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(AutoEncoder, self).__init__()
		self.cfg = cfg
		self.encoder = AE_encoder(cfg)
		self.decoder = AE_decoder(cfg)
		
		# self.loss = torch.nn.BCELoss()
		# self.loss = torch.nn.L1Loss()
		self.loss = torch.nn.MSELoss()
		
		
	def forward(self, img):
		img_features = self.encoder(img)
		predicted = self.decoder(img_features)
		return predicted

	def training_step(self, batch, batch_idx):
		target = batch 
		predicted = self.forward(target)
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		target = batch 
		predicted = self.forward(target)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE #0.003
		# opt = torch.optim.Adam(list(self.decoder.parameters()), lr)
		opt = torch.optim.Adam(list(self.encoder.parameters())+
			list(self.decoder.parameters()), lr)
			  
		return opt

class AutoEncoder_old(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(AutoEncoder_old, self).__init__()
		self.cfg = cfg
		self.encoder = AE_encoder_old(cfg)
		self.decoder = AE_decoder_old(cfg)
		
		# self.loss = torch.nn.BCELoss()
		# self.loss = torch.nn.L1Loss()
		self.loss = torch.nn.MSELoss()
		
		
	def forward(self, img):
		img_features = self.encoder(img)
		predicted = self.decoder(img_features)
		return predicted

	def training_step(self, batch, batch_idx):
		target = batch 
		predicted = self.forward(target)
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		target = batch 
		predicted = self.forward(target)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE #0.003
		# opt = torch.optim.Adam(list(self.decoder.parameters()), lr)
		opt = torch.optim.Adam(list(self.encoder.parameters())+
			list(self.decoder.parameters()), lr)
			  
		return opt

class Matrix_Model(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(Matrix_Model, self).__init__()
		self.cfg = cfg
		self.encoder = Encoder(self.cfg)
		self.decoder = Matrix_Decoder(self.cfg)
		self.mlp = MLP(self.cfg)

		self.loss = torch.nn.BCELoss()
		
	def forward(self, img_a, img_b):
		img_features_a = self.encoder(img_a)
		img_features_b = self.encoder(img_b)
		# batch_size,320(64x5),4,4  or batch_size,160(32*5),4,4
		# batch_size,1280
		combined_features = torch.cat([img_features_a, img_features_b], dim=1)
		predicted = self.decoder(combined_features)

		return predicted

	def training_step(self, batch, batch_idx):
		img_a, img_b, target = batch 
		
		predicted = self.forward(img_a, img_b)
		
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		img_a, img_b, target = batch 
		
		predicted = self.forward(img_a, img_b)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE
		weight_decay = self.cfg.TRAIN.L2_PENALTY
		
		opt = torch.optim.Adam(list(self.encoder.parameters())+
			list(self.decoder.parameters())+
			list(self.mlp.parameters()), lr, weight_decay=weight_decay)
			  
		return opt