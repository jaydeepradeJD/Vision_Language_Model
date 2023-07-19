import torch
from torch import nn
from .attention import TransformerBlock

class Conv3D_Up(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, dropout=None):
		super(Conv3D_Up, self).__init__()
		if dropout is None:	
			self.layers = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, stride=1),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						)
		else:
			self.layers = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						nn.Dropout3d(dropout),
						nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, stride=1),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						nn.Dropout3d(dropout),
						)

	def forward(self, x):
		return self.layers(x)
	
	
# class Conv3D_Up(nn.Module):
# 	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2):
# 		super(Conv3D_Up, self).__init__()
# 		self.layers = nn.Sequential(
# 					  nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
# 					  nn.BatchNorm3d(c_out),
# 					  nn.ReLU(),
# 					  )

# 	def forward(self, x):
# 		return self.layers(x)


class Decoder(nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()
		self.cfg = cfg
		self.c_in = 400 #240 #1360 #240
		self.c_out = 512 
		# Layer Definition
						

		self.layer1 = nn.Sequential(
			nn.Conv2d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(self.c_out),
			nn.ReLU(),
			nn.Conv2d(self.c_out, self.c_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(self.c_out),
			nn.ReLU(),
			)
		    # 4x4x512

		self.c_in = 128
		self.c_out = 64
		
		self.layer2 = Conv3D_Up(self.c_in, self.c_out)
		# 8x8x8x64

		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer3 = Conv3D_Up(self.c_in, self.c_out)
		# 16x16x16x32
		
		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer4 = Conv3D_Up(self.c_in, self.c_out)
		# 32x32x32x16


		self.layer5 = nn.Sequential(
			nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		
		x = self.layer1(x)
		x = x.view(-1, 128, 4, 4, 4)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)

class ResBlock3D(nn.Module):
	def __init__(self, c_in):
		super(ResBlock3D, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm3d(c_in),
			nn.ReLU(),
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm3d(c_in),
			nn.ReLU()
			)
	def forward(self, x):
		return x + self.layers(x)
	
class Conv3D_Up_Res(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, dropout=None):
		super(Conv3D_Up_Res, self).__init__()
		self.dropout = dropout
		
		self.Conv3D_Up = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						)
		if self.dropout is not None:
			self.dropout = nn.Dropout3d(dropout)

		self.ResBlock3D = ResBlock3D(c_out)
	def forward(self, x):
		x = self.Conv3D_Up(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.ResBlock3D(x)
		return x

class OnlySeq_Decoder(nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 256 #128 #64 #128 
		# Layer Definition

		self.layer1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm3d(self.c_out),
			nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.layer1.append(nn.Dropout3d(cfg.NETWORK.DROPOUT))
		# 4x4x4x128
			
		self.ResBlock3D_1 = ResBlock3D(self.c_out)
		# if cfg.NETWORK.DROPOUT is not None:
		# 	self.drop1 = nn.Dropout3d(cfg.NETWORK.DROPOUT)
		# 4x4x4x128
		
		# self.c_in = 128 
		# self.c_out = 64 
		self.layers = nn.ModuleList([])	
		for i in range(3):
			self.c_in = self.c_out
			self.c_out //= 2
			self.layers.append(Conv3D_Up_Res(self.c_in, self.c_out, dropout=cfg.NETWORK.DROPOUT))
			

		self.layer5 = nn.Sequential(
			nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)
		if cfg.NETWORK.TRANSFORMER:
			self.transformer_blocks = nn.ModuleList([])
			for _ in range(cfg.NETWORK.TRANSFORMER_NUM_BLOCKS):
				self.transformer_blocks.append(TransformerBlock(128 , cfg.NETWORK.TRANSFORMER_NUM_HEADS)) # (embd_dim, n_head)
				
		else:	
			self.fc1 = nn.Sequential(
				nn.Linear(1280, 1280),
				nn.BatchNorm1d(1280),
				nn.ReLU()
			)
			if cfg.NETWORK.DROPOUT is not None:
				self.fc1.append(nn.Dropout(cfg.NETWORK.DROPOUT))

			self.fc2 = nn.Sequential(
				nn.Linear(1280, 1280),
				nn.BatchNorm1d(1280),
				nn.ReLU()
			)

			if cfg.NETWORK.DROPOUT is not None:
				self.fc2.append(nn.Dropout(cfg.NETWORK.DROPOUT))
		
	def forward(self, x):
		# x: (batch_size, 1280)
		if self.cfg.NETWORK.TRANSFORMER:
			# x = x.view(-1, 1280, 1)
			x = x.view(-1, 10, 128) #(batch_zize, seq_len, emb_dim)
			for transformer in self.transformer_blocks:
				x = transformer(x)
		else:
			x = x.view(-1, 1280)
			x = self.fc1(x)
			x = self.fc2(x)
		
		x = x.view(-1, 20, 4, 4, 4)
		x = self.layer1(x)
		x = self.ResBlock3D_1(x)
		for layer in self.layers:
			x = layer(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)


class OnlySeq_Decoder_with_4point2Mn_params(nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder_with_4point2Mn_params, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 128 #64 #128 
		# Layer Definition
						

		self.layer1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm3d(self.c_out),
			nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.layer1.append(nn.Dropout3d(cfg.NETWORK.DROPOUT))
		# 4x4x512

		self.c_in = 128 #64 #128
		self.c_out = 64 #32 #64
		
		self.layer2 = Conv3D_Up(self.c_in, self.c_out, dropout=cfg.NETWORK.DROPOUT)
		# 8x8x8x64

		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer3 = Conv3D_Up(self.c_in, self.c_out, dropout=cfg.NETWORK.DROPOUT)
		# 16x16x16x32
		
		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer4 = Conv3D_Up(self.c_in, self.c_out, dropout=cfg.NETWORK.DROPOUT)
		# 32x32x32x16


		self.layer5 = nn.Sequential(
			nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)

		self.fc1 = nn.Sequential(
			nn.Linear(1280, 1280),
			nn.BatchNorm1d(1280),
			nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.fc1.append(nn.Dropout(cfg.NETWORK.DROPOUT))

		self.fc2 = nn.Sequential(
			nn.Linear(1280, 1280),
			nn.BatchNorm1d(1280),
			nn.ReLU()
		)

		if cfg.NETWORK.DROPOUT is not None:
			self.fc2.append(nn.Dropout(cfg.NETWORK.DROPOUT))
		
	def forward(self, x):
		x = x.view(-1, 1280)
		x = self.fc1(x)
		x = self.fc2(x)
		
		x = x.view(-1, 20, 4, 4, 4)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)

class OnlySeq_Decoder_previous_4mn_params(nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder_previous_4mn_params, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 128 #64 #128 
		# Layer Definition
						

		self.layer1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm3d(self.c_out),
			nn.ReLU()
		)
		# 4x4x512

		self.c_in = 128 #64 #128
		self.c_out = 64 #32 #64
		
		self.layer2 = Conv3D_Up(self.c_in, self.c_out)
		# 8x8x8x64

		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer3 = Conv3D_Up(self.c_in, self.c_out)
		# 16x16x16x32
		
		self.c_in = self.c_out
		self.c_out //= 2
		
		self.layer4 = Conv3D_Up(self.c_in, self.c_out)
		# 32x32x32x16


		self.layer5 = nn.Sequential(
			nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)

		self.fc1 = nn.Sequential(
			nn.Linear(1280, 1280),
			nn.ReLU()
		)

		self.fc2 = nn.Sequential(
			nn.Linear(1280, 1280),
			nn.ReLU()
		)

	def forward(self, x):
		x = x.view(-1, 1280)
		x = self.fc1(x)
		x = self.fc2(x)
		
		x = x.view(-1, 20, 4, 4, 4)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)
