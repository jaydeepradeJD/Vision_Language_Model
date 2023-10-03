import torch
from torch import nn

class Conv2D(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, padding=1, stride=1, num_groups=8):
		super(Conv2D, self).__init__()
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.GroupNorm(num_groups, c_out),
					  torch.nn.ReLU(),
					  )
		self.res_block = ResBlock2D(c_out, num_groups=num_groups)

	def forward(self, x):
		x = self.layers(x)
		x = self.res_block(x)
		return x


class ResBlock2D(nn.Module):
	def __init__(self, c_in, num_groups=8):
		super(ResBlock2D, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.ReLU(),
			nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			)
		self.relu = nn.ReLU()
	def forward(self, x):
		return self.relu(x + self.layers(x))
		
class Conv2D_Down(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, padding=1, stride=2, num_groups=8, dropout=None):
		super(Conv2D_Down, self).__init__()
		self.dropout = dropout
		self.conv = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.GroupNorm(num_groups, c_out),
					  torch.nn.ReLU(),
					  )
		self.res_block = ResBlock2D(c_out, num_groups=num_groups)
		if self.dropout is not None:
			self.dropout = torch.nn.Dropout2d(dropout)
	def forward(self, x):
		x = self.conv(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.res_block(x)
		
		return x
# Add resnet encoder modules as well as use dictionary for encoding dimension choices to make code more readable

class Encoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Encoder, self).__init__()
		self.cfg = cfg
		self.latent_dims = {16384:1024, 8192:512, 4096:256, 2048:128, 1024:64, 512:32}
		
		if self.cfg.DATASET.GRAYSCALE:
			self.c_in = 1
		else:
			self.c_in = 3

		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:		
			self.c_out = 32 #4 #8 
			self.layer1 = Conv2D_Down(self.c_in, self.c_out, 7, 3, 2, self.cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
			# 32 x 112 x 112
			self.layer2 = Conv2D_Down(self.c_out, self.c_out * 2, 3, 1, 2, num_groups=self.cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
			# 64 x 56 x 56
			self.layer3 = Conv2D_Down(self.c_out * 2, self.c_out * 4, 3, 1, 2, num_groups=self.cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
			# 128 x 28 x 28
			self.layer4 = Conv2D_Down(self.c_out * 4, self.c_out * 8, 3, 1, 2, num_groups=self.cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
			# 256 x 14 x 14
			self.layer5 = Conv2D_Down(self.c_out * 8, self.c_out * 16, 3, 1, 2, num_groups=self.cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
			# 512 x 7 x 7
			self.layer6 = Conv2D_Down(self.c_out * 16, self.latent_dims[self.cfg.NETWORK.LATENT_DIM], 3, 1, 2, num_groups=self.cfg.NETWORK.GROUP_NORM_GROUPS)
			# [1024, 512, 256, 128, 64, 32] x 4 x 4

	def forward(self, rendering_images):
		# print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
		rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
		rendering_images = torch.split(rendering_images, 1, dim=0)
		image_features = []

		for img in rendering_images:
			# if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:
			# 	features = self.layer11(img.squeeze(dim=0))
			# 	features = self.layer1(features)	
			# if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:
			features = self.layer1(img.squeeze(dim=0))	
			features = self.layer2(features)
			features = self.layer3(features)
			features = self.layer4(features)
			features = self.layer5(features)
			encoded_features = self.layer6(features)
			image_features.append(encoded_features)

		image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
		# print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7]) / torch.Size([batch_size, n_views, 512, 4, 4]) / torch.Size([batch_size, n_views, 256, 4, 4])
		
		# image_features = torch.cat(image_features, dim=1) # this is when not using pix2vox  
		# batch_size x 160 (32x5) x 4 x 4 
		return image_features


# Encoder for input size 1080x1080

'''
		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:		
			self.c_out = 16 
			self.layer11 = torch.nn.Sequential(torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=11, padding=0, stride=4),
						  torch.nn.GroupNorm(self.c_out),
						  torch.nn.ReLU())
			# 268 x 268 x 16		

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer1 = Conv2D_Down(self.c_in, self.c_out, 7, 1, 1)
			# 132 x 132 x 32

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer2 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 66 x 66 x 64
			
			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer3 = Conv2D_Down(self.c_in, self.c_out, 3, 0, 1)
			# 32 x 32 x 128
			
			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer4 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			 # 16 x 16 x 256

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer5 = Conv2D_Down(self.c_in, self.c_out, 3, 0, 1) # Might need to check the dimensions here when input is 1080 x 1080
			 # 7 x 7 x 512


'''
