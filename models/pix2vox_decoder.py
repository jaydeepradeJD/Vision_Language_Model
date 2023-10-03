import torch
from torch import nn

class ResBlock3D(nn.Module):
	def __init__(self, c_in, num_groups=8):
		super(ResBlock3D, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.ReLU(),
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.ReLU()
			)
	def forward(self, x):
		return x + self.layers(x)
	
class Conv3D_Up_Res(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, num_groups=8, dropout=None):
		super(Conv3D_Up_Res, self).__init__()
		self.dropout = dropout
		
		self.Conv3D_Up = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.GroupNorm(num_groups, c_out),
						nn.ReLU(),
						)
		if self.dropout is not None:
			self.dropout = nn.Dropout3d(dropout)

		self.ResBlock3D = ResBlock3D(c_out, num_groups=num_groups)
	def forward(self, x):
		x = self.Conv3D_Up(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.ResBlock3D(x)
		return x

class Decoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()
		self.cfg = cfg
		self.c_in = None
		self.c_out = 512 #256 #128 #512
		# Layer Definition
		self.latent_dims_imgs = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		self.latent_dims_imgs_seq = {16384:2208, 8192:1184, 4096:672, 2048:416, 1024:288, 512:224}
		
		# if self.cfg.NETWORK.USE_SEQ:
		# 	self.latent_dims_seq = {16384:2208, 8192:1184, 4096:672, 2048:416, 1024:288, 512:224}
		# 	self.latent_dims_imgs = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		
		# else:
		# 	self.latent_dims = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		
		if self.cfg.NETWORK.USE_SEQ:
			self.c_in = self.latent_dims_imgs_seq[self.cfg.NETWORK.LATENT_DIM]
		else:
			self.c_in = self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM]
		self.layer1 = Conv3D_Up_Res(self.c_in, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT) 
		# 512 x 4 x 4 x 4
		self.layer2 = Conv3D_Up_Res(self.c_out, self.c_out//2, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 256 x 8 x 8 x 8
		self.layer3 = Conv3D_Up_Res(self.c_out//2, self.c_out//4, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 128 x 16 x 16 x 16
		self.layer4 = Conv3D_Up_Res(self.c_out//4, self.c_out//16, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 32 x 32 x 32

		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out//16, 1, kernel_size=3, padding=1, stride=1),
			torch.nn.Sigmoid()
		)
		# 1 x 32 x 32 x 32

	def forward(self, image_features, seq_emd=None):
		if seq_emd is not None:
			seq_emd = seq_emd.view(-1, 160, 2, 2, 2)

		image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
		image_features = torch.split(image_features, 1, dim=0)
		raw_features = []
		gen_volumes = []
		for features in image_features:
			# gen_volume = features.view(-1, self.latent_dims[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
			# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2
			if seq_emd is not None:
				gen_volume = features.view(-1, self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
				# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2	
				gen_volume = torch.cat((gen_volume, seq_emd), dim=1)
				# [2208, 1184, 672, 416, 288, 224] x 2 x 2 x 2
			else:
				gen_volume = features.view(-1, self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
				# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2
			gen_volume = self.layer1(gen_volume)
			# 512 x 4 x 4 x 4
			gen_volume = self.layer2(gen_volume)
			# 256 x 8 x 8 x 8
			gen_volume = self.layer3(gen_volume)
			# 128 x 16 x 16 x 16
			gen_volume = self.layer4(gen_volume)
			# 32 x 32 x 32 x 32
			raw_feature = gen_volume
			
			gen_volume = self.layer5(gen_volume)
			# 1 x 32 x 32 x 32

			raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
			# 33 x 32 x 32 x 32
			gen_volumes.append(torch.squeeze(gen_volume, dim=1))

			raw_features.append(raw_feature)

		gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
		# batch_size x n_views x 32 x 32 x 32
		raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
		# batch_size x n_views x 33 x 32 x 32 x 32
		return raw_features, gen_volumes


class Decoder_64(torch.nn.Module):
	def __init__(self, cfg):
		super(Decoder_64, self).__init__()
		self.cfg = cfg
		self.c_in = None
		self.c_out = 512 #512 #256 #128 #512
		# Layer Definition
		self.latent_dims_imgs = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		self.latent_dims_imgs_seq = {16384:2208, 8192:1184, 4096:672, 2048:416, 1024:288, 512:224}
		
		# if self.cfg.NETWORK.USE_SEQ:
		# 	self.latent_dims_seq = {16384:2208, 8192:1184, 4096:672, 2048:416, 1024:288, 512:224}
		# 	self.latent_dims_imgs = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		
		# else:
		# 	self.latent_dims = {16384:2048, 8192:1024, 4096:512, 2048:256, 1024:128, 512:64}
		
		if self.cfg.NETWORK.USE_SEQ:
			self.c_in = self.latent_dims_imgs_seq[self.cfg.NETWORK.LATENT_DIM]
		else:
			self.c_in = self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM]
		self.layer1 = Conv3D_Up_Res(self.c_in, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT) 
		# 512 x 4 x 4 x 4
		self.layer2 = Conv3D_Up_Res(self.c_out, self.c_out//2, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 256 x 8 x 8 x 8
		self.layer3 = Conv3D_Up_Res(self.c_out//2, self.c_out//4, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 128 x 16 x 16 x 16
		self.layer4 = Conv3D_Up_Res(self.c_out//4, self.c_out//16, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 32 x 32 x 32
		self.layer5 = Conv3D_Up_Res(self.c_out//16, self.c_out//16, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 64 x 64 x 64


		self.layer6 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out//16, 1, kernel_size=3, padding=1, stride=1),
			torch.nn.Sigmoid()
		)
		# 1 x 32 x 32 x 32

	def forward(self, image_features, seq_emd=None):
		if seq_emd is not None:
			seq_emd = seq_emd.view(-1, 160, 2, 2, 2)

		image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
		image_features = torch.split(image_features, 1, dim=0)
		raw_features = []
		gen_volumes = []
		for features in image_features:
			# gen_volume = features.view(-1, self.latent_dims[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
			# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2
			if seq_emd is not None:
				gen_volume = features.view(-1, self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
				# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2	
				gen_volume = torch.cat((gen_volume, seq_emd), dim=1)
				# [2208, 1184, 672, 416, 288, 224] x 2 x 2 x 2
			else:
				gen_volume = features.view(-1, self.latent_dims_imgs[self.cfg.NETWORK.LATENT_DIM], 2, 2, 2)
				# [2048, 1024, 512, 256, 128, 64] 2 x 2 x 2
			gen_volume = self.layer1(gen_volume)
			# 512 x 4 x 4 x 4
			gen_volume = self.layer2(gen_volume)
			# 256 x 8 x 8 x 8
			gen_volume = self.layer3(gen_volume)
			# 128 x 16 x 16 x 16
			gen_volume = self.layer4(gen_volume)
			# 32 x 32 x 32 x 32
			gen_volume = self.layer5(gen_volume)
			# 32 x 64 x 64 x 64
			raw_feature = gen_volume
			
			gen_volume = self.layer6(gen_volume)
			# 1 x 64 x 64 x 64 

			raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
			# 33 x 64 x 64 x 64
			gen_volumes.append(torch.squeeze(gen_volume, dim=1))

			raw_features.append(raw_feature)

		gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
		# batch_size x n_views x 64 x 64 x 64
		raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
		# batch_size x n_views x 33 x 64 x 64 x 64
		return raw_features, gen_volumes


