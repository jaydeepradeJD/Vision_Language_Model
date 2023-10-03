import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
	def __init__(self, c_in, num_groups=8):
		super(ResBlock3D, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.ReLU(),
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			)
		self.relu = nn.ReLU()
	def forward(self, x):
		return self.relu(x + self.layers(x))
	

class UNet_Down_Res(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size, padding, stride, num_groups=8, dropout=None):
		super(UNet_Down_Res, self).__init__()
		self.dropout = dropout
		self.resblock = ResBlock3D(c_in, num_groups=num_groups)#, kernel_size, padding, stride)
		self.conv_down = torch.nn.Sequential(
					  torch.nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.GroupNorm(num_groups, c_out),
					  torch.nn.ReLU(),
					  )
		if self.dropout is not None:
			self.dropout = nn.Dropout3d(dropout)
	
	def forward(self, x):
		x = self.resblock(x)
		if self.dropout is not None:
			x = self.dropout(x)
		# out = self.maxpool(x)
		out = self.conv_down(x)
		
		return out


class UNet_Up_Res(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, num_groups=8, dropout=None):
		super(UNet_Up_Res, self).__init__()
		self.dropout = dropout
		
		self.Conv3D_Up = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.GroupNorm(num_groups, c_out),
						nn.ReLU(),
						)
		if self.dropout is not None:
			self.dropout = nn.Dropout3d(dropout)
		self.conv_layer = nn.Conv3d(c_out*2, c_out, kernel_size=3, padding=1, stride=1)
		self.resblock = ResBlock3D(c_out, num_groups=num_groups)
	def forward(self, x, skip):
		x = self.Conv3D_Up(x)
		x = torch.cat((x, skip), dim=1)
		x = self.conv_layer(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.resblock(x)
		return x

class Refiner(torch.nn.Module):
	def __init__(self, cfg):
		super(Refiner, self).__init__()
		self.cfg = cfg
		self.c_in = 1
		self.c_out = 32
		
		# 1 x 32 x 32 x 32
		self.conv_layer_1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, padding=1, stride=1),
			nn.GroupNorm(self.cfg.NETWORK.GROUP_NORM_GROUPS, self.c_out),
			nn.ReLU()
			)
		# 32 x 32 x 32 x 32
		self.conv_down1 = UNet_Down_Res(self.c_out, self.c_out, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 16 x 16 x 16
		self.conv_down2 = UNet_Down_Res(self.c_out, self.c_out*2, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 64 x 8 x 8 x 8
		self.conv_down3 = UNet_Down_Res(self.c_out*2, self.c_out*4, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 128 x 4 x 4 x 4
		# self.conv_down4 = UNet_Down_Res(self.c_out*4, self.c_out*8, kernel_size=3, padding=1, stride=1, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)

		self.conv_up1 = UNet_Up_Res(self.c_out*4, self.c_out*2, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 64 x 8 x 8 x 8
		self.conv_up2 = UNet_Up_Res(self.c_out*2, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 16 x 16 x 16
		self.conv_up3 = UNet_Up_Res(self.c_out, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 32 x 32 x 32 x 32
		self.final_conv_layer = nn.Sequential(
			nn.Conv3d(self.c_out, self.c_in, kernel_size=3, padding=1, stride=1),
			nn.Sigmoid()
		)
		# 1 x 32 x 32 x 32
	
	def forward(self, x):
		x = x.unsqueeze(1) # 1 x 32 x 32 x 32
		x1 = self.conv_layer_1(x) # 32 x 32 x 32 x 32
		x2 = self.conv_down1(x1) # 32 x 16 x 16 x 16
		x3 = self.conv_down2(x2) # 64 x 8 x 8 x 8
		x4 = self.conv_down3(x3) # 128 x 4 x 4 x 4
		# x5 = self.conv_down4(x4)
		u = self.conv_up1(x4, x3) # 64 x 8 x 8 x 8
		u = self.conv_up2(u, x2) # 32 x 16 x 16 x 16
		u = self.conv_up3(u, x1) # 32 x 32 x 32 x 32
		output = self.final_conv_layer(u) # 1 x 32 x 32 x 32
		return output.squeeze(dim=1)

class Refiner_64(torch.nn.Module):
	def __init__(self, cfg):
		super(Refiner_64, self).__init__()
		self.cfg = cfg
		self.c_in = 1
		self.c_out = 64
		
		# 1 x 64 x 64 x 64
		self.conv_layer_1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, padding=1, stride=1),
			nn.GroupNorm(self.cfg.NETWORK.GROUP_NORM_GROUPS, self.c_out),
			nn.ReLU()
			)
		# 64 x 64 x 64 x 64
		self.conv_down1 = UNet_Down_Res(self.c_out, self.c_out, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 64 x 32 x 32 x 32
		self.conv_down2 = UNet_Down_Res(self.c_out, self.c_out*2, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 128 x 16 x 16 x 16
		self.conv_down3 = UNet_Down_Res(self.c_out*2, self.c_out*4, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 256 x 8 x 8 x 8
		self.conv_down4 = UNet_Down_Res(self.c_out*4, self.c_out*8, 3, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 512 x 4 x 4 x 4
		
		self.conv_up1 = UNet_Up_Res(self.c_out*8, self.c_out*4, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 256 x 8 x 8 x 8
		self.conv_up2 = UNet_Up_Res(self.c_out*4, self.c_out*2, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 128 x 16 x 16 x 16
		self.conv_up3 = UNet_Up_Res(self.c_out*2, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 64 x 32 x 32 x 32
		self.conv_up4 = UNet_Up_Res(self.c_out, self.c_out, 4, 1, 2, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS, dropout=cfg.NETWORK.DROPOUT)
		# 64 x 64 x 64 x 64

		self.final_conv_layer = nn.Sequential(
			nn.Conv3d(self.c_out, self.c_in, kernel_size=3, padding=1, stride=1),
			nn.Sigmoid()
		)
		# 1 x 64 x 64 x 64
	
	def forward(self, x):
		x = x.unsqueeze(1) # 1 x 64 x 64 x 64
		x1 = self.conv_layer_1(x) # 64 x 64 x 64 x 64
		x2 = self.conv_down1(x1) # 64 x 32 x 32 x 32
		x3 = self.conv_down2(x2) # 128 x 16 x 16 x 16
		x4 = self.conv_down3(x3) # 256 x 8 x 8 x 8
		x5 = self.conv_down4(x4) # 512 x 4 x 4 x 4
		u = self.conv_up1(x5, x4) # 256 x 8 x 8 x 8
		u = self.conv_up2(u, x3) # 128 x 16 x 16 x 16
		u = self.conv_up3(u, x2) # 64 x 32 x 32 x 32
		u = self.conv_up4(u, x1) # 64 x 64 x 64 x 64
		output = self.final_conv_layer(u) # 1 x 64 x 64 x 64
		return output.squeeze(dim=1) # 64 x 64 x 64
