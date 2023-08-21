import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(torch.nn.Module):
	def __init__(self, c_in, kernel_size=3, padding=1, stride=1):
		super(ResBlock3D, self).__init__()
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv3d(c_in, c_in, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm3d(c_in),
					  torch.nn.ReLU(),
					  torch.nn.Conv3d(c_in, c_in, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm3d(c_in),
					  torch.nn.ReLU(),					  
					  )

	def forward(self, x):
		res = x
		x = self.layers(x)
		out = x + res
		return out

class UNet_Down_Res(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size, padding, stride, dropout=None):
		super(UNet_Down_Res, self).__init__()
		self.dropout = dropout
		self.resblock = ResBlock3D(c_in)#, kernel_size, padding, stride)
		self.conv_down = nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride)
		# self.maxpool = torch.nn.MaxPool3d(kernel_size=2)
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
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, dropout=None):
		super(UNet_Up_Res, self).__init__()
		self.dropout = dropout
		
		self.Conv3D_Up = nn.Sequential(
						nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						nn.BatchNorm3d(c_out),
						nn.ReLU(),
						)
		if self.dropout is not None:
			self.dropout = nn.Dropout3d(dropout)
		self.conv_layer = nn.Conv3d(c_out*2, c_out, kernel_size=3, padding=1, stride=1)
		self.resblock = ResBlock3D(c_out)
	def forward(self, x, skip):
		x = self.Conv3D_Up(x)
		x = torch.cat((x, skip), dim=1)
		x = self.conv_layer(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.resblock(x)
		return x

class UNet3D(torch.nn.Module):
	def __init__(self, cfg):
		super(UNet3D, self).__init__()
		self.cfg = cfg
		self.c_in = 1
		self.c_out = 32
		
		# 32x32x32x1
		self.conv_layer_1 = nn.Sequential(
			nn.Conv3d(self.c_in, self.c_out, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm3d(self.c_out),
			nn.ReLU()
			)
		# 32x32x32x32
		self.conv_down1 = UNet_Down_Res(self.c_out, self.c_out, kernel_size=3, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 16x16x16x32
		self.conv_down2 = UNet_Down_Res(self.c_out, self.c_out*2, kernel_size=3, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 8x8x8x64
		self.conv_down3 = UNet_Down_Res(self.c_out*2, self.c_out*4, kernel_size=3, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 4x4x4x128
		# self.conv_down4 = UNet_Down_Res(self.c_out*4, self.c_out*8, kernel_size=3, padding=1, stride=1, dropout=cfg.NETWORK.DROPOUT)

		self.conv_up1 = UNet_Up_Res(self.c_out*4, self.c_out*2, kernel_size=4, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 8x8x8x64
		self.conv_up2 = UNet_Up_Res(self.c_out*2, self.c_out, kernel_size=4, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 16x16x16x32
		self.conv_up3 = UNet_Up_Res(self.c_out, self.c_out, kernel_size=4, padding=1, stride=2, dropout=cfg.NETWORK.DROPOUT)
		# 32x32x32x32
		self.final_conv_layer = nn.Sequential(
			nn.Conv3d(self.c_out, self.c_in, kernel_size=1, padding=0, stride=1),
			nn.Sigmoid()
		)
		# 32x32x32x1
	
	def forward(self, x):
		x = x.unsqueeze(1) # 32x1x32x32x32
		x1 = self.conv_layer_1(x) # 32x32x32x32
		x2 = self.conv_down1(x1) # 16x16x16x32
		x3 = self.conv_down2(x2) # 8x8x8x64
		x4 = self.conv_down3(x3) # 4x4x4x128
		# x5 = self.conv_down4(x4)
		x = self.conv_up1(x4, x3) # 8x8x8x64
		x = self.conv_up2(x, x2) # 16x16x16x32
		x = self.conv_up3(x, x1) # 32x32x32x32
		output = self.final_conv_layer(x) # 32x32x32x1
		return output.squeeze(dim=1)