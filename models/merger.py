import torch
from torch import nn

class ResBlock3D(nn.Module):
	def __init__(self, c_in, num_groups=8):
		super(ResBlock3D, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.LeakyReLU(0.2),
			nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			nn.GroupNorm(num_groups, c_in),
			nn.LeakyReLU(0.2)
			)
	def forward(self, x):
		return x + self.layers(x)

class Conv_3D(nn.Module):
	def __init__(self, c_in, c_out, kernel_size=3, padding=1, stride=1, num_groups=8):
		super(Conv_3D, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
			nn.GroupNorm(num_groups, c_out),
			nn.LeakyReLU(0.2)
			)
		
	def forward(self, x):
		return self.layer(x)

class Merger(torch.nn.Module):
	def __init__(self, cfg):
		super(Merger, self).__init__()
		self.cfg = cfg

		# Layer Definition
		# self.layer1 = ResBlock3D(33, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS)
		# self.layer2 = ResBlock3D(33, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS)
		# self.layer3 = ResBlock3D(33, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS)
		# self.layer4 = ResBlock3D(33, num_groups=cfg.NETWORK.GROUP_NORM_GROUPS)
		
		self.layer1 = Conv_3D(33, 33, num_groups=3)
		self.layer2 = Conv_3D(33, 33, num_groups=3)
		self.layer3 = Conv_3D(33, 33, num_groups=3)
		self.layer4 = Conv_3D(33, 33, num_groups=3)
		
		self.layer5 = Conv_3D(132, 33, num_groups=3)
		self.layer6 = Conv_3D(33, 1, num_groups=1)


	def forward(self, raw_features, coarse_volumes):
		n_views_rendering = coarse_volumes.size(1)
		raw_features = torch.split(raw_features, 1, dim=1)
		volume_weights = []

		for i in range(n_views_rendering):
			raw_feature = torch.squeeze(raw_features[i], dim=1)
			# 33 x 32 x 32 x 32
			volume_weight1 = self.layer1(raw_feature)
			# 33 x 32 x 32 x 32
			volume_weight2 = self.layer2(volume_weight1)
			# 33 x 32 x 32 x 32
			volume_weight3 = self.layer3(volume_weight2)
			# 33 x 32 x 32 x 32
			volume_weight4 = self.layer4(volume_weight3)
			# 33 x 32 x 32 x 32
			
			volume_weight = self.layer5(torch.cat([
				volume_weight1, volume_weight2, volume_weight3, volume_weight4
			], dim=1))
			# 33 x 32 x 32 x 32
			volume_weight = self.layer6(volume_weight)
			# 1 x 32 x 32 x 32

			volume_weight = torch.squeeze(volume_weight, dim=1)
			# 32 x 32 x 32
			volume_weights.append(volume_weight)
			
		volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
		# batch_size x n_views x 32 x 32 x 32
		volume_weights = torch.softmax(volume_weights, dim=1)
		coarse_volumes = coarse_volumes * volume_weights
		# batch_size x n_views x 32 x 32 x 32
		coarse_volumes = torch.sum(coarse_volumes, dim=1)

		return torch.clamp(coarse_volumes, min=0, max=1)
