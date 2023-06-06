import torch

class Conv3D_Up(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, dropout=None):
		super(Conv3D_Up, self).__init__()
		if dropout is None:	
			self.layers = torch.nn.Sequential(
						torch.nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						torch.nn.BatchNorm3d(c_out),
						torch.nn.ReLU(),
						torch.nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, stride=1),
						torch.nn.BatchNorm3d(c_out),
						torch.nn.ReLU(),
						)
		else:
			self.layers = torch.nn.Sequential(
						torch.nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						torch.nn.BatchNorm3d(c_out),
						torch.nn.ReLU(),
						torch.nn.Dropout3d(dropout),
						torch.nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, stride=1),
						torch.nn.BatchNorm3d(c_out),
						torch.nn.ReLU(),
						torch.nn.Dropout3d(dropout),
						)

	def forward(self, x):
		return self.layers(x)
	
	
# class Conv3D_Up(torch.nn.Module):
# 	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2):
# 		super(Conv3D_Up, self).__init__()
# 		self.layers = torch.nn.Sequential(
# 					  torch.nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
# 					  torch.nn.BatchNorm3d(c_out),
# 					  torch.nn.ReLU(),
# 					  )

# 	def forward(self, x):
# 		return self.layers(x)


class Decoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()
		self.cfg = cfg
		self.c_in = 400 #240 #1360 #240
		self.c_out = 512 
		# Layer Definition
						

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(self.c_out),
			torch.nn.ReLU(),
			torch.nn.Conv2d(self.c_out, self.c_out, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(self.c_out),
			torch.nn.ReLU(),
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


		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		
		x = self.layer1(x)
		x = x.view(-1, 128, 4, 4, 4)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)

class ResBlock3D(torch.nn.Module):
	def __init__(self, c_in):
		super(ResBlock3D, self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm3d(c_in),
			torch.nn.ReLU(),
			torch.nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm3d(c_in),
			torch.nn.ReLU()
			)
	def forward(self, x):
		res = x
		x = self.layers(x)
		x += res
		return x
	
class Conv3D_Up_Res(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size=4, padding=1, stride=2, dropout=None):
		super(Conv3D_Up_Res, self).__init__()
		self.dropout = dropout
		if self.dropout is not None:
			self.dropout = torch.nn.Dropout3d(dropout)

		self.Conv3D_Up = torch.nn.Sequential(
						torch.nn.ConvTranspose3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
						torch.nn.BatchNorm3d(c_out),
						torch.nn.ReLU(),
						)
		self.ResBlock3D = ResBlock3D(c_out)
	def forward(self, x):
		if self.dropout is not None:
			x = self.dropout(x)
		x = self.Conv3D_Up(x)
		x = self.ResBlock3D(x)
		return self.layers(x)

class OnlySeq_Decoder(torch.nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 128 #64 #128 
		# Layer Definition

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm3d(self.c_out),
			torch.nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.layer1.append(torch.nn.Dropout3d(cfg.NETWORK.DROPOUT))
		# 4x4x4x128
			
		self.ResBlock3D_1 = ResBlock3D(self.c_in)
		if cfg.NETWORK.DROPOUT is not None:
			self.drop1 = torch.nn.Dropout3d(cfg.NETWORK.DROPOUT)
		# 4x4x4x128
		
		self.c_in = 128 
		self.c_out = 64 
		self.layers = torch.nn.ModuleList([])	
		for i in range(3):
			self.layers.append(torch.nn.Sequential([Conv3D_Up_Res(self.c_in, self.c_out, dropout=cfg.NETWORK.DROPOUT)]))
			self.c_in = self.c_out
			self.c_out //= 2


		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)

		self.fc1 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.BatchNorm1d(1280),
			torch.nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.fc1.append(torch.nn.Dropout(cfg.NETWORK.DROPOUT))

		self.fc2 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.BatchNorm1d(1280),
			torch.nn.ReLU()
		)

		if cfg.NETWORK.DROPOUT is not None:
			self.fc2.append(torch.nn.Dropout(cfg.NETWORK.DROPOUT))
		
	def forward(self, x):
		x = x.view(-1, 1280)
		x = self.fc1(x)
		x = self.fc2(x)
		
		x = x.view(-1, 20, 4, 4, 4)
		x = self.layer1(x)
		x = self.ResBlock3D_1(x)
		x = self.layers(x)
		output = self.layer5(x)
		
		return output.squeeze(dim=1)


class OnlySeq_Decoder_with_4point2Mn_params(torch.nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder_with_4point2Mn_params, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 128 #64 #128 
		# Layer Definition
						

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm3d(self.c_out),
			torch.nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.layer1.append(torch.nn.Dropout3d(cfg.NETWORK.DROPOUT))
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


		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)

		self.fc1 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.BatchNorm1d(1280),
			torch.nn.ReLU()
		)
		if cfg.NETWORK.DROPOUT is not None:
			self.fc1.append(torch.nn.Dropout(cfg.NETWORK.DROPOUT))

		self.fc2 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.BatchNorm1d(1280),
			torch.nn.ReLU()
		)

		if cfg.NETWORK.DROPOUT is not None:
			self.fc2.append(torch.nn.Dropout(cfg.NETWORK.DROPOUT))
		
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

class OnlySeq_Decoder_previous_4mn_params(torch.nn.Module):
	def __init__(self, cfg):
		super(OnlySeq_Decoder_previous_4mn_params, self).__init__()
		self.cfg = cfg
		self.c_in = 20
		self.c_out = 128 #64 #128 
		# Layer Definition
						

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_in, self.c_out, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm3d(self.c_out),
			torch.nn.ReLU()
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


		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv3d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)

		self.fc1 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.ReLU()
		)

		self.fc2 = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.ReLU()
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
