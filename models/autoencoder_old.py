import torch

class Conv2D_Down(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size, padding, stride):
		super(Conv2D_Down, self).__init__()
	
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),
					  torch.nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),
					  torch.nn.MaxPool2d(kernel_size=2)
					  )

	def forward(self, x):
		return self.layers(x)

class Conv2D(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size=3, padding=1, stride=1):
		super(Conv2D, self).__init__()
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU())

	def forward(self, x):
		return self.layers(x)

class Encoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Encoder, self).__init__()
		self.cfg = cfg
		self.c_in = 3
		self.c_out = 16 #4 #8 
		self.layer1 = Conv2D(self.cfg, self.c_in, self.c_out, kernel_size=5, padding=2, stride=1)
		# 224 x 224 x 16

		self.c_in = self.c_out
		self.c_out = self.c_out * 2
		self.layer2 = Conv2D_Down(self.cfg, self.c_in, self.c_out, kernel_size=5, padding=2, stride=1)
		# 112 x 112 x 32

		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		self.layer3 = Conv2D_Down(self.cfg, self.c_in, self.c_out, 3, 1, 1)
		# 56 x 56 x 64

		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		self.layer4 = Conv2D_Down(self.cfg, self.c_in, self.c_out, 3, 1, 1)
		# 28 x 28 x 128
		
		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		self.layer5 = Conv2D_Down(self.cfg, self.c_in, self.c_out, 3, 1, 1)
		# 14 x 14 x 256
		
		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		self.layer6 = Conv2D_Down(self.cfg, self.c_in, self.c_out, 3, 1, 1)
		 # 7 x 7 x 512
		
		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		if self.cfg.NETWORK.LATENT_DIM == 12544:
			self.layer7 = Conv2D(self.cfg, self.c_in, 256, kernel_size=3, padding=1, stride=1)
			 # 7 x 7 x 256

		if self.cfg.NETWORK.LATENT_DIM == 8192:
			self.layer7 = Conv2D(self.cfg, self.c_in, 512, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 512

		if self.cfg.NETWORK.LATENT_DIM == 4096:
			self.layer7 = Conv2D(self.cfg, self.c_in, 256, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 256
		
		if self.cfg.NETWORK.LATENT_DIM == 2048:
			self.layer7 = Conv2D(self.cfg, self.c_in, 128, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 128
		
		if self.cfg.NETWORK.LATENT_DIM == 1024:
			self.layer7 = Conv2D(self.cfg, self.c_in, 64, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 64
		
		if self.cfg.NETWORK.LATENT_DIM == 512:
			self.layer7 = Conv2D(self.cfg, self.c_in, 32, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 32

	def forward(self, x):

		features = self.layer1(x)
		features = self.layer2(features)
		features = self.layer3(features)
		features = self.layer4(features)
		features = self.layer5(features)
		features = self.layer6(features)
		img_features = self.layer7(features)
		
		return img_features
 
class Conv2D_Up(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size=4, padding=1, stride=2):
		super(Conv2D_Up, self).__init__()
		
		self.layers = torch.nn.Sequential(
					  torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),
					  torch.nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),
					  
					  )

	def forward(self, x):
		return self.layers(x)

class Decoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()
		self.cfg = cfg
		# Layer Definition
		
		if self.cfg.NETWORK.LATENT_DIM == 4096:
			self.c_in = 256

		self.c_out = 512 
		self.layer1 = Conv2D_Up(self.cfg, self.c_in, self.c_out, 4, 0, 1)
		# 7 x7 x 512
		
		self.c_in = self.c_out
		self.c_out //= 2
		self.layer2 = Conv2D_Up(self.cfg, self.c_in, self.c_out)
		# 14 x 14 x 256

		self.c_in = self.c_out
		self.c_out //= 2
		self.layer3 = Conv2D_Up(self.cfg, self.c_in, self.c_out)
		# 28 x 28 x 128


		self.c_in = self.c_out
		self.c_out //= 2
		self.layer4 = Conv2D_Up(self.cfg, self.c_in, self.c_out)
		# 56 x 56 x 64


		self.c_in = self.c_out
		self.c_out //= 2
		self.layer5 = Conv2D_Up(self.cfg, self.c_in, self.c_out)
		# 112 x 112 x 32


		self.c_in = self.c_out
		self.c_out //= 2
		self.layer6 = Conv2D_Up(self.cfg, self.c_in, self.c_out, 6, 2, 2)
		# 224 x 224 x 16

		
		self.layer7 = torch.nn.Sequential(
			torch.nn.Conv2d(self.c_out, 1, kernel_size=3, stride=1, padding=1),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		
		output = self.layer7(x)
		
		return output