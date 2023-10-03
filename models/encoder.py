import torch

class Conv2D_Down(torch.nn.Module):
	def __init__(self, c_in, c_out, kernel_size, padding, stride):
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
	def __init__(self, c_in, c_out, kernel_size=3, padding=1, stride=1):
		super(Conv2D, self).__init__()
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),
					  torch.nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1),
					  torch.nn.BatchNorm2d(c_out),
					  torch.nn.ReLU(),					  
					  )

	def forward(self, x):
		return self.layers(x)

# Add resnet encoder modules as well as use dictionary for encoding dimension choices to make code more readable

class Encoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Encoder, self).__init__()
		self.cfg = cfg
		if self.cfg.DATASET.GRAYSCALE:
			self.c_in = 1
		else:
			self.c_in = 3

		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:		
			self.c_out = 32 #4 #8 
			self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=5, padding=2, stride=2),
						  torch.nn.BatchNorm2d(self.c_out),
						  torch.nn.ReLU(),
						  torch.nn.Conv2d(self.c_out, self.c_out, kernel_size=3, padding=1, stride=1),
					  	  torch.nn.BatchNorm2d(self.c_out),
					      torch.nn.ReLU(),
					      )

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer2 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 56 x 56 x 64

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer3 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 28 x 28 x 128
			
			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer4 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 14 x 14 x 256
			
			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer5 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			 # 7 x 7 x 512
		
		self.c_in = self.c_out
		self.c_out = self.c_out * 2

		if self.cfg.NETWORK.LATENT_DIM == 12544:
			self.layer6 = Conv2D(self.c_in, 256, kernel_size=3, padding=1, stride=1)
			 # 7 x 7 x 256

		if self.cfg.NETWORK.LATENT_DIM == 8192:
			self.layer6 = Conv2D(self.c_in, 512, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 512

		if self.cfg.NETWORK.LATENT_DIM == 4096:
			self.layer6 = Conv2D(self.c_in, 256, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 256
		
		if self.cfg.NETWORK.LATENT_DIM == 2048:
			self.layer6 = Conv2D(self.c_in, 128, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 128
		
		if self.cfg.NETWORK.LATENT_DIM == 1024:
			self.layer6 = Conv2D(self.c_in, 64, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 64
		
		if self.cfg.NETWORK.LATENT_DIM == 512:
			self.layer6 = Conv2D(self.c_in, 32, kernel_size=3, padding=1, stride=2)
			# 4 x 4 x 32

		#self.final_layer = Conv2D(256, 32, kernel_size=3, padding=1, stride=1)
		# 4x4x32
		self.final_layer = Conv2D(256, 64, kernel_size=3, padding=1, stride=1)
		# 4x4x64


	def forward(self, rendering_images):
		# print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
		rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
		rendering_images = torch.split(rendering_images, 1, dim=0)

		image_features = []

		for img in rendering_images:
			if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:
				features = self.layer11(img.squeeze(dim=0))
				features = self.layer1(features)	
			if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:
				features = self.layer1(img.squeeze(dim=0))	
			features = self.layer2(features)
			features = self.layer3(features)
			features = self.layer4(features)
			features = self.layer5(features)
			features = self.layer6(features)
			encoded_features = self.final_layer(features)
			image_features.append(encoded_features)

		# image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
		image_features = torch.cat(image_features, dim=1) 
		# batch_size x 160 (32x5) x 4 x 4 
		# print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7]) / torch.Size([batch_size, n_views, 512, 4, 4]) / torch.Size([batch_size, n_views, 256, 4, 4])
		return image_features


# Encoder for input size 1080x1080

'''
		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:		
			self.c_out = 16 
			self.layer11 = torch.nn.Sequential(torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=11, padding=0, stride=4),
						  torch.nn.BatchNorm2d(self.c_out),
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

class Matrix_Encoder(torch.nn.Module):
	def __init__(self, cfg):
		super(Matrix_Encoder, self).__init__()
		self.cfg = cfg
		if self.cfg.DATASET.GRAYSCALE:
			self.c_in = 1
		else:
			self.c_in = 3

		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:		
			self.c_out = 32 #4 #8 
			self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=5, padding=2, stride=2),
						  torch.nn.BatchNorm2d(self.c_out),
						  torch.nn.ReLU(),
						  torch.nn.Conv2d(self.c_out, self.c_out, kernel_size=3, padding=1, stride=1),
					  	  torch.nn.BatchNorm2d(self.c_out),
					      torch.nn.ReLU(),
					      )

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer2 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 56 x 56 x 64

			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer3 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 28 x 28 x 128
			
			self.c_in = self.c_out
			self.c_out = self.c_out * 2

			self.layer4 = Conv2D_Down(self.c_in, self.c_out, 3, 1, 1)
			# 14 x 14 x 256

			self.final_layer = Conv2D(256, 64, kernel_size=3, padding=1, stride=1)
			


	def forward(self, rendering_images):
		# print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
		rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
		rendering_images = torch.split(rendering_images, 1, dim=0)
		image_features = []

		for img in rendering_images:
			if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:
				features = self.layer11(img.squeeze(dim=0))
				features = self.layer1(features)	
			if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 224:
				features = self.layer1(img.squeeze(dim=0))	
			features = self.layer2(features)
			features = self.layer3(features)
			features = self.layer4(features)
			
			encoded_features = self.final_layer(features)
			image_features.append(encoded_features)

		# image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
		image_features = torch.cat(image_features, dim=1) 
		# batch_size x 160 (32x5) x 4 x 4 
		# print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7]) / torch.Size([batch_size, n_views, 512, 4, 4]) / torch.Size([batch_size, n_views, 256, 4, 4])
		return image_features


# Encoder for input size 1080x1080

'''
		if self.cfg.CONST.IMG_W == self.cfg.CONST.IMG_H == 1080:		
			self.c_out = 16 
			self.layer11 = torch.nn.Sequential(torch.nn.Conv2d(self.c_in, self.c_out, kernel_size=11, padding=0, stride=4),
						  torch.nn.BatchNorm2d(self.c_out),
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
