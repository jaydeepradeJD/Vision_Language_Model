import os
import cv2
import numpy as np
import random
import torch.utils.data.dataset
import torch


class ProteinDataset(torch.utils.data.Dataset):
	def __init__(self, cfg, dataset_type, n_views_rendering, representation_type='surface_with_inout', transforms=None, grayscale=False, background=False, big_dataset=False, pix2vox=False):
		self.cfg = cfg
		self.dataset_type = dataset_type
		self.n_views_rendering = n_views_rendering
		self.representation_type = representation_type
		self.transforms = transforms
		self.grayscale = grayscale
		self.background = background
		self.big_dataset = big_dataset
		self.pix2vox = pix2vox
		self.IMG_SIZE = self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W
		if self.big_dataset:
			self.representation_type = 'surface_trimesh_voxels'
			if self.cfg.DATASET.FIXED_VIEWS:
				self.representation_type = 'fixed_views'
			self.metadata_path = '/scratch/bbmw/jd23697/scripts_bigData'
			self.seq_embd_path = '/scratch/bbmw/jd23697/AF_swissprot_seq_embds'
			
		else:
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts'
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/PDBs_seq_emds'

		
		if self.dataset_type == 'train':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples.txt')
			else:
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			if self.cfg.DATASET.FIXED_VIEWS:
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples_fixed_views.txt')
			with open(train_samples_filename, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]
				
		
		if self.dataset_type == 'val':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples.txt')
			else:
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			if self.cfg.DATASET.FIXED_VIEWS:
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples_fixed_views.txt')
			with open(val_samples_filename, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]
			
		if self.dataset_type == 'test':
			with open(os.path.join(self.metadata_path, 'test_samples.txt'), 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

	def __len__(self):
		if self.dataset_type == 'test':
			return self.cfg.TEST.NUM_SAMPLES
		else:
			return len(self.dirs)

	def __getitem__(self, idx):
		if not (self.dataset_type == 'test'):
			print(idx, self.dirs[idx])
			filepath = os.path.join(str(self.dirs[idx]), str(self.representation_type))
			#if self.representation_type == 'surface_with_inout_fixed_views':
			if self.cfg.DATASET.FIXED_VIEWS:
				views = random.sample(range(6), self.n_views_rendering)
			else:
				views = random.sample(range(25), self.n_views_rendering)

			rendering_images = []
			for v in views:
				if self.background:
					filename = os.path.join(filepath, str(v) +'_with_bg.png')
				else:
					filename = os.path.join(filepath, str(v) +'.png')
				if not self.grayscale:
					try:
						rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
					except AttributeError:
						view = random.sample(range(25), 1)
						filename = os.path.join(filepath, str(view) +'.png')
						rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

				if self.grayscale:
					rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
				rendering_images.append(rendering_image)
			
			rendering_images = np.asarray(rendering_images)
			if self.grayscale:
				rendering_images = np.expand_dims(rendering_images, axis=-1)
			
			# if self.representation_type == 'surface_with_inout_fixed_views':
			# 	filepath = os.path.join(str(self.dirs[idx]), 'surface_with_inout')
			if self.cfg.DATASET.FIXED_VIEWS:
				volume_path = '/scratch/bbmw/jd23697/AF_swissprot_transfer_to_bridges2'
				folder_name = self.dirs[idx].split('/')[-1]
				filepath = os.path.join(volume_path, folder_name,'surface_trimesh_voxels')


			volume = np.load(os.path.join(filepath, '%s.npz'%self.cfg.DATASET.OUTPUT_RES))['arr_0']
			# volume = np.load(os.path.join(filepath, '128.npz'))['arr_0']
			
			volume = volume.astype(np.float32)

			if self.transforms:
				rendering_images = self.transforms(rendering_images)

			if self.pix2vox and not (self.cfg.NETWORK.USE_SEQ):
				return rendering_images, volume
			else:
				# Sequnece embeddings
				folder_name =  self.dirs[idx].split('/')[-1]
				seq_emd_filename = folder_name.split('.')[0] + '_esm2_t33.txt'
				seq_emd_path = os.path.join(self.seq_embd_path, folder_name, seq_emd_filename)
				seq_emd = np.loadtxt(seq_emd_path, dtype=np.float32)
				seq_emd = torch.from_numpy(seq_emd)
				return rendering_images, seq_emd, volume
			
		else:

			#filepath = str(self.dirs[idx])
			rendering_images = []
			
			# if filepath.split('/')[-1] = 'Extracted_SingleMolecule':
			if idx < 10: #5 :
				if self.background:
					# filepath = '/scratch/bbmw/jd23697/WRAC/SingleMolecule/'
					# # filepath = '/scratch/bbmw/jd23697/WRAC/SingleMolecule_V2/'
					# filepath = '/scratch/bbmw/jd23697/WRAC/SingleMolecule_high_res/'
					# filepath = '/scratch/bbmw/jd23697/WRAC/SingleMolecule_high_res_NeuralNetwork/'
					filepath = '/scratch/bbmw/jd23697/WRAC/SingleMolecule_All/'
					
					
				else:
					# filepath = '/scratch/bbmw/jd23697/WRAC/Extracted_SingleMolecule/'
					# filepath = '/scratch/bbmw/jd23697/WRAC/Extracted_SingleMolecule_V2/'
					filepath = '/scratch/bbmw/jd23697/WRAC/Extracted_SingleMolecule_All/'
					
					
				image_names = os.listdir(filepath)
				# image_names = ['1.jpg', '2.jpg', '4.jpg', '7.jpg', '20.jpg', '24.jpg', '26.jpg']
				# image_names = ['1_1.jpg', '2_1.jpg', '4_1.jpg', '4_4.jpg', '5_1.jpg', '5_5.jpg', '5_9.jpg']
				# image_names = ['1.jpg', '1_1.jpg', '3_1.jpg', '4.jpg', '4_1.jpg', '4_3.jpg', '4_4.jpg', '5_1.jpg',
				#    				'5_5.jpg', '5_9.jpg', '7.jpg', '28.jpg', '29.jpg', '34.jpg', '40.jpg']
				views = random.sample(range(len(image_names)), self.n_views_rendering)
				for v in views:
					filename = os.path.join(filepath, image_names[v])
					if not self.grayscale:
						# rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
						rendering_image = cv2.imread(filename).astype(np.float32) / 255.
						
					if self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
					rendering_image = cv2.resize(rendering_image, self.IMG_SIZE, interpolation=cv2.INTER_CUBIC)
					rendering_images.append(rendering_image)
			else:
				filepath = '/scratch/bbmw/jd23697/WRAC/surface_with_inout'
				if self.cfg.DATASET.FIXED_VIEWS:
					filepath = '/scratch/bbmw/jd23697/WRAC/surface_with_inout_fixed_views'
					views = random.sample(range(6), self.n_views_rendering)
				else:
					views = random.sample(range(25), self.n_views_rendering)
					#views = np.arange(self.n_views_rendering)
				for v in views:
					if self.background:
						filename = os.path.join(filepath, str(v) +'_with_bg.png')
					else:
						filename = os.path.join(filepath, str(v) +'.png')
					if not self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
					if self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
					rendering_images.append(rendering_image)
					
			rendering_images = np.asarray(rendering_images)
			if self.grayscale:
				rendering_images = np.expand_dims(rendering_images, axis=-1)
			

			volume = np.load(os.path.join('/scratch/bbmw/jd23697/WRAC/surface_with_inout/', '%s.npz'%self.cfg.DATASET.OUTPUT_RES))['arr_0']
			# volume = np.load(os.path.join(filepath, '128.npz'))['arr_0']
			
			volume = volume.astype(np.float32)

			if self.transforms:
				rendering_images = self.transforms(rendering_images)
			
			# return 'WRAC_Protein', 'WRAC_Protein', rendering_images, volume
			if self.pix2vox and not (self.cfg.NETWORK.USE_SEQ):
				return rendering_images, volume
			else:
				# Sequnece embeddings
				folder_name =  self.dirs[idx].split('/')[-1]
				seq_emd_filename = folder_name.split('.')[0] + '_esm2_t33.txt'
				seq_emd_path = os.path.join(self.seq_embd_path, folder_name, seq_emd_filename)
				seq_emd = np.loadtxt(seq_emd_path, dtype=np.float32)
				seq_emd = torch.from_numpy(seq_emd)
				return rendering_images, seq_emd, volume
			

class SequenceDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_type, representation_type='surface_with_inout', big_dataset=False):
		self.dataset_type = dataset_type
		self.representation_type = representation_type
		self.big_dataset = big_dataset
		if self.big_dataset:
			self.representation_type = 'surface_trimesh_voxels'
		if self.big_dataset:
			self.metadata_path = '/scratch/bbmw/jd23697/scripts_bigData'
			self.seq_embd_path = '/scratch/bbmw/jd23697/AF_swissprot_seq_embds'
		else:
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts' 
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/PDBs_seq_emds'
		if self.dataset_type == 'train':
			with open(os.path.join(self.metadata_path, 'train_samples.txt'), 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

		if self.dataset_type == 'val':
			with open(os.path.join(self.metadata_path, 'val_samples.txt'), 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

		if self.dataset_type == 'test':
			with open(os.path.join(self.metadata_path, 'test_samples.txt'), 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

	def __len__(self):
		return len(self.dirs)

	def __getitem__(self, idx):
		if not (self.dataset_type == 'test'):

			filepath = os.path.join(str(self.dirs[idx]), str(self.representation_type))
			volume = np.load(os.path.join(filepath, '32.npz'), allow_pickle=True)['arr_0']
			# volume = np.load(os.path.join(filepath, '128.npz'))['arr_0']
		
			volume = volume.astype(np.float32)

			# Sequnece embeddings
			folder_name =  self.dirs[idx].split('/')[-1]
			seq_emd_filename = folder_name.split('.')[0] + '_esm2_t33.txt'
			seq_emd_path = os.path.join(self.seq_embd_path, folder_name, seq_emd_filename)
			seq_emd = np.loadtxt(seq_emd_path, dtype=np.float32)
			seq_emd = torch.from_numpy(seq_emd)
			return seq_emd, volume


class ProteinAutoEncoderDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_type, transforms=None, background=False):
		self.dataset_type = dataset_type
		self.transforms = transforms
		self.background = background
		if self.dataset_type == 'train':
			if self.background:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/train_samples_bg_AutoEncoder.txt'
			else:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/train_samples_AutoEncoder.txt'

			with open(filepath, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]


		if self.dataset_type == 'val':
			if self.background:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/val_samples_bg_AutoEncoder.txt'
			else:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/val_samples_AutoEncoder.txt'
			with open(filepath, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

		if self.dataset_type == 'test':
			if self.background:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/test_samples_bg_AutoEncoder.txt'
			else:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts/test_samples_AutoEncoder.txt'
			with open(filepath, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]
				# random.shuffle(self.dirs)

	def __len__(self):
		return len(self.dirs)
	
	def __getitem__(self, idx):
		
		filename = self.dirs[idx]
		image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
		image = np.asarray(image)
		
		if self.transforms:
			image = self.transforms(image)

		return image

		
