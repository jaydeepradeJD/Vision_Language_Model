import os
import cv2
import numpy as np
import random
import torch.utils.data.dataset
import torch
from transform_matrix import transform_matrix


class ProteinDataset(torch.utils.data.Dataset):
	def __init__(self, cfg, dataset_type, n_views_rendering, representation_type='surface_with_inout', transforms=None, grayscale=False, background=False, big_dataset=False):
		self.cfg = cfg
		self.dataset_type = dataset_type
		self.n_views_rendering = n_views_rendering
		self.representation_type = representation_type
		self.transforms = transforms
		self.grayscale = grayscale
		self.background = background
		self.big_dataset = big_dataset
		
		if self.big_dataset:
			self.representation_type = 'surface_trimesh_voxels'
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/AF_swissprot_seq_embds'
			
		else:
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts'
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/PDBs_seq_emds'

		
		if self.dataset_type == 'train':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples.txt')
			else:
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			with open(train_samples_filename, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]
				
		
		if self.dataset_type == 'val':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples.txt')
			else:
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			with open(val_samples_filename, 'r') as f:
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
			if self.representation_type == 'surface_with_inout_fixed_views':
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
					rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
				if self.grayscale:
					rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
				rendering_images.append(rendering_image)
			
			rendering_images = np.asarray(rendering_images)
			if self.grayscale:
				rendering_images = np.expand_dims(rendering_images, axis=-1)
			
			if self.representation_type == 'surface_with_inout_fixed_views':
				filepath = os.path.join(str(self.dirs[idx]), 'surface_with_inout')

			volume = np.load(os.path.join(filepath, '32.npz'))['arr_0']
			# volume = np.load(os.path.join(filepath, '128.npz'))['arr_0']
			
			volume = volume.astype(np.float32)

			if self.transforms:
				rendering_images = self.transforms(rendering_images)

			# Sequnece embeddings
			folder_name =  self.dirs[idx].split('/')[-1]
			seq_emd_filename = folder_name.split('.')[0] + '_esm2_t33.txt'
			seq_emd_path = os.path.join(self.seq_embd_path, folder_name, seq_emd_filename)
			seq_emd = np.loadtxt(seq_emd_path, dtype=np.float32)
			seq_emd = torch.from_numpy(seq_emd)
			return rendering_images, seq_emd, volume

		else:

			filepath = str(self.dirs[idx])
			rendering_images = []
			
			# if filepath.split('/')[-1] = 'Extracted_SingleMolecule':
			if idx < 5 :
				if self.background:
					filepath = '/work/mech-ai-scratch/jrrade/Protein/WRAC/Extracted_SingleMolecule_with_bg/'
				else:
					filepath = '/work/mech-ai-scratch/jrrade/Protein/WRAC/Extracted_SingleMolecule/'
				image_names = os.listdir(filepath)
				views = random.sample(range(22), self.n_views_rendering)
				for v in views:
					filename = os.path.join(filepath, image_names[v])
					if not self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
					if self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
					rendering_images.append(rendering_image)
			else:
				filepath = '/work/mech-ai-scratch/jrrade/Protein/WRAC/%s'%self.representation_type
				if self.representation_type == 'surface_with_inout_fixed_views':
					views = random.sample(range(6), self.n_views_rendering)
				else:
					views = random.sample(range(25), self.n_views_rendering)
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
			

			volume = np.load(os.path.join('/work/mech-ai-scratch/jrrade/Protein/WRAC/%s/'%self.representation_type, '32.npz'))['arr_0']
			# volume = np.load(os.path.join(filepath, '128.npz'))['arr_0']
			
			volume = volume.astype(np.float32)

			if self.transforms:
				rendering_images = self.transforms(rendering_images)
			
			return 'WRAC_Protein', 'WRAC_Protein', rendering_images, volume


class SequenceDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_type, representation_type='surface_with_inout', big_dataset=False):
		self.dataset_type = dataset_type
		self.representation_type = representation_type
		self.big_dataset = big_dataset
		if self.big_dataset:
			self.representation_type = 'surface_trimesh_voxels'
		if self.big_dataset:
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/AF_swissprot_seq_embds'
		else:
			self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/scripts' 
			self.seq_embd_path = '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/PDBs_seq_emds'
		if self.dataset_type == 'train':
			# with open(os.path.join(self.metadata_path, 'train_samples.txt'), 'r') as f:
			with open(os.path.join(self.metadata_path, 'train_samples_256.txt'), 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

		if self.dataset_type == 'val':
			# with open(os.path.join(self.metadata_path, 'val_samples.txt'), 'r') as f:
			with open(os.path.join(self.metadata_path, 'val_samples_256.txt'), 'r') as f:
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

		
class ProteinTransformDataset(torch.utils.data.Dataset):
	def __init__(self, cfg, dataset_type, n_views_rendering, representation_type='surface_trimesh_voxels', transforms=None, grayscale=False, background=False):
		self.representation_type = representation_type
		self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
		self.cfg = cfg
		self.dataset_type = dataset_type
		self.n_views_rendering = n_views_rendering
		self.transforms = transforms
		self.grayscale = grayscale
		self.background = background
		self.transform_matrices = transform_matrix()
		self.pre_array = []
		self.after_array = []


		if self.dataset_type == 'train':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples.txt')
			else:
				train_samples_filename = os.path.join(self.metadata_path, 'train_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			with open(train_samples_filename, 'r') as f:
				dir_list = f.readlines()
				self.dirs = [d.strip() for d in dir_list]

		if self.dataset_type == 'val':
			if self.cfg.DATASET.NUM_SAMPLES == 'whole_data':
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples.txt')
			else:
				val_samples_filename = os.path.join(self.metadata_path, 'val_samples_%s.txt'%self.cfg.DATASET.NUM_SAMPLES)
			with open(val_samples_filename, 'r') as f:
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
			#double the views for pairing
			views = random.sample(range(25), self.n_views_rendering * 2)
			matrices = self.transform_matrices.get_transforms(views, filepath)
			rendering_images = [[],[]]

			for i in range(2):
				for v in views[i::2]:
					if self.background:
						filename = os.path.join(filepath, str(v) +'_with_bg.png')
					else:
						filename = os.path.join(filepath, str(v) +'.png')
					if not self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

					if self.grayscale:
						rendering_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
					rendering_images[i].append(rendering_image)
				self.pre_array = rendering_images[i]
				rendering_images[i] = np.asarray(rendering_images[i])
				if self.grayscale:
					rendering_images[i] = np.expand_dims(rendering_images[i], axis=-1) 
				self.after_array = rendering_images[i]
			
			return rendering_images[0], rendering_images[1], matrices
		