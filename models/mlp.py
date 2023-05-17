import torch

class MLP(torch.nn.Module):
	def __init__(self, cfg):
		super(MLP, self).__init__()


		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(1280, 1280),
			torch.nn.BatchNorm1d(1280),
			torch.nn.ReLU()
		)

	def forward(self, seq_emd):
		return self.mlp(seq_emd)