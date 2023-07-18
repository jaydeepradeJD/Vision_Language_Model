import math
import torch
import torch.nn.functional as F
from torch import nn

# Code reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

def scaled_dot_product(q, k, v, mask=None):
	d_k = q.size()[-1]
	attn_logits = torch.matmul(q, k.transpose(-2, -1))
	attn_logits = attn_logits / math.sqrt(d_k)
	if mask is not None:
		attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
	attention = F.softmax(attn_logits, dim=-1)
	values = torch.matmul(attention, v)
	return values, attention

class MultiheadAttention(nn.Module):

	def __init__(self, embed_dim, num_heads):
		super().__init__()
		assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
				
		# Stack all weight matrices 1...h together for efficiency
		# Note that in many implementations you see "bias=False" which is optional
		self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
		self.o_proj = nn.Linear(embed_dim, embed_dim)

		self._reset_parameters()

	def _reset_parameters(self):
		# Original Transformer initialization, see PyTorch documentation
		nn.init.xavier_uniform_(self.qkv_proj.weight)
		self.qkv_proj.bias.data.fill_(0)
		nn.init.xavier_uniform_(self.o_proj.weight)
		self.o_proj.bias.data.fill_(0)

	def forward(self, x, return_attention=False):
		batch_size, seq_length, _ = x.size()
		qkv = self.qkv_proj(x)

		# Separate Q, K, V from linear output
		qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
		qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
		q, k, v = qkv.chunk(3, dim=-1)

		# Determine value outputs
		values, attention = scaled_dot_product(q, k, v)
		values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
		values = values.reshape(batch_size, seq_length, self.embed_dim)
		o = self.o_proj(values)

		if return_attention:
			return o, attention
		else:
			return o
		
# Code reference: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class FeedFoward(nn.Module):
	""" a simple linear layer followed by a non-linearity """

	def __init__(self, embd_dim, dropout=0.2):
		super().__init__()
		self.embd_dim = embd_dim
		self.net = nn.Sequential(
			nn.Linear(self.embd_dim, 4 * self.embd_dim),
			nn.ReLU(),
			nn.Linear(4 * self.embd_dim, self.embd_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)

class TransformerBlock(nn.Module):
	""" Transformer block: communication followed by computation """

	def __init__(self, embd_dim, n_head):
		# embd_dim: embedding dimension, n_head: the number of heads we'd like
		super().__init__()
		self.embd_dim = embd_dim
		
		self.sa = MultiheadAttention(self.embd_dim, n_head) # def __init__(self, input_dim, embed_dim, num_heads):
		self.ffwd = FeedFoward(self.embd_dim)
		self.ln1 = nn.LayerNorm(self.embd_dim)
		self.ln2 = nn.LayerNorm(self.embd_dim)

	def forward(self, x):
		# x = x + self.sa(self.ln1(x))
		# x = x + self.ffwd(self.ln2(x))
		x = self.ln1(x + self.sa(x))
		x = self.ln2(x + self.ffwd(x))
		return x