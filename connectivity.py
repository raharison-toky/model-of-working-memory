import torch
from einops import rearrange, repeat
from typing import Tuple

def random_sparse(m:int,
				  n:int,
				  p:float)->torch.Tensor:
	"""
	Function for creating a sparse m x n with zeros and ones with some probability.

	Args:
		m: number of post-synaptic neurons
		n: number of pre-synaptic neurons
		p: probability of connection
	
	Returns:
		mat: m x n sparse matrix
	"""
	mat = torch.zeros(m,n)
	mat[torch.rand(m,n)<p] = 1
	return mat

def sparse_square(n:int,
				  p:float):
	"""
	Function for creating a n x n sparse matrix with zeros on the diagonal

	Args:
		n: number of neurons
		p: probability of connection

	Returns:
		mat: n x n sparse matrix
	"""
	mat = torch.zeros(n,n)
	mat[torch.rand(n,n)<p] = 1
	return mat*(1-torch.eye(n))

def slow_fast_pair(m:int,
				   n:int,
				   p_connection:float,
				   ratio_slow:float,)->torch.Tensor:

	"""
	Function for obtaining pairs of connectivity matrices for slow/fast currents.
	The matrices will not overlap so connections can either be slow or fast.

	Args:
		m: number of post-synaptic neurons
		n: number of pre-synaptic neurons
		p_connection: probability of connection:
		ratio_slow: ratio of slow connection over all connections

	Returns:
		slow: sparse matrix for slow currents
		fast: sparse matrix for fast currents
	"""

	p_slow = p_connection*ratio_slow
	p_fast = p_connection*(1-ratio_slow)

	if m==n:
		slow = sparse_square(m,p_slow)
		fast = sparse_square(m,p_fast)
		fast = (1-slow)*fast
		return slow,fast

	else:
		slow = random_sparse(m,n,p_slow)
		fast = random_sparse(m,n,p_fast)
		fast = (1-slow)*fast
		return slow,fast
	
def initialize_j(u_mat:torch.Tensor,
				 p_pot:float,
				 j_p:float,
				 j_d:float)->Tuple[torch.Tensor]:

	"""
	Function for filling connectivity matrix with one of two possible conductance value.
	Also initializes the hidden synaptic variable L

	Args:
		u_mat: connectivity matrix
		p_pot: probability for potentiated conductance
		j_p: conductivity for potentiated synapse
		j_d: conductivity for depressed synapse

	Returns:
		conductance matrix
	"""
	j = j_d*torch.ones_like(u_mat)
	initalization = torch.rand_like(j)
	x_init = torch.zeros_like(u_mat)
	x_init[initalization<=p_pot] = 1
	j[initalization<=p_pot] = j_p
	return (j*u_mat,x_init*u_mat)