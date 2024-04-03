import torch
from einops import rearrange, repeat
from typing import Tuple

@torch.jit.script
def dx_dt(x:torch.Tensor,
		  u:float,
		  tau_r:float,
		  x_sigma_v:torch.Tensor,
		  dt:float)->torch.Tensor:

	"""
	Function for obtaining the update using forward Euler for synaptic resource dynamics

	Args:
		x: resources at time t
		u: cost of spiking
		tau_r: recovery time
		x_sigma_v: boolean vector for spiking neurons
		dt: time step for forward Euler

	Returns:
		update for slow currents at time t+1
	"""
	return dt*((1-x)/tau_r)-u*x_sigma_v

@torch.jit.script
def r_t(l:torch.Tensor,
		alpha:float,
		beta:float,
		theta_x:float)->torch.Tensor:

	"""
	Function for computing the recovery term for long-term plasticity dynamics

	Args:
		l: plasticity latent variable vector at time t
		alpha: upward recovery factor
		beta: downward recovery factor
		theta_x: plasticity threshold

	Returns:
		vector of recovery term at time t
	"""
	return -alpha*((theta_x-l)>=0) + beta*((l-theta_x)>=0)

@torch.jit.script
def f_v(v:torch.Tensor,
		a:float,
		b:float,
		theta_ltp:float,
		theta_ltd:float,
		v_th:torch.Tensor,
		v_reset:torch.Tensor)->torch.Tensor:
	"""
	Function for verifying if post and pre-synaptic neurons are firing together.

	Args:
		v: membrane potential at time t
		a: step up value for synchronous firing
		b: step down value for asynchronous firing
		theta_ltp: threshold potential for long-term potentiation
		theta_ltd: threshold potential for long-term depression
		v_th: threshold potential for action potential firing
		v_reset: reset potential

	Returns:
		vector for tracking synchronous firing
	"""
	return a*(theta_ltp<=v)*(v<=v_th) - b*(v<=theta_ltd)*(v<=v_reset)

@torch.jit.script
def dl_dt(h_t:torch.Tensor,
		  w:torch.Tensor,
		  r_t:torch.Tensor,
		  dt:float):
	
	"""
	Function for the long-term plasticity latent variable update with forward Euler

	Args:
		h_t: Hebbian term matrix
		w: masking matrix for excitatory-excitatory synapses
		r_t: recovery term matrix
		dt: time step for forward Euler

	Returns:
		n x n latent variable for plasticity
	"""
	return dt*w*(r_t+h_t)

@torch.jit.script
def update_j(j:torch.Tensor,
			 j_p:float,
			 j_d:float,
			 threshold:float,
			 l_old:torch.Tensor,
			 l_new:torch.Tensor):

	"""
	Function for updating conductivity matrices based on plasticity latent variable

	Args:
		j: conductivity matrix
		j_p: potentiated conductance
		j_d: depressed conductance
		threshold: latent variable threshold for long-term plasticity
		l_old: latent variable value at time t
		l_new: updated latent variable value

	Returns:
		updated conductivity matrix
	"""
	a = (l_old<=threshold)*(threshold<l_new)
	b = (l_old>=threshold)*(threshold>l_new)
	return j_p*a + j_d*b + j*(1-(a+b).int())