import torch
from einops import rearrange, repeat
from typing import Tuple

def sigma(v_t:torch.Tensor,
		  v_th:torch.Tensor)->torch.Tensor:
	"""
	Function for verifying which neurons are spiking

	Args:
		v_t: membrane potentials at time t
		v_th: vector of threshold potentials (may not be uniform)

	Returns:
		tensor of 0 (not spiking), 1 (spiking) 
	"""
	return (v_t>=v_th).float()

def dis_dt(i_s:torch.Tensor,
		   tau_s:float,
		   dt:float,
		   j_slow:torch.Tensor,
		   x_sigma_v:torch.Tensor)->torch.Tensor:

	"""
	Function for obtaining the update using forward Euler for slow current dynamics
	tau_s di/dt = - i_s + j sigma(v,v_th)

	Args:
		i_s: slow currents at time t
		tau_s: time constant for slow currents
		dt: time step for forward Euler
		u_slow: synaptic conductance matrix for slow currents
		x_sigma_v: boolean vector for spiking neurons

	Returns:
		update for slow currents at time t+1
	"""
	return (dt/tau_s)*(-i_s + j_slow @ x_sigma_v)

def da_dt(v:torch.Tensor,
		  tau_r:torch.Tensor,
		  v_th:torch.Tensor,
		  dt:float):
	
	"""
	Function for the spiking hidden variable update using forward Euler

	Args:
		v: membrane potential at time t
		tau_r: vector of membrane refractory time constant
		v_th: vector of membrane threshold potentials
		dt: time step for forward Euler

	Returns:
		vector of spiking hidden variable  update
	"""
	return (1/tau_r)*dt*(v>=v_th)

def dv_dt(v,i,tau_m,v_th,dt):

	"""
	Function for the membrane potential update using forward Euler

	Args:
		v: membrane potential at time t
		i: total currents at time t
		tau_m: vector of membrane 
		v_th: vector of membrane threshold potentials
		dt: time step for forward Euler

	Returns:
		vector of spiking hidden variable update
	"""
	return (-v/tau_m+i)*dt*(v<=v_th)