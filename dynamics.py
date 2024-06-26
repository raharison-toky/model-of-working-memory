import torch
from einops import rearrange, repeat
from typing import Tuple
torch.manual_seed(0)

@torch.jit.script
def sigma(v_t:torch.Tensor,
          v_th:torch.Tensor,
          dtype:torch.dtype)->torch.Tensor:
    """
    Function for verifying which neurons are spiking

    Args:
        v_t: membrane potentials at time t
        v_th: vector of threshold potentials (may not be uniform)

    Returns:
        tensor of 0 (not spiking), 1 (spiking) 
    """
    return (v_t>=v_th).to(dtype)

@torch.jit.script
def dis_dt(i_s:torch.Tensor,
           tau_s:torch.Tensor,
           dt:float,
           j_slow:torch.Tensor,
           x_sigma_v:torch.Tensor)->torch.Tensor:

    """
    Function for obtaining the update using forward Euler for slow current dynamics
    tau_s di/dt = - i_s + j sigma(v,v_th)

    Args:
        i_s: slow currents at time t
        tau_s: tensor of time constants for slow currents
        dt: time step for forward Euler
        u_slow: synaptic conductance matrix for slow currents
        x_sigma_v: boolean vector for spiking neurons

    Returns:
        update for slow currents at time t+1
    """
    return i_s + (dt/tau_s)*(-i_s + (1/dt)*j_slow @ x_sigma_v)

@torch.jit.script
def da_dt(v:torch.Tensor,
          a:torch.Tensor,
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
    da = (1/tau_r)*dt*torch.logical_or(v>=v_th,a>0)
    return torch.zeros_like(a) + (a+da)*(a<1)

@torch.jit.script
def ds_dt(v:torch.Tensor,
          s:torch.Tensor,
          v_th:torch.Tensor,
          dt:float):
    
    """
    Function updating the counter for time since last spike

    Args:
        v: membrane potential at time t
        s: time counter
        v_th: vector of membrane threshold potentials
        dt: time step for forward Euler

    Returns:
        vector of spiking hidden variable  update
    """

    return torch.zeros_like(s)*(v>=v_th) + (s+dt)*(v<v_th)

@torch.jit.script
def dv_dt(v:torch.Tensor,
          a:torch.Tensor,
          i:torch.Tensor,
          v_reset:torch.Tensor,
          tau_m:torch.Tensor,
          v_th:torch.Tensor,
          dt:float):

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
    dv = (-v/tau_m+i/tau_m)*dt
    return v_reset*torch.logical_or(v>=v_th,a>0) + (v+dv)*torch.logical_and(v<v_th,a==0)