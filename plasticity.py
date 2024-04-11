import torch
from einops import rearrange, repeat
from typing import Tuple
torch.manual_seed(0)

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
    return x + dt*(((1-x)/tau_r)-u*x_sigma_v/dt)

@torch.jit.script
def r_t(l:torch.Tensor,
        alpha:float,
        beta:float,
        theta_x:float)->torch.Tensor:

    """
    Function for computing the refresh term for long-term plasticity dynamics

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
        theta_ltd:float,)->torch.Tensor:
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
    return a*(v>=theta_ltp) - b*(v<theta_ltd)
    
@torch.jit.script
def dl_dt(h_t:torch.Tensor,
          w:torch.Tensor,
          r_t:torch.Tensor,
          l:torch.Tensor,
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
    return l+dt*w*(r_t+(h_t/dt))

@torch.jit.script
def update_j(j:torch.Tensor,
             j_p:float,
             j_d:float,
             threshold:float,
             l_old:torch.Tensor,
             l_new:torch.Tensor,
             u:torch.Tensor):

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
    return u*(j_p*a + j_d*b + j*(1-(a+b).int()))

@torch.jit.script
def h_mat(f_v_t:torch.Tensor,
          sigma_v:torch.Tensor):
    
    return (f_v_t.unsqueeze(-1) @ sigma_v.unsqueeze(0)).T

@torch.jit.script
def get_mat(potentials:torch.Tensor, 
            v_th:torch.Tensor, 
            v_ltp:float, 
            v_ltd:float, 
            a:float, 
            b:float):
    # Create a meshgrid of indices for potentials
    i, j = torch.meshgrid(potentials, potentials)
    
    # Create masks for conditions
    mask_th = (j >= v_th).float()
    mask_ltp = (i >= v_ltp).float()
    mask_ltd = (i < v_ltd).float()
    
    # Calculate mat using vectorized operations
    mat = (mask_th * (a * mask_ltp - b * mask_ltd))
    
    return mat

@torch.jit.script
def simple_stdp(spike_chrono:torch.Tensor,
                v:torch.Tensor,
                v_th:torch.Tensor,
                a:float,
                b:float):
    
    i,j = torch.meshgrid(spike_chrono,spike_chrono)
    spiking = (j-i)
    plus_mask = ((0 <= spiking) & (spiking <= 5)).float()
    minus_mask = ((0 > spiking) & (-spiking <= 75)).float()
    return (a*plus_mask - b*minus_mask)*(v>=v_th)

@torch.jit.script
def simplified_stdp(spike_chrono:torch.Tensor,
                v:torch.Tensor,
                v_th:torch.Tensor,
                a:float,
                b:float):
    
    i,j = torch.meshgrid(spike_chrono,spike_chrono)
    spiking = (j-i)
    if any(torch.isnan(spiking)):
        raise ValueError("spiking is problematic")
    plus_mask = torch.clamp(torch.exp(-spiking/15.2)-torch.exp(-spiking/2.0),0,None)
    minus_mask = torch.clamp(torch.exp(spiking/2)-torch.exp(spiking/33.2),None,0)
    result = (a*plus_mask + b*minus_mask)*(v>=v_th)
    if any(torch.isnan(result)):
        print("plus")
        raise ValueError("NaN")
    return result

@torch.jit.script
def continuous_connectivity(j:torch.Tensor,
                            j_p:float,
                            j_d:float,
                            l:torch.Tensor,
                            w_mat:torch.Tensor):
    
    return w_mat*torch.clamp(l*(j_p-j_d)+j_d,j_d,j_p) + torch.logical_not(w_mat)*j