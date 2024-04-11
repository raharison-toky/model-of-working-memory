import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
torch.manual_seed(0)
# @torch.jit.script
def get_prototype_indices(p:int,
                          n_per_prot:int,
                          n_tot:int):
    
    """
    Function for randomly assigning neurons to p prototypes (input types)

    Args:
        p: number of prototypes
        n_per_prot: number of neurons per prototypes
        n_tot: total number of neurons

    Returns:
        p x n_per_prot tensor with the indices of the neurons assigned to each prototypew
    """

    indices = []

    for i in range(p):
        indices.append(torch.randperm(n_tot)[0:n_per_prot])

    return torch.stack(indices)

# @torch.jit.script
def byte_word(prototype:torch.Tensor,
              n:int):
    
    """
    Function for obtaining the byte word (one hot encoding) of all neurons w.r.t. each prototype

    Args:
        prototype: prototype indices tensor (from get_prototype_indices)
        n: number of neurons
    Returns:
        n x number of prototype one-hot tensor
    """

    p = len(prototype)
    one_hot_tensor = torch.zeros(n, p)
    for i, vector in enumerate(prototype):
        indices = torch.tensor(vector)
        one_hot_tensor[indices, i] = 1
    return one_hot_tensor

# @torch.jit.script
def noisy_prototype(byte_coding:torch.Tensor,
                    p:int,
                    x_noise:float):
     
    """
    Obtains noisy prototype vectors (randomly reassigns neurons)
    Neurons assigned to prototype p will have a higher probability of being selective to p.

    Args:
        byte_coding: one-hot tensor of neurons w.r.t. all prototypes
        p: target prototype
        x_noixe: noisyness of reassignment
    Returns:
        one-hot vector for prototype p
    """

    indices = torch.zeros_like(byte_coding[:,0])
    rand_indices = torch.rand_like(indices)
    indices[(byte_coding[:,p]==1) & (rand_indices>=x_noise)] = 1
    indices[(byte_coding[:,p]==0) & (rand_indices>(1-x_noise))] = 1
    return indices

# @torch.jit.script
def stimulus(prototype:torch.Tensor,
             num_excitatory:int,
             mean_excitatory:float,
             mean_inhibitory:float,
             std_excitatory:float,
             std_inhibitory:float,
             contrast_excitatory:float,
             contrast_inhibitory:float,
             t_stim:int):
     
    """
    Generates a chunk of stimulus for a target prototype

    Args:
        prototype: (noisy) one-hot prototype vector
        num_excitatory: number of excitatory neurons
        mean_excitatory: mean current to excitatory neurons
        mean_inhibitory: mean current to inhibitory neurons
        std_excitatory: standard deviation of current to excitatory neurons
        std_inhibitory: standard deviation of current to inhibitory neurons
        contrast_excitatory: factor to multiply mean current for assigned excitatory neurons
        contrast_inhibitory: factor to multiply mena current for assigned inhibitory neurons
        t_sim: duration of stimuli

    Returns:
        t_sim x total number of neurons tensor of external current over time
    """
    
    means = torch.zeros_like(prototype)
    means[0:num_excitatory][prototype[0:num_excitatory]==1] = mean_excitatory*contrast_excitatory
    means[0:num_excitatory][prototype[0:num_excitatory]==0] = mean_excitatory
    means[num_excitatory:][prototype[num_excitatory:]==1] = mean_inhibitory*contrast_inhibitory
    means[num_excitatory:][prototype[num_excitatory:]==0] = mean_inhibitory

    std = std_excitatory*torch.ones_like(prototype)
    std[num_excitatory:] = std_inhibitory

    means = torch.tile(means,(t_stim//32,1))
    std = torch.tile(std,(t_stim//32,1))
    signal = torch.normal(means,std)
    signal = rearrange(signal,"t n -> n 1 t")
    signal = F.interpolate(signal,[t_stim])
    signal = rearrange(signal,"n 1 t -> t n")
    return signal

# @torch.jit.script
def generate_trials(byte_coding:torch.Tensor,
                   x_noise:float,
                   num_excitatory:int,
                   mean_excitatory:float,
                   mean_inhibitory:float,
                   std_excitatory:float,
                   std_inhibitory:float,
                   contrast_excitatory:float,
                   contrast_inhibitory:float,
                   t_stim:int,
                   t_rest:int,
                   num_repetitions:int,
                   initial_rest:bool=True):
    
    """
    Creates the total external currents for a full experiment

    Args:
        byte_coding: one-hot tensor of all neurons w.r.t. all prototypes
        num_excitatory: number of excitatory neurons
        mean_excitatory: mean current to excitatory neurons
        mean_inhibitory: mean current to inhibitory neurons
        std_excitatory: standard deviation of current to excitatory neurons
        std_inhibitory: standard deviation of current to inhibitory neurons
        contrast_excitatory: factor to multiply mean current for assigned excitatory neurons
        contrast_inhibitory: factor to multiply mena current for assigned inhibitory neurons
        t_sim: duration of stimuli
        t_rest: duration of rest in between stimuli
        num_repetitions: number of chunks of stimuli presentation
    Returns:
        tuple with external currents and event markers for stimuli presentation
    """

    if initial_rest:
        i_ext = [stimulus(prototype=torch.zeros_like(byte_coding[:,0]),num_excitatory=num_excitatory,
                          mean_excitatory=mean_excitatory,mean_inhibitory=mean_inhibitory,
                          std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=1.0,contrast_inhibitory=1.0,t_stim=t_rest)]
        events = [torch.zeros(t_rest)]
    else:
        i_ext = []
        events = []
    num_prototypes = byte_coding.shape[-1]
    

    for _ in range(num_repetitions):
        order = torch.randperm(num_prototypes)
        for stim in order:
            noisy_prot = noisy_prototype(byte_coding=byte_coding,p=stim,x_noise=x_noise)
            stim_arr = stimulus(prototype=noisy_prot,num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=contrast_excitatory,contrast_inhibitory=contrast_inhibitory,t_stim=t_stim)
            
            rest = stimulus(prototype=noisy_prot,num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=1.0,contrast_inhibitory=1.0,t_stim=t_rest)
            # print(stim_arr.shape)
            # print(rest.shape)
            i_ext.append(torch.concat((stim_arr,rest),0))
            events.append(torch.concat(((stim+1)*torch.ones(t_stim),torch.zeros(t_rest)),0))
    return torch.concat(i_ext,0),torch.concat(events,0)

def manual_trials(trials:torch.Tensor,
                   byte_coding:torch.Tensor,
                   x_noise:float,
                   num_excitatory:int,
                   mean_excitatory:float,
                   mean_inhibitory:float,
                   std_excitatory:float,
                   std_inhibitory:float,
                   contrast_excitatory:float,
                   contrast_inhibitory:float):
    
    """
    Creates the total external currents for a full experiment

    Args:
        byte_coding: one-hot tensor of all neurons w.r.t. all prototypes
        num_excitatory: number of excitatory neurons
        mean_excitatory: mean current to excitatory neurons
        mean_inhibitory: mean current to inhibitory neurons
        std_excitatory: standard deviation of current to excitatory neurons
        std_inhibitory: standard deviation of current to inhibitory neurons
        contrast_excitatory: factor to multiply mean current for assigned excitatory neurons
        contrast_inhibitory: factor to multiply mena current for assigned inhibitory neurons
        t_sim: duration of stimuli
        t_rest: duration of rest in between stimuli
        num_repetitions: number of chunks of stimuli presentation
    Returns:
        tuple with external currents and event markers for stimuli presentation
    """

    i_ext = []
    events = []
    
    for i in trials:
        if i[0] == 0:
            stim_arr = stimulus(prototype=torch.zeros_like(byte_coding[:,0]),num_excitatory=num_excitatory,
                          mean_excitatory=mean_excitatory,mean_inhibitory=mean_inhibitory,
                          std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=1.0,contrast_inhibitory=1.0,t_stim=i[1])
        else:
            noisy_prot = noisy_prototype(byte_coding=byte_coding,p=i[0]-1,x_noise=x_noise)
            stim_arr = stimulus(prototype=noisy_prot,num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=contrast_excitatory,contrast_inhibitory=contrast_inhibitory,t_stim=i[1])
        events_i = i[0]*torch.ones(i[1])
        i_ext.append(stim_arr)
        events.append(events_i)

    return torch.concat(i_ext,0),torch.concat(events,0)

def associativity_trials(pairs,
                         t_stim,
                         t_rest,
                         num_repetitions,
                         byte_coding:torch.Tensor,
                         x_noise:float,
                         num_excitatory:int,
                         mean_excitatory:float,
                         mean_inhibitory:float,
                         std_excitatory:float,
                         std_inhibitory:float,
                         contrast_excitatory:float,
                         contrast_inhibitory:float):
    
    events_order = []
    
    for _ in range(num_repetitions):
        order = torch.randperm(len(pairs))
        for j in order:
            events_order.append([0,t_rest])
            events_order.append([pairs[j][0],t_stim])
            events_order.append([pairs[j][1],t_stim])

    print(events_order)

    return manual_trials(events_order,
                         byte_coding=byte_coding,
                         x_noise=x_noise,
                         num_excitatory=num_excitatory,
                         mean_excitatory=mean_excitatory,
                         mean_inhibitory=mean_inhibitory,
                         std_excitatory=std_excitatory,
                         std_inhibitory=std_inhibitory,
                         contrast_excitatory=contrast_excitatory,
                         contrast_inhibitory=contrast_inhibitory)

def random_associativity(stim_neurons,
                         reward_neurons):
    
    pre = torch.randperm(len(stim_neurons))
    post = torch.randperm(len(reward_neurons))

    return [[stim_neurons[i],reward_neurons[j]] for i,j in zip(pre,post)]