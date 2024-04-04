import torch

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

    means = torch.tile(means,(t_stim,1))
    std = torch.tile(std,(t_stim,1))
    signal = torch.normal(means,std)
    return signal

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
                   num_repetitions:int):
    
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
    i_ext = [stimulus(prototype=torch.zeros_like(byte_coding[:,0]),num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=1,contrast_inhibitory=1,t_stim=t_rest)]
    num_prototypes = byte_coding.shape[-1]
    events = [torch.zeros(t_rest)]

    for i in range(num_repetitions):
        order = torch.randperm(num_prototypes)
        for stim in order:
            noisy_prot = noisy_prototype(byte_coding=byte_coding,p=stim,x_noise=x_noise)
            stim_arr = stimulus(prototype=noisy_prot,num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=contrast_excitatory,contrast_inhibitory=contrast_inhibitory,t_stim=t_stim)
            
            rest = stimulus(prototype=noisy_prot,num_excitatory=num_excitatory,mean_excitatory=mean_excitatory,
                   mean_inhibitory=mean_inhibitory,std_excitatory=std_excitatory,std_inhibitory=std_inhibitory,
                   contrast_excitatory=1,contrast_inhibitory=1,t_stim=t_rest)
            print(stim_arr.shape)
            print(rest.shape)
            i_ext.append(torch.concat((stim_arr,rest),0))
            events.append(torch.concat(((stim+1)*torch.ones(t_stim),torch.zeros(t_rest)),0))
    return torch.concat(i_ext,0),torch.concat(events,0)