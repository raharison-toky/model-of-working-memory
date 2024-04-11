import torch
from connectivity import*
from dynamics import*
from plasticity import*
from stimuli import*
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict
import argparse
from einops import reduce
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")
    DTYPE = torch.float16
else:
    torch.set_default_dtype(torch.float32)
    DTYPE = torch.float32

STORAGE_DEVICE = "cuda"

DEBUG = True
MODE = "stdp"
# should probably use something like yaml but good enough for now

from constants import*

parser = argparse.ArgumentParser()

parser.add_argument("-a","--associativity",help="associativity experiment",action=argparse.BooleanOptionalAction,default=False)
parser.add_argument("-n","--nmda",help="reduce NMDA conductance",action=argparse.BooleanOptionalAction,default=False)
parser.add_argument("-g","--gaba",help="reduce GABA conductance",action=argparse.BooleanOptionalAction,default=False)
parser.add_argument("-f","--name",help="name of save folder/experiment",type=str)
parser.add_argument("-e","--epochs",help="number of epochs (repetitions)",type=int)
parser.add_argument("-x","--noise",help="prototype noise",type=float)

args = parser.parse_args()

if args.nmda:
    print("Reducing NMDA currents")
    NMDA_SCALING = 0.5
else:
    print("Normal NMDA currents")
    NMDA_SCALING = 1

if args.gaba:
    print("Reducing GABA currents")
    GABA_SCALING = 1.5
else:
    print("Normal GABA currents")
    GABA_SCALING = 1

if args.name is None:
    EXPERIMENT_NAME = "debug"
else:
    EXPERIMENT_NAME = args.name

if args.associativity:
    print("Running associativity experiment")
else:
    print("Running normal experiment")

if args.noise is not None:
    X_NOISE = args.noise

if args.epochs is not None:
    NUM_REPETITIONS = args.epochs
ASSOCIATIVITY_EXPERIMENTS = args.associativity

def get_connectivity_matrices(fast_inhibitory=1,
                              fast_excitatory=1,
                              slow_inhibitory=1,
                              slow_excitatory=1):

    U_e_to_e_slow,U_e_to_e_fast = slow_fast_pair(NUM_EXCITATORY,NUM_EXCITATORY,P_CONNECTION,RATIO_SLOW_EXCITATORY_EXCITATORY)
    U_i_to_e_slow,U_i_to_e_fast = slow_fast_pair(NUM_EXCITATORY,NUM_INHIBITORY,P_CONNECTION,RATIO_SLOW_EXCITATORY_INHIBITORY)
    U_e_to_i_slow,U_e_to_i_fast = slow_fast_pair(NUM_INHIBITORY,NUM_EXCITATORY,P_CONNECTION,RATIO_SLOW_INHIBITORY_EXCITATORY)
    U_i_to_i_slow,U_i_to_i_fast = slow_fast_pair(NUM_INHIBITORY,NUM_INHIBITORY,P_CONNECTION,RATIO_SLOW_INHIBITORY_INHIBITORY)

    U_to_e_fast = torch.concat((U_e_to_e_fast,U_i_to_e_fast),1)
    U_to_i_fast = torch.concat((U_e_to_i_fast,U_i_to_i_fast),1)
    U_fast = torch.concat((U_to_e_fast,U_to_i_fast))

    U_to_e_slow = torch.concat((U_e_to_e_slow,U_i_to_e_slow),1)
    U_to_i_slow = torch.concat((U_e_to_i_slow,U_i_to_i_slow),1)
    U_slow = torch.concat((U_to_e_slow,U_to_i_slow))

    scaling_to_e_fast = torch.concat((fast_excitatory*U_e_to_e_fast,
                                      fast_inhibitory*U_i_to_e_fast),1)
    
    scaling_to_i_fast = torch.concat((fast_excitatory*U_e_to_i_fast,
                                      fast_inhibitory*U_i_to_i_fast),1)
    
    scaling_fast = torch.concat((scaling_to_e_fast,scaling_to_i_fast))
    
    scaling_to_e_slow = torch.concat((slow_excitatory*U_e_to_e_slow,
                                      slow_inhibitory*U_i_to_e_slow),1)
    
    scaling_to_i_slow = torch.concat((slow_excitatory*U_e_to_i_slow,
                                      slow_inhibitory*U_i_to_i_slow),1)
    
    scaling_slow = torch.concat((scaling_to_e_slow,scaling_to_i_slow))

    J_e_to_e_fast,l_e_to_e_fast = initialize_j(U_e_to_e_fast,p_pot=P_POTENTIATED,j_p=J_POTENTIATED,j_d=J_DEPRESSED)
    J_e_to_e_slow,l_e_to_e_slow = initialize_j(U_e_to_e_slow,p_pot=P_POTENTIATED,j_p=J_POTENTIATED,j_d=J_DEPRESSED)
    
    J_i_to_e_slow,_ = initialize_j(U_i_to_e_slow,p_pot=1,j_p=J_I_TO_E,j_d=0)
    J_i_to_e_fast,_ = initialize_j(U_i_to_e_fast,p_pot=1,j_p=J_I_TO_E,j_d=0)

    J_e_to_i_slow,_ = initialize_j(U_e_to_i_slow,p_pot=1,j_p=J_E_TO_I,j_d=0)
    J_e_to_i_fast,_ = initialize_j(U_e_to_i_fast,p_pot=1,j_p=J_E_TO_I,j_d=0)

    J_i_to_i_slow,_ = initialize_j(U_i_to_i_slow,p_pot=1,j_p=J_I_TO_I,j_d=0)
    J_i_to_i_fast,_ = initialize_j(U_i_to_i_fast,p_pot=1,j_p=J_I_TO_I,j_d=0)
    
    J_to_e_fast = torch.concat((J_e_to_e_fast,J_i_to_e_fast),1)
    J_to_i_fast = torch.concat((J_e_to_i_fast,J_i_to_i_fast),1)
    J_fast = torch.concat((J_to_e_fast,J_to_i_fast))

    J_to_e_slow = torch.concat((J_e_to_e_slow,J_i_to_e_slow),1)
    J_to_i_slow = torch.concat((J_e_to_i_slow,J_i_to_i_slow),1)
    J_slow = torch.concat((J_to_e_slow,J_to_i_slow))

    w_excitatory = torch.concat((U_e_to_e_fast + U_e_to_e_slow, torch.zeros_like(U_i_to_e_fast)),1)
    w_inhibitory = torch.zeros_like((U_to_i_fast))
    w_mat = torch.concat((w_excitatory,w_inhibitory))

    l_excitatory = torch.concat((l_e_to_e_fast+l_e_to_e_slow,torch.zeros_like(U_i_to_e_fast)),1)
    l_mat = torch.concat((l_excitatory,w_inhibitory))

    return U_slow,U_fast,J_slow,J_fast,w_mat,l_mat, scaling_fast,scaling_slow

@torch.jit.script
def write_results(full_dict: Dict[int,Dict[str,torch.Tensor]],
                  v_t:torch.Tensor,
                  neuron_byte_words:torch.Tensor,
                  num_subsample:int,
                  idx:int):
    
    for p in range(neuron_byte_words.shape[-1]):

        full_dict[p]["membrane_potential_excitatory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["membrane_potential_inhibitory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["spiking_state_var_excitatory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["spiking_state_var_inhibitory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["slow_currents_excitatory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["slow_currents_inhibitory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["fast_currents_excitatory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()
        full_dict[p]["fast_currents_inhibitory"][:,idx+1] = v_t[neuron_byte_words[:,p].to(torch.bool)][0:num_subsample].cpu()

    return full_dict

@torch.jit.script
def get_mean(l_mat:torch.Tensor,
             w_mat:torch.Tensor,
             selection_matrices:list[torch.Tensor]):
    return torch.stack([torch.mean(l_mat[torch.logical_and(w_mat,i)]) for i in selection_matrices])

@torch.jit.script
def get_fraction_above(l_mat: torch.Tensor,
                       w_mat: torch.Tensor,
                       selective_to_selective: torch.Tensor,
                       selective_to_non: torch.Tensor):
    above_threshold_selective = torch.sum(torch.logical_and(w_mat, selective_to_selective) & (l_mat > 0.5)).float() / torch.sum(torch.logical_and(w_mat, selective_to_selective))
    above_threshold_nonselective = torch.sum(torch.logical_and(w_mat, selective_to_non) & (l_mat > 0.5)).float() / torch.sum(torch.logical_and(w_mat, selective_to_non))
    return torch.stack((above_threshold_selective, above_threshold_nonselective))

def get_selectivity_mat(selection_vector):
    # Create a meshgrid of indices for selection_vector
    i, j = torch.meshgrid(selection_vector, selection_vector)
    
    # Calculate mat_1 and mat_2 using vectorized operations
    mat_1 = (i & j).float()
    mat_2 = ((~i) & j).float()
    
    return mat_1, mat_2

def custom_selection_mat(presynaptic_vector,
                         postsynaptic_vector):
    
    return postsynaptic_vector.unsqueeze(-1).float() @ presynaptic_vector.unsqueeze(0).float()

def get_association_pairs(neuron_byte_words,
                          pairs):
    
    association_mat = []
    specific_mat = []
    for pair in pairs:
        pre = neuron_byte_words[:,pair[0]-1].to(torch.bool)
        post = neuron_byte_words[:,pair[1]-1].to(torch.bool)
        association_mat.append(custom_selection_mat(pre,post))
        specific_mat.append(custom_selection_mat(pre,pre))
        specific_mat.append(custom_selection_mat(post,post))

    specific_mat = reduce(torch.stack(specific_mat),"b x y -> x y","max")
    
    association_mat = reduce(torch.stack(association_mat),"b x y -> x y","max")

    association_mat = torch.logical_and(association_mat,torch.logical_not(specific_mat))
    return association_mat,specific_mat

@torch.jit.script
def stdp_simulation_step(v:torch.Tensor,
                    i_s:torch.Tensor,
                    i_f:torch.Tensor,
                    i_e:torch.Tensor,
                    x:torch.Tensor,
                    a:torch.Tensor,
                    s:torch.Tensor,
                    l:torch.Tensor,
                    j_slow:torch.Tensor,
                    j_fast:torch.Tensor,
                    w:torch.Tensor,
                    scaling_slow:torch.Tensor,
                    scaling_fast:torch.Tensor,
                    v_th:torch.Tensor,
                    dtype:torch.dtype,
                    tau_s:torch.Tensor,
                    dt:float,
                    tau_ref:torch.Tensor,
                    v_reset:torch.Tensor,
                    tau_m:torch.Tensor,
                    u:float,
                    tau_rec:float,
                    alpha_rec:float,
                    beta_rec:float,
                    theta_x:float,
                    a_hebbian:float,
                    b_hebbian:float,
                    j_p:float,
                    j_d:float,):
    
    # we must first compute the total currents and update the membrane potential

    sigma_v = sigma(v,v_th=v_th,dtype=dtype)
    x_sigma_v = x*sigma_v

    i_s_new = dis_dt(i_s,tau_s=tau_s,dt=dt,j_slow=scaling_slow*j_slow,x_sigma_v=x_sigma_v)

    i_f_new = (scaling_fast*j_fast) @ x_sigma_v

    i_t = i_s + i_f + i_e

    # we can then update the membrane potential

    a_new = da_dt(v,a,tau_r=tau_ref,v_th=v_th,dt=dt)

    v_new = dv_dt(v=v,a=a,i=i_t,v_reset=v_reset,tau_m=tau_m,
               v_th=v_th,dt=dt)
    
    s_new = ds_dt(v=v,s=s,v_th=v_th,dt=dt)

    # finally, we can update the hidden variables for plasticity

    x_new = dx_dt(x,u=u,tau_r=tau_rec,x_sigma_v=x_sigma_v,dt=dt)
    
    r_mat = r_t(l=l,alpha=alpha_rec,beta=beta_rec,theta_x=theta_x)

    h_t = simplified_stdp(spike_chrono=s,v=v,v_th=v_th,a=a_hebbian,b=b_hebbian)

    l_new = dl_dt(h_t=h_t,w=w,r_t=r_mat,l=l,dt=dt)
    
    j_fast_new = continuous_connectivity(j=j_fast,j_p=j_p,j_d=j_d,l=l,w_mat=w)

    j_slow_new = continuous_connectivity(j=j_slow,j_p=j_p,j_d=j_d,l=l,w_mat=w)
    
    return v_new,i_s_new,i_f_new,x_new,a_new,s_new,j_slow_new,j_fast_new,torch.clamp(l_new,0,1)

@torch.jit.script
def simulation_step(v:torch.Tensor,
                    i_s:torch.Tensor,
                    i_f:torch.Tensor,
                    i_e:torch.Tensor,
                    x:torch.Tensor,
                    a:torch.Tensor,
                    l:torch.Tensor,
                    j_slow:torch.Tensor,
                    j_fast:torch.Tensor,
                    w:torch.Tensor,
                    u_slow:torch.Tensor,
                    u_fast:torch.Tensor,
                    v_th:torch.Tensor,
                    dtype:torch.dtype,
                    tau_s:torch.Tensor,
                    dt:float,
                    tau_ref:torch.Tensor,
                    v_reset:torch.Tensor,
                    tau_m:torch.Tensor,
                    u:float,
                    tau_rec:float,
                    alpha_rec:float,
                    beta_rec:float,
                    theta_x:float,
                    a_hebbian:float,
                    b_hebbian:float,
                    theta_ltp:float,
                    theta_ltd:float,
                    j_p:float,
                    j_d:float,):
    
    # we must first compute the total currents and update the membrane potential

    sigma_v = sigma(v,v_th=v_th,dtype=dtype)
    x_sigma_v = x*sigma_v

    i_s_new = dis_dt(i_s,tau_s=tau_s,dt=dt,j_slow=j_slow,x_sigma_v=x_sigma_v)

    i_f_new = j_fast @ x_sigma_v

    i_t = i_s + i_f + i_e

    # we can then update the membrane potential

    a_new = da_dt(v,a,tau_r=tau_ref,v_th=v_th,dt=dt)

    v_new = dv_dt(v=v,a=a,i=i_t,v_reset=v_reset,tau_m=tau_m,
               v_th=v_th,dt=dt)

    # finally, we can update the hidden variables for plasticity

    x_new = dx_dt(x,u=u,tau_r=tau_rec,x_sigma_v=x_sigma_v,dt=dt)
    
    r_mat = r_t(l=l,alpha=alpha_rec,beta=beta_rec,theta_x=theta_x)
    
    # f_v_t = f_v(v=v,a=a_hebbian,b=b_hebbian,theta_ltp=theta_ltp,
    #          theta_ltd=theta_ltd)
    
    # h_t = h_mat(f_v_t=f_v_t,sigma_v=sigma_v)

    h_t = get_mat(v,v_th=v_th,v_ltp=theta_ltp,v_ltd=theta_ltd,
                  a=a_hebbian,b=b_hebbian)
    
    l_new = dl_dt(h_t=h_t,w=w,r_t=r_mat,l=l,dt=dt)
    
    j_fast_new = update_j(j=j_fast,j_p=j_p,j_d=j_d,threshold=theta_x,l_old=l,
                       l_new=l_new,u=u_fast)
    
    j_slow_new = update_j(j=j_slow,j_p=j_p,j_d=j_d,threshold=theta_x,l_old=l,
                       l_new=l_new,u=u_slow)
    
    return v_new,i_s_new,i_f_new,x_new,a_new,j_slow_new,j_fast_new,torch.clamp(l_new,0,1)

def simulation(minibatch=1,
               total_repetitions=15,
               associativity_experiment=False,
               name="debug"):

    U_slow,U_fast,J_slow,J_fast,W_mat,L_mat,scaling_fast,scaling_slow = get_connectivity_matrices(fast_inhibitory=GABA_SCALING,
                                                                                                  fast_excitatory=1,
                                                                                                  slow_inhibitory=1,
                                                                                                  slow_excitatory=NMDA_SCALING)
    NUM_NEURONS = NUM_EXCITATORY + NUM_INHIBITORY

    if associativity_experiment:
        prototype_indices = get_prototype_indices(6,PER_PROTOTYPES,NUM_NEURONS)
    else:
        prototype_indices = get_prototype_indices(NUM_PROTOTYPES,PER_PROTOTYPES,NUM_NEURONS)

    neuron_byte_words = byte_word(prototype_indices,NUM_NEURONS)

    start_time = name
    
    for batch in range(total_repetitions):
        pairs = [[1,2]]

        if associativity_experiment:
            I_EXT,markers = associativity_trials(pairs=pairs,
                                                 t_stim=int(512/DT),
                                                 t_rest=int(1024/DT),
                                                 num_repetitions=minibatch,
                                                 byte_coding=neuron_byte_words,
                                                 x_noise=X_NOISE,
                                                 num_excitatory=NUM_EXCITATORY,
                                                 mean_excitatory=MU_MEAN_EXCITATORY,
                                                 mean_inhibitory=MU_MEAN_INHIBITORY,
                                                 std_excitatory=SIGMA_STD_EXCITATORY,
                                                 std_inhibitory=SIGMA_STD_INHIBITORY,
                                                 contrast_excitatory=G_CONTRAST_EXCITATORY,
                                                 contrast_inhibitory=G_CONTRAST_INHIBITORY,)
        else:
            I_EXT,markers = generate_trials(byte_coding=neuron_byte_words,
                                    x_noise=X_NOISE,
                                    num_excitatory=NUM_EXCITATORY,
                                    mean_excitatory=MU_MEAN_EXCITATORY,
                                    mean_inhibitory=MU_MEAN_INHIBITORY,
                                    std_excitatory=SIGMA_STD_EXCITATORY,
                                    std_inhibitory=SIGMA_STD_INHIBITORY,
                                    contrast_excitatory=G_CONTRAST_EXCITATORY,
                                    contrast_inhibitory=G_CONTRAST_INHIBITORY,
                                    t_stim=int(512/DT),
                                    t_rest=int(1024/DT),
                                    num_repetitions=minibatch,
                                    initial_rest=batch==0)
        
        I_EXT = I_EXT.cpu().T
        markers = markers.cpu().T
        t_simulation = I_EXT.shape[-1]
        print(I_EXT.shape)

        NUM_SUBSAMPLE = 50

        A_SPIKING_STATE_VAR = torch.zeros([NUM_SUBSAMPLE,t_simulation]).to(STORAGE_DEVICE)
        X_RESOURCES = torch.zeros_like(A_SPIKING_STATE_VAR).to(STORAGE_DEVICE)

        selection_vector = neuron_byte_words[:,0].to(torch.bool)

        HEBBIAN_MEAN = torch.zeros([6,t_simulation])
        reward_selection = neuron_byte_words[:,1].to(torch.bool)
        all_others = torch.logical_not(selection_vector)
        anti_selective = torch.logical_not(torch.logical_or(selection_vector,reward_selection))

        selective_selective = custom_selection_mat(selection_vector,selection_vector)
        selective_nonselective = custom_selection_mat(selection_vector,anti_selective)
        selective_reward = custom_selection_mat(selection_vector,reward_selection)
        selective_allothers = custom_selection_mat(selection_vector,all_others)
        all_associations,all_specific = get_association_pairs(neuron_byte_words,pairs)
        selection_matrices = [selective_selective,selective_nonselective,selective_reward,
                              selective_allothers,all_associations,all_specific]

        if DEBUG:
            POTENTIAL = torch.zeros([NUM_EXCITATORY+NUM_INHIBITORY,t_simulation]).to(STORAGE_DEVICE)

        d = {"orig_slow":J_slow.cpu(),"orig_fast":J_fast.cpu()}

        i_e_t = I_EXT[:,0]
        print(L_mat[selection_vector][0:NUM_SUBSAMPLE].shape)

        HEBBIAN_MEAN[:,0] = get_mean(l_mat=J_fast,w_mat=W_mat,selection_matrices=selection_matrices)

        if batch == 0:
            print("Setting initial values from scratch")
            v_t = torch.zeros_like(i_e_t)
            i_s_t = torch.zeros_like(i_e_t)
            i_f_t = torch.zeros_like(i_e_t)
            x_t = torch.ones_like(i_e_t)
            a_t = torch.zeros_like(i_e_t)
            s_t = torch.zeros_like(i_e_t)
        else:
            print("Setting initial values from previous batch")

            A_SPIKING_STATE_VAR[:,0] = a_t[selection_vector][0:NUM_SUBSAMPLE].to(STORAGE_DEVICE)
            X_RESOURCES[:,0] = x_t[selection_vector][0:NUM_SUBSAMPLE].to(STORAGE_DEVICE)

        for idx in tqdm(range(I_EXT.shape[-1]-1)):

            i_e_t = I_EXT[:,idx]

            if  MODE=="normal":

                v_t,i_s_t,i_f_t,x_t,a_t,J_slow,J_fast,L_mat = simulation_step(v=v_t.cuda(),
                                                                        i_s=i_s_t.cuda(),
                                                                        i_f=i_f_t.cuda(),
                                                                        i_e=i_e_t.cuda(),
                                                                        x=x_t.cuda(),
                                                                        a=a_t.cuda(),
                                                                        l=L_mat,
                                                                        j_slow=J_slow,
                                                                        j_fast=J_fast,
                                                                        w=W_mat,
                                                                        u_slow=U_slow,
                                                                        u_fast=U_fast,
                                                                        v_th=THRESHOLD_POTENTIAL_VECTOR,
                                                                        dtype=DTYPE,
                                                                        tau_s=TAU_S_VEC,
                                                                        dt=DT,
                                                                        tau_ref=MEMBRANE_REFRACTORY_VEC,
                                                                        v_reset=V_RESET_VECT,
                                                                        tau_m=MEMBRANE_CONST_VEC,
                                                                        u=U_SPIKING_COST,
                                                                        tau_rec=TAU_RECOVERY,
                                                                        alpha_rec=ALPHA_PLASTICITY_RECOVERY,
                                                                        beta_rec=BETA_PLASTICITY_RECOVERY,
                                                                        theta_x=THETA_X_PLASTICITY_THRESHOLD,
                                                                        a_hebbian=A_HEBBIAN,
                                                                        b_hebbian=B_HEBBIAN,
                                                                        theta_ltp=THETA_LTP,
                                                                        theta_ltd=THETA_LTD,
                                                                        j_p=J_POTENTIATED,
                                                                        j_d=J_DEPRESSED,)
            elif MODE=="stdp":
                v_t,i_s_t,i_f_t,x_t,a_t,s_t,J_slow,J_fast,L_mat = stdp_simulation_step(v=v_t.cuda(),
                                                                        i_s=i_s_t.cuda(),
                                                                        i_f=i_f_t.cuda(),
                                                                        i_e=i_e_t.cuda(),
                                                                        x=x_t.cuda(),
                                                                        a=a_t.cuda(),
                                                                        s=s_t.cuda(),
                                                                        l=L_mat,
                                                                        j_slow=J_slow,
                                                                        j_fast=J_fast,
                                                                        w=W_mat,
                                                                        scaling_slow=scaling_slow,
                                                                        scaling_fast=scaling_fast,
                                                                        v_th=THRESHOLD_POTENTIAL_VECTOR,
                                                                        dtype=DTYPE,
                                                                        tau_s=TAU_S_VEC,
                                                                        dt=DT,
                                                                        tau_ref=MEMBRANE_REFRACTORY_VEC,
                                                                        v_reset=V_RESET_VECT,
                                                                        tau_m=MEMBRANE_CONST_VEC,
                                                                        u=U_SPIKING_COST,
                                                                        tau_rec=TAU_RECOVERY,
                                                                        alpha_rec=ALPHA_PLASTICITY_RECOVERY,
                                                                        beta_rec=BETA_PLASTICITY_RECOVERY,
                                                                        theta_x=THETA_X_PLASTICITY_THRESHOLD,
                                                                        a_hebbian=A_HEBBIAN,
                                                                        b_hebbian=B_HEBBIAN,
                                                                        j_p=J_POTENTIATED,
                                                                        j_d=J_DEPRESSED,)

            A_SPIKING_STATE_VAR[:,idx+1] = a_t[selection_vector][0:NUM_SUBSAMPLE].to(STORAGE_DEVICE)
            X_RESOURCES[:,idx+1] = x_t[selection_vector][0:NUM_SUBSAMPLE].to(STORAGE_DEVICE)
            HEBBIAN_MEAN[:,idx+1] = get_mean(l_mat=J_fast,w_mat=W_mat,
                                             selection_matrices=selection_matrices).to(STORAGE_DEVICE)
            POTENTIAL[:,idx+1] = s_t.to(STORAGE_DEVICE)

        if not os.path.isdir("results"):
            os.makedirs("results")
        
        d["spiking_state"] = A_SPIKING_STATE_VAR.cpu()
        d["resources"] = X_RESOURCES.cpu()
        d["final_slow"] = J_slow.cpu()
        d["final_fast"] = J_fast.cpu()
        d["selection_vector"] = selection_vector.cpu()
        d["hebbian_dynamics"] = HEBBIAN_MEAN.cpu()
        d["byte_word"] = neuron_byte_words.cpu()
        d["w_mat"] = W_mat[selection_vector][0:NUM_SUBSAMPLE].cpu()
        d["markers"] = markers
        if DEBUG:
            d["potential"] = POTENTIAL.cpu()
        torch.save(d,os.path.join("results",f"results_{start_time}_{batch}.pt"))

if __name__ == "__main__":
    
    simulation(1,NUM_REPETITIONS,ASSOCIATIVITY_EXPERIMENTS,EXPERIMENT_NAME)