import torch
from connectivity import*
from constants import*
from dynamics import*
from plasticity import*
from tqdm import tqdm
import os
from datetime import datetime

if torch.cuda.is_available():
	torch.set_default_dtype(torch.float16)
	torch.set_default_device("cuda")
	DTYPE = torch.float16
else:
	torch.set_default_dtype(torch.float32)
	DTYPE = torch.float32

def get_connectivity_matrices():

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

	return U_slow,U_fast,J_slow,J_fast,w_mat,l_mat

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

	# a_new = da_dt(v,a,MEMBRANE_REFRACTORY_VEC,THRESHOLD_POTENTIAL_VECTOR,DT)
	a_new = da_dt(v,a,tau_r=tau_ref,v_th=v_th,dt=dt)
	# v_new = dv_dt(v=v,a=a,i=i_t,v_reset=V_RESET_VECT,tau_m=MEMBRANE_CONST_VEC,
	# 		   v_th=THRESHOLD_POTENTIAL_VECTOR,dt=DT)
	v_new = dv_dt(v=v,a=a,i=i_t,v_reset=v_reset,tau_m=tau_m,
			   v_th=v_th,dt=dt)

	# finally, we can update the hidden variables for plasticity

	# x_new = dx_dt(x,U_SPIKING_COST,TAU_RECOVERY,x_sigma_v,DT)
	x_new = dx_dt(x,u=u,tau_r=tau_rec,x_sigma_v=x_sigma_v,dt=dt)

	# r_mat = r_t(l=l,alpha=ALPHA_PLASTICITY_RECOVERY,beta=BETA_PLASTICITY_RECOVERY,
	# 		 theta_x=THETA_X_PLASTICITY_THRESHOLD)
	
	r_mat = r_t(l=l,alpha=alpha_rec,beta=beta_rec,theta_x=theta_x)

	# f_v_t = f_v(v,A_HEBBIAN,B_HEBBIAN,THETA_LTP,THETA_LTD,THRESHOLD_POTENTIAL_VECTOR,
	# 		 V_RESET_VECT)
	
	f_v_t = f_v(v,a=a_hebbian,b=b_hebbian,theta_ltp=theta_ltp,
			 theta_ltd=theta_ltd,v_th=v_th,v_reset=v_reset)
	
	h_t = h_mat(f_v_t,sigma_v)
	
	# l_new = dl_dt(h_t,w,r_mat,l,DT)
	l_new = dl_dt(h_t=h_t,w=w,r_t=r_mat,l=l,dt=dt)

	# we don't want to create any new slow/fast connection so we need to mask
	
	# j_fast_new = update_j(j_fast,J_POTENTIATED,J_DEPRESSED,THETA_X_PLASTICITY_THRESHOLD,
	# 				   l,l_new,u_fast)
	
	j_fast_new = update_j(j=j_fast,j_p=j_p,j_d=j_d,threshold=theta_x,l_old=l,
					   l_new=l_new,u=u_fast)
	
	# j_slow_new = update_j(j_slow,J_POTENTIATED,J_DEPRESSED,THETA_X_PLASTICITY_THRESHOLD,
	# 				   l,l_new,u_slow)
	
	j_slow_new = update_j(j=j_slow,j_p=j_p,j_d=j_d,threshold=theta_x,l_old=l,
					   l_new=l_new,u=u_slow)
	
	return v_new,i_s_new,i_f_new,x_new,a_new,j_slow_new,j_fast_new

if __name__ == "__main__":
	U_slow,U_fast,J_slow,J_fast,W_mat,L_mat = get_connectivity_matrices()
	NUM_NEURONS = NUM_EXCITATORY + NUM_INHIBITORY
	V_MEMBRANE_POTENTIALS = torch.zeros([NUM_EXCITATORY + NUM_INHIBITORY,T_IDX_MAX])
	A_SPIKING_STATE_VAR = torch.zeros_like(V_MEMBRANE_POTENTIALS)
	X_RESOURCE_STATE_VAR = torch.ones_like(V_MEMBRANE_POTENTIALS)
	I_S_SLOW_CURRENTS = torch.zeros_like(V_MEMBRANE_POTENTIALS)
	I_F_FAST_CURRENTS = torch.zeros_like(V_MEMBRANE_POTENTIALS)

	AMPLITUDES = 5*torch.rand([NUM_NEURONS])
	AMPLITUDES = rearrange(AMPLITUDES,"n -> n 1")
	PHASES = 10*torch.rand([NUM_NEURONS])
	FREQUENCIES = 5*torch.rand([NUM_NEURONS])+5

	t_sin = repeat(TIMELINE,"t ->t n",n=NUM_NEURONS)
	t_sin = (t_sin/FREQUENCIES) + PHASES
	t_sin = rearrange(t_sin,"t n -> n t")

	I_EXT = AMPLITUDES*torch.sin(t_sin)+AMPLITUDES

	for idx in tqdm(range(T_IDX_MAX-1)):
		v_t = V_MEMBRANE_POTENTIALS[:,idx]
		i_s_t = I_S_SLOW_CURRENTS[:,idx]
		i_f_t = I_F_FAST_CURRENTS[:,idx]
		i_e_t = I_EXT[:,idx]
		x_t = X_RESOURCE_STATE_VAR[:,idx]
		a_t = A_SPIKING_STATE_VAR[:,idx]

		v_new,i_s_new,i_f_new,x_new,a_new,J_slow,J_fast = simulation_step(v=v_t,
																		i_s=i_s_t,
																		i_f=i_f_t,
																		i_e=i_e_t,
																		x=x_t,
																		a=a_t,
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
		
		V_MEMBRANE_POTENTIALS[:,idx+1] = v_new
		I_S_SLOW_CURRENTS[:,idx + 1] = i_s_new
		I_F_FAST_CURRENTS[:,idx + 1] = i_f_new
		X_RESOURCE_STATE_VAR[:,idx + 1] = x_new
		A_SPIKING_STATE_VAR[:,idx + 1] = a_new

	if not os.path.isdir("results"):
		os.makedirs("results")
	
	d = {"membrane_potentials":V_MEMBRANE_POTENTIALS.cpu(),
	  "slow_currents":I_S_SLOW_CURRENTS.cpu(),
	  "fast_current":I_F_FAST_CURRENTS.cpu(),
	  "x_resources":X_RESOURCE_STATE_VAR.cpu(),
	  "spiking_state":A_SPIKING_STATE_VAR.cpu()}
	
	torch.save(d,os.path.join("results",f"results_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"))