import torch
from connectivity import*
from constants import*
from dynamics import*
from plasticity import*
from tqdm import tqdm

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
					u_fast:torch.Tensor):
	
	# we must first compute the total currents and update the membrane potential

	sigma_v = sigma(v,THRESHOLD_POTENTIAL_VECTOR,DTYPE)
	x_sigma_v = x*sigma_v

	i_s_new = i_s + dis_dt(i_s,TAU_S_VEC,DT,j_slow,x_sigma_v)

	i_f_new = j_fast @ x_sigma_v
	# i_f_new = torch.sparse.mm(j_fast.to(torch.float32).to_sparse(),x_sigma_v.to(torch.float32).unsqueeze(-1))
	# i_f_new = i_f_new.to(DTYPE).squeeze()

	i_t = i_s + i_f + i_e

	# we can then update the membrane potential

	da = da_dt(v,MEMBRANE_REFRACTORY_VEC,THRESHOLD_POTENTIAL_VECTOR,DT)
	dv = dv_dt(v,i_t,MEMBRANE_CONST_VEC,THRESHOLD_POTENTIAL_VECTOR,DT)
	a_new = torch.zeros_like(a) + (a+da)*(a<1)
	v_new = V_RESET_VECT*(a>=1) + (v+dv)*(a<1)

	# finally, we can update the hidden variables for plasticity

	x_new = x + dx_dt(x,U_SPIKING_COST,TAU_RECOVERY,x_sigma_v,DT)

	r_mat = r_t(l=l,alpha=ALPHA_PLASTICITY_RECOVERY,beta=BETA_PLASTICITY_RECOVERY,
			 theta_x=THETA_X_PLASTICITY_THRESHOLD)

	f_v_t = f_v(v,A_HEBBIAN,B_HEBBIAN,THETA_LTP,THETA_LTD,THRESHOLD_POTENTIAL_VECTOR,
			 V_RESET_VECT)
	
	h_t = rearrange(f_v_t,"n -> n 1") @ rearrange(sigma_v,"n -> 1 n")
	
	l_new = l + dl_dt(h_t,w,r_mat,DT)

	# we don't want to create any new slow/fast connection so we need to mask
	
	j_fast_new = u_fast*update_j(j_fast,J_POTENTIATED,J_DEPRESSED,THETA_X_PLASTICITY_THRESHOLD,
					   l,l_new)
	
	j_slow_new = u_slow*update_j(j_slow,J_POTENTIATED,J_DEPRESSED,THETA_X_PLASTICITY_THRESHOLD,
					   l,l_new)
	
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
																		u_fast=U_fast)
		
		V_MEMBRANE_POTENTIALS[:,idx+1] = v_new
		I_S_SLOW_CURRENTS[:,idx + 1] = i_s_new
		I_F_FAST_CURRENTS[:,idx + 1] = i_f_new
		X_RESOURCE_STATE_VAR[:,idx + 1] = x_new
		A_SPIKING_STATE_VAR[:,idx + 1] = a_new