import torch

# ----------------------
# ----------------------

NUM_EXCITATORY = 160
NUM_INHIBITORY = 40
P_CONNECTION = 0.25 # should be 0.2 but 0.4 for debugging small networks

# for generalizability, all connections could be slow or fast
RATIO_SLOW_EXCITATORY_EXCITATORY = 0.5
RATIO_SLOW_INHIBITORY_INHIBITORY = 0.1
RATIO_SLOW_EXCITATORY_INHIBITORY = 0
RATIO_SLOW_INHIBITORY_EXCITATORY = 0

P_POTENTIATED = 0.2 # 0.2
J_POTENTIATED = 0.21 # 0.21
J_DEPRESSED = 0.03 # 0.03
J_E_TO_I = 0.08 # 0.08
J_I_TO_E = -0.18 #-0.18
J_I_TO_I = -0.18 #-0.18

# ----------------------
# ----------------------

REFRACTORY_TIME = 2
MEMBRANE_REFRACTORY_VEC = REFRACTORY_TIME*torch.ones([NUM_EXCITATORY + NUM_INHIBITORY])

MEMBRANE_CONST_EXCITATORY = 20
MEMBRANE_CONST_INHIBITORY = 10
MEMBRANE_CONST_VEC = torch.concat((MEMBRANE_CONST_EXCITATORY*torch.ones(NUM_EXCITATORY),
						  MEMBRANE_CONST_INHIBITORY*torch.ones(NUM_INHIBITORY)))

THRESHOLD_POTENTIAL = 20
V_RESET_EXCITATORY = 15
V_RESET_INHIBITORY = 10
THRESHOLD_POTENTIAL_VECTOR = THRESHOLD_POTENTIAL*torch.ones([NUM_EXCITATORY + NUM_INHIBITORY])
V_RESET_VECT = torch.concat((V_RESET_EXCITATORY*torch.ones(NUM_EXCITATORY),
							 V_RESET_INHIBITORY*torch.ones(NUM_INHIBITORY)))

# ----------------------
# ----------------------

DT = 0.1
T_MAX = 1000
TIMELINE = torch.linspace(0,T_MAX,int(T_MAX/DT))
T_IDX_MAX = len(TIMELINE)

# ----------------------
# ----------------------

TAU_S_EXCITATORY = 	10
TAU_S_INHIBITORY = 10
TAU_S_VEC = torch.concat((TAU_S_EXCITATORY*torch.ones(NUM_EXCITATORY),
						  TAU_S_INHIBITORY*torch.ones(NUM_INHIBITORY)))

# ----------------------
# ----------------------

TAU_RECOVERY = 200

# ----------------------
# ----------------------

ALPHA_PLASTICITY_RECOVERY = 0.0147
BETA_PLASTICITY_RECOVERY = 0.01
THETA_X_PLASTICITY_THRESHOLD = 0.4
U_SPIKING_COST = 0.45

# ----------------------
# ----------------------

A_HEBBIAN = 0.25
B_HEBBIAN = 0.17
THETA_LTP = 17.5
THETA_LTD = 15.5