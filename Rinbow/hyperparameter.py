import numpy as np

#The size of trainig example 
BATCH_SIZE=32

# size of frame the input tensor to the model.
WIDTH=84
HIGHT=84
CHANNELs=4

# the Number of support in discrete distribution parameter 
ATOMS=51
# the range that the support value derived from.
VMIN=-10
VMAX=10

#numbe of action
ACTIONS=4 #env.action_space.n

# the step size between supports 
DELTA_Z=(VMAX-VMIN)/(ATOMS-1)

# the distribuation parameter.
Z=np.linspace(VMIN,VMAX,ATOMS,dtype=np.float32)