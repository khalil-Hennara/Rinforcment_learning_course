# import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 

from hyperparameter import *

class SumTree:
    """
    This class is implementation to Sum Tree data structure this data structure is compleat binary Tree 
    and is the same as Priority Qeue where the root node contain the sum of all leaf node in the Tree 
    using this data structure reduce the update time to O(log(n)) and find the sum to O(1).
    The root node have index = 1 and keep the 0 index unuse and that for make the left children is 2*parent_index and that just for simplicity.
    we implement this data structure for using in Proportional Prioritization in the Prioritized Experience Replay from deepmid
    
    Attributes: 
       size : represent the size of buffer 
       tree : it's the binary Tree use to the priority which need to get the sum and update.
       data : where we save the information 
       current_pos: is the pointer to the current postion of the data 
       n_entries : number of information we have.
       max_prio :  this is not relate to the Sum Tree but we need to trake the Max priority for using in our algorithm
                   and by doing this we reduce the time to O(1)
    method :
        -add : using this method to add transition to data list and priority to the tree and updat the tree by 'Bubbel up' as the Priority Qeue 
        -update :  just like 'Bubbel up' where add the priority to empty slot in the tree and then update it's parent until we rich the root node.
        -total  : return the root value which is the sum of all leaves in the tree
        -get_sample : travers in tree to find the priority and the data linked to that priority.
    """
    def __init__(self,buffer_size):
        self.size=buffer_size
        self.tree=[0]*2*buffer_size
        self.data=[]
        self.current_pos=0
        self.n_entries=0
        self.max_prio=1
    
    def add(self,transition,priority):
        """
        in This function we add a new data to the tree 
        parameter: 
            transition: is data we will save it 
            priority : is the value we add to the tree and then update the sum over the whole tree.
        return :
            None, working inplace.
        """
        if self.n_entries<self.size:    
            self.n_entries += 1
            self.data.append(transition)
            node_index=self.current_pos + self.size
            self.update(node_index,priority)
            self.current_pos = (self.current_pos+1) % self.size
        else:
            self.data[self.current_pos]=transition
            
            node_index = self.current_pos + self.size
        
            self.update(node_index,priority)
        
            self.current_pos =(self.current_pos + 1) % self.size
        
            
    def total(self):
        """
        This function return the sum of all leaves in the tree which is root node
        """
        return self.tree[1]
    
    def update(self,node,priority):
        """
        This function update the nodes in the tree adter we add new node or update a value of exist node 
        parameter :
            -node : is the index of node we want to update it's value which is leave
            -priority : is the value to add to tree
        """
        
        self.max_prio=max(self.max_prio,priority)
        
        different=priority-self.tree[node]
        
        self.tree[node] = priority
        
        parent = node // 2
        
        self.tree[parent] += different
        
        while parent !=1:
            parent =parent // 2
            self.tree[parent] += different
    
    
    def _retrieve(self,index,random_value):
        """
        This function is for find the right value in the tree after we provide a random value 
        """
        left_chield = 2*index
        right_chield= 2*index + 1
        
        if left_chield >= len(self.tree):
            return index
        
        if self.tree[left_chield]>=random_value:
            return self._retrieve(left_chield,random_value)
        else :
            return self._retrieve(right_chield,random_value-self.tree[left_chield])
    
    def get_sample(self,random_value):
        """
        In this function we travers the whole tree from the root node to the bottom of the tree using the retrieve method 
        """
        index=self._retrieve(0,random_value)
        
        priority=self.tree[index]
        
        data_index=index-self.size
        try:
        	transition=self.data[data_index]
        except:
        	print("index : ",index)
        	print("data_index:",data_index)
        	print("current_pos",self.current_pos)
        	 
        return (transition,priority,index)
        
        
        
        
class Memory:
    """
    This class is used to creat our replay buffer to use in the *Prioritized Experience Replay* 
    
    Attributes:
        -size : the replay buffer size.
        -tree : is the Sum Tree data structure .
        -alpha : the parameter using in the the stochastic probability priority=(p_i+epsilon)**alpha 
        -epsilon :  is positive small value added to the error 
        -beta : is the prameter used to calculate the weigths
        -BETA_0 : is the statrting value of the beta 
        -BETA_FRAMS : is used for update the beta factor during the training.
    
    method:
        -add : used to add data to the tree.
        -sample_experiences : this function is used to sample experiences from the tree
        -update : update the priority in the tree after training one batch of data
    """
    BETA_0=0.4
    BETA_FRAMES=1e7
    
    def __init__(self,size):
        self.size=size
        self.tree=SumTree(size)
        self.alpha=0.6
        self.epsilon=1e-5
        self.beta=0.4
    
    def _update_beta(self):
        """
        This function update the beta parameter from 0.4 to 1 after 1e7 frame 
        """
        tmp=self.beta+(1-self.BETA_0)/self.BETA_FRAMES
        self.beta=min(1,tmp)
        
        
    def add(self,transition):
        """
        this function add a new data to the memory with priority= max pi 
        and that is the reson, we keep track of the Max priority in the Sum Tree class
        
        parameter :
            -transition : is the data we add (state, action ,reward, done ,next_state ) we add this record to the Tree with 
                          max priority , and that ensure that, this transition will train on it at least once.
        """
        priority=self.tree.max_prio
        self.tree.add(transition,priority)
        
    def sample_experiences(self,batch_size):
        """
        This function used to sample data from the memory 
        parameter :
            - batch_size :  the number of training example we will use for training.
        return :
            - batch : is the data we sample from the memory list with shape [batch_size,1] where each value is (state,action,reward,done,next_state)
            - weights : is the importance sampling (IS) weights 
            - indices :  the index of each training example used for train for used later in update the priority in the Tree.
        """
        batch_data=[]
        indecis=[]
        priorites=[]
        p_total=self.tree.total()
        segment=p_total/batch_size
        self._update_beta()
        for i in range(batch_size):
            low = segment * i
            hight = segment * (i + 1)
            
            value =np.random.uniform(low, hight)
            
            transition,priority,index=self.tree.get_sample(value)
            batch_data.append(transition)
            indecis.append(index)
            priorites.append(priority)
            
        priorites=np.array(priorites)/p_total
        
        weights=np.power(self.tree.n_entries*priorites,-self.beta)
        weights /= np.max(weights)
        
        return (batch_data,weights,indecis)
        
    def update(self,indices,errors):
        """
        This function receive the errors from the model and calculate the priority then calls the update function 
        from the Sum Tree opject on each node used for training.
        parameter : 
            - indices : the index of each training example used for train.
            -error : is the error for each transition used in training model.
 
        """
        priorites=(np.abs(errors)+self.epsilon)**self.alpha
        for index,priority in zip(indices,priorites):
            self.tree.update(index,priority)
            
            
class FactorisedNoisyLayer(keras.layers.Layer):
    
    def __init__(self,units,sigma=0.5,activation=None,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.sigma_0=sigma
        self.activation=keras.activations.get(activation)
    
    def build(self,batch_input_shape):
        std = (3 / batch_input_shape[-1])**0.5
        
        #trainable variable 
        kernel_init = tf.random_uniform_initializer(minval=-std,maxval=std)
        self.kernel = tf.Variable(initial_value=kernel_init(shape=[batch_input_shape[-1],self.units],dtype=tf.float32),name='kernel',trainable=True)
        self.bias =tf.Variable(initial_value=kernel_init(shape=[self.units,],dtype=np.float32),name='bias',trainable=True)
        
        
        sigma_init = tf.constant_initializer(value=self.sigma_0)
        self.sigma_weights = tf.Variable(initial_value=sigma_init(shape=[batch_input_shape[-1],self.units],dtype=tf.float32),name='mu_weights',trainable=True)
        self.sigma_bias =tf.Variable(initial_value=sigma_init(shape=[self.units,],dtype=np.float32),name='mu_bias',trainable=True)
        
    def call(self,inputs):
        
        in_shape=inputs.shape.as_list()[-1]
        
        epsilon_weights = tf.random.normal(shape=(in_shape,1))
        epsilon_bias = tf.random.normal(shape=(1,self.units))
        
        eps_in = tf.sign(epsilon_weights) * tf.math.sqrt(tf.abs(epsilon_weights))
        eps_out = tf.sign(epsilon_bias) * tf.math.sqrt(tf.abs(epsilon_bias))
        
        b = self.bias + self.sigma_bias * eps_out
       
        noise = eps_in @ eps_out
        
        v = self.kernel + self.sigma_weights * noise
        
        return self.activation(inputs @ v + b)
    
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])


class NoisyLayer(keras.layers.Layer):
    
    def __init__(self,units,sigma=0.017,activation=None,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.sigma=sigma
        self.activation=keras.activations.get(activation)
    
    def build(self,batch_input_shape):
        std = (3 / batch_input_shape[-1])**0.5
        #trainable variable 
        kernel_init = tf.random_uniform_initializer(minval=-std,maxval=std)
        self.kernel = tf.Variable(initial_value=kernel_init(shape=[batch_input_shape[-1],self.units],dtype=tf.float32),name='kernel',trainable=True)
        self.bias =tf.Variable(initial_value=kernel_init(shape=[self.units,],dtype=np.float32),name='bias',trainable=True)
        
        
        sigma_init = tf.constant_initializer(value=self.sigma)
        self.sigma_weights = tf.Variable(initial_value=sigma_init(shape=[batch_input_shape[-1],self.units],dtype=tf.float32),name='sigma_weights',trainable=True)
        self.sigma_bias =tf.Variable(initial_value=sigma_init(shape=[self.units,],dtype=tf.float32),name='sigma_bias',trainable=True)
        
    def call(self,inputs):
        in_shape=inputs.shape.as_list()[-1]
        epsilon_weights = tf.random.normal(shape=(in_shape,self.units))
        epsilon_bias = tf.random.normal(shape=(self.units,))
        v = self.kernel + self.sigma_weights * epsilon_weights
        b = self.bias + self.sigma_bias * epsilon_bias
        return self.activation(inputs @ v  + b)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    
def distribution_projection(next_value_dist,rewards,gamma,dones):
        batch_size=rewards.shape[0]
        # in this line we calculate the next value return 
        T_Z=np.expand_dims(rewards,1)+ gamma*np.expand_dims((1-dones),1)*np.expand_dims(Z,0)
        # clip the value in range [VMIN,VMAX]
        clip_T_Z=np.clip(T_Z,VMIN,VMAX)
        
        # next return value postion  bj = (T_z - VMIN)/ Dz
        value_dist_pos=(clip_T_Z-VMIN)/DELTA_Z
        
        # l= lower [bj] , u= upper [bj] 
        lower_bound=np.floor(value_dist_pos).astype(int)
        upper_bound=np.ceil(value_dist_pos).astype(int)
        
        # this is the target distribuation  
        target_distribuation = np.zeros((batch_size,ATOMS))
        
        for i in np.arange(batch_size):
            for j in np.arange(ATOMS):
                if lower_bound[i,j]==upper_bound[i,j]:
                    target_distribuation[i,lower_bound[i,j]] += next_value_dist[i,j]
                else:
                    target_distribuation[i,lower_bound[i,j]]+=(next_value_dist[i,j]*(upper_bound-value_dist_pos)[i,j])
                    target_distribuation[i,upper_bound[i,j]]+=(next_value_dist[i,j]*(value_dist_pos-lower_bound)[i,j])
        return target_distribuation
