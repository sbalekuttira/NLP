
# coding: utf-8

# In[131]:


import torch
from torch import autograd, nn
import numpy as np
import torch.nn.functional as F


# In[132]:


batch_size=5
input_neurons=6

hidden_neurons=10
output_neurons=5




input_num = np.random.rand(5,6)
input_num=torch.from_numpy(np.array(input_num,dtype=np.float32))

input_torch=autograd.Variable(input_num)


target_num = np.zeros(shape=(5,5))
np.fill_diagonal(target_num, 1)
target_num=torch.from_numpy(np.array(target_num,dtype=np.float32))

target_torch=autograd.Variable((target_num).long())


# In[133]:


#print input_num
#print input_torch
print target_torch


# In[134]:


class Net(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_neurons, hidden_neurons) 
        #self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_neurons, output_neurons)  
    
    def forward(self, x):
        x = self.layer_1(x)
        x = F.tanh(x)
        x = self.layer_2(x)
        x = F.softmax(x)
        return x
    
net = Net(input_neurons, hidden_neurons, output_neurons)


# In[135]:


output=net(input_torch)


# In[136]:


print output


# In[137]:


print target_torch


# In[138]:


loss= F.nll_loss(output,target_torch)

