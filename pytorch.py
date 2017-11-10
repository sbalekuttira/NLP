
# coding: utf-8

# In[199]:


import torch
from torch import autograd, nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# In[200]:


batch_size=5
input_neurons=6

hidden_neurons=10
output_neurons=5

learning_r=0.001



input_num = np.random.rand(5,6)
input_num=torch.from_numpy(np.array(input_num,dtype=np.float32))

input=autograd.Variable(input_num)

target_num =np.random.rand(5,5)
np.fill_diagonal(target_num, 1)
#target_nnum=np.array(target_num)
target_num=torch.from_numpy(np.array(target_num,dtype=np.float32))

target=autograd.Variable((target_num))

#target = autograd.Variable((torch.rand(batch_size) * output_neurons).long())


# In[201]:


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


# In[202]:


opt=torch.optim.Adam(params=net.parameters(),lr=learning_r)

for epoch in range(100):
    output=net(input)
    loss = nn.MSELoss()
    loss_is=loss(output,target)
    print loss_is
    net.zero_grad()
    loss_is.backward()
    opt.step()
    

