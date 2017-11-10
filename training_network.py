
# coding: utf-8

# In[15]:


import torch
from torch import autograd, nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import gzip
import cPickle
import subprocess
import argparse
import sys


# In[16]:


with gzip.open('atis.small.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())


# In[17]:


word_vector_size=300
tag_vector_size=127
word_vectors=np.random.rand(len(idx2word),word_vector_size)
tag_vectors=np.random.rand(len(idx2label),tag_vector_size)
start_tag=np.random.rand(1,tag_vector_size)
end_tag=np.random.rand(1,tag_vector_size)
tag_labels_embedding=np.zeros(shape=(len(idx2label),len(idx2label)))
np.fill_diagonal(tag_labels_embedding, 1)
input_embed=[] 
input_embedding_labels=[]


# In[18]:


print word_vectors
print tag_vectors.shape
print tag_labels_embedding


# In[19]:


count=0
for i in range(len(train_lex)):
    for j in range (len(train_lex[i])):
        count=count+1
        current_word_id=train_lex[i][j]
        current_word_label=train_y[i][j]
        input_embedding_labels.append(tag_labels_embedding[current_word_label]) 
        if(j==0):
                input_tag_embd=start_tag
                input_word_embd=word_vectors[current_word_id]
                input_tag_embd=input_tag_embd.reshape(1,tag_vector_size)
                input_word_embd=input_word_embd.reshape(1,word_vector_size)
                input_embed.append(np.concatenate([input_word_embd,input_tag_embd],axis=1))
                
                
        elif (j > 0) :
            input_tag_embd=tag_vectors[train_y[i][j-1]]
            input_word_embd=word_vectors[current_word_id]
            input_word_embd=input_word_embd.reshape(1,word_vector_size)
            input_tag_embd=input_tag_embd.reshape(1,tag_vector_size)
            input_embed.append(np.concatenate([input_word_embd,input_tag_embd],axis=1))
           
        


# In[20]:


input_embed=np.asarray(input_embed,dtype=np.float32)
input_embedding_labels=np.asarray(input_embedding_labels)
print input_embed.shape
print input_embedding_labels.shape
input_embedding=np.zeros(shape=(input_embed.shape[0],input_embed.shape[2]))
print input_embedding.shape
for i in range (input_embedding.shape[0]):
    for j in range (input_embedding.shape[1]):
        input_embedding[i][j]=input_embed[i][0][j]
print input_embedding.shape


# In[21]:


print input_embedding[0]
print input_embedding_labels[0]


# In[22]:


#batch_size=5
input_neurons=input_embedding.shape[1]

hidden_neurons=40
output_neurons=input_embedding_labels.shape[1]

learning_r=0.01


#input_num = np.random.rand(5,6)
input_num=torch.from_numpy(np.array(input_embedding,dtype=np.float32))

input=autograd.Variable(input_num,requires_grad=False)

#target_num =np.random.rand(5,5)
#np.fill_diagonal(target_num, 1)
#target_nnum=np.array(target_num)
target_num=torch.from_numpy(np.array(input_embedding_labels,dtype=np.float32))

target=autograd.Variable((target_num))


# In[23]:


print len(target)


# In[24]:


print target


# In[25]:


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


# In[26]:


def correct(output , target):
    correct_out=0
    for i in range (len(output)):
        maxs=output[i][0]
        max_it=0;
        target_it=0
        for j in range (127):
            a=output[i][j]
           # print a
            #print maxs
            if( a > maxs):
                maxs=output[i][j]
                max_it=j
                            
            if(target[i][j]==1):
                target_it=j
        #print "target id" ,
        #print (target_it)
        #print "predicted id",
        #print (max_it)
        if(target_it==max_it):
            correct_out=correct_out+1
    print correct_out
    print len(output)


# In[27]:


opt=torch.optim.Adam(params=net.parameters(),lr=learning_r)

for epoch in range(40):
    print epoch
    output=net(input)
    loss = nn.MSELoss()
    loss_is=loss(output,target)
    #print loss_is
    correct(output.data,target.data)
    net.zero_grad()
    loss_is.backward()
    opt.step()
    


# In[ ]:


print input.data[0]


# In[ ]:


print target.data


# In[ ]:


print output[0][1]

