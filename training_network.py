
# coding: utf-8

# In[144]:


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


# In[145]:


with gzip.open('atis.small.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())


# In[146]:


word_vector_size=300
tag_vector_size=len(idx2label) + 2
word_vectors=np.random.rand(len(idx2word),word_vector_size)
tag_vectors=np.zeros(shape=(len(idx2label)+2,len(idx2label)+2))
#print tag_vectors.shape
np.fill_diagonal(tag_vectors, 1)
start_tag=tag_vectors[127]
end_tag=tag_vectors[128]


# In[147]:


def embedding(train_lex,train_y):
    
    
    #tag_labels_embedding=np.zeros(shape=(len(idx2label)+1,len(idx2label)+1))
    #np.fill_diagonal(tag_labels_embedding, 1)
    input_embed=[] 
    input_embedding_labels=[]
    
    for i in range(len(train_lex)):
        for j in range (len(train_lex[i])):
            current_word_id=train_lex[i][j]
            current_word_label=train_y[i][j]
            input_embedding_labels.append(tag_vectors[current_word_label]) 
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
    
    input_embed=np.asarray(input_embed,dtype=np.float32)
    input_embedding_labels=np.asarray(input_embedding_labels)
    print input_embed.shape
    print input_embedding_labels.shape
    input_embedding=np.zeros(shape=(input_embed.shape[0],input_embed.shape[2]))
    print input_embedding.shape
    for i in range (input_embedding.shape[0]):
        for j in range (input_embedding.shape[1]):
            input_embedding[i][j]=input_embed[i][0][j]
    
    #print input_embedding.shape
    return input_embedding , input_embedding_labels    


# In[148]:


print tag_vectors.shape


# In[149]:


#input_embedding,input_embedding_labels=embedding(train_lex,train_y)


# In[150]:


#print input_embedding.shape
#print input_embedding_labels.shape


# In[151]:


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


# In[152]:


print input_embedding_labels.shape[1]


# In[153]:


print tag_vectors.shape


# In[154]:


def neural_net(input_embedding,input_embedding_labels):
    input_neurons=input_embedding.shape[1]
    hidden_neurons=140
    output_neurons=input_embedding_labels.shape[1]
    learning_r=0.01
  
    input_num=torch.from_numpy(np.array(input_embedding,dtype=np.float32))
    input=autograd.Variable(input_num,requires_grad=False)
   
    target_num=torch.from_numpy(np.array(input_embedding_labels,dtype=np.float32))
    target=autograd.Variable((target_num))
   
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
    opt=torch.optim.Adam(params=net.parameters(),lr=learning_r)

    for epoch in range(1):
        print epoch
        output=net(input)
        loss = nn.MSELoss()
        loss_is=loss(output,target)
        print loss_is
        #correct(output.data,target.data)
        net.zero_grad()
        loss_is.backward()
        opt.step()
    print (output)
    
    return net


# In[155]:


input_embedding,input_embedding_labels=embedding(train_lex,train_y)


# In[156]:


print tag_vectors.shape


# In[157]:



net=neural_net(input_embedding,input_embedding_labels)


# In[158]:


print net


# In[159]:


#torch.save(net,'/home/sbalekuttira/trained_model')


# In[176]:


for sent in range (len(test_lex)):
    len_sentence = len(test_lex[sent]) + 2
    num_of_labels = 129
    dp_table = np.zeros((num_of_labels,len_sentence))
    backtrace = np.zeros((num_of_labels,len_sentence))
    
    start_label_pos = num_of_labels - 2
    dp_table[start_label_pos][0] = 1
    for j in range (len(test_lex[sent])):
        current_word_id=test_lex[sent][j]
        #print current_word_id
        probs=np.zeros(shape=(129,129))
        for k in range (129):
                input_tag_embed=tag_vectors[k]
                #print input_tag_embed.size
                input_word_embed=word_vectors[current_word_id]
                input_embedding=np.concatenate([input_word_embed,input_tag_embed],axis=0)
               # print input_embedding
                input_embedding=torch.from_numpy(np.array(input_embedding,dtype=np.float32))
                input=autograd.Variable(input_embedding,requires_grad=False)
               # print input
                output=net(input)
                #print output.data[0]
                for h in range(129):
                    probs[k][h]=output.data[h]
                
               # print probs[k]
        
        for i in range(len_sentence - 1):
               # print(dp_table)
    #Output table from neural network
                word_output_table = np.random.rand(num_of_labels,num_of_labels)
               # print(word_output_table)
    
                max_vector = []
                max_vector[:] = []
                back_vector = []
                back_vector[:] = []
    #multiple previous label probability with neural network output and store the max
                for col in range(num_of_labels):
                    word_output_table[:,col] = np.multiply(word_output_table[:,col],dp_table[:,i])
                    col_to_check = word_output_table[:,col].tolist()
                    max_col = max(col_to_check)
                    max_vector.append(max_col)
                    back_vector.append(col_to_check.index(max_col))
                   # print(word_output_table)
                    #print(max_vector)
    
    #fill the dp table
                for col in range(num_of_labels):
                    dp_table[col][i+1] = max_vector[col]
                    backtrace[col][i+1] = back_vector[col]
                  #  print(dp_table)
                   # print(backtrace)
            
    end_label_pos = num_of_labels - 1

    i = len_sentence - 1
    index_to_check = end_label_pos
    final_label = []
    final_label[:]=[]
    while(i > 1):
        vector_to_check = backtrace[:,i]
        #print vector_to_check
        #print index_to_check
        final_label.append(int(vector_to_check[index_to_check]))
        index_to_check = int(vector_to_check[index_to_check])
        i = i-1

        #print final_label
    final_label.reverse()
    print len(final_label)


# In[174]:


print test_lex[0]
print test_y[0]

