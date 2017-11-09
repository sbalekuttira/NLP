
# coding: utf-8

# In[261]:


import gzip
import cPickle
import numpy as np
import subprocess
import argparse
import sys
import torch
from torch import autograd, nn
with gzip.open('atis.small.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
    


# In[262]:


#print type(idx2label)
#print type(idx2word)
 #'''
  #  To have a look what the original data look like, commnet them before your submission
   # '''
print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
print test_y[0], map(lambda t: idx2label[t], test_y[0])


# In[263]:


#initilise all the vectors with random numbers
#added extra POS tag 'start' and 'end'

word_vector_size=3
tag_vector_size=3
word_vectors=np.random.rand(len(idx2word),word_vector_size)
tag_vectors=np.random.rand(len(idx2label),tag_vector_size)
start_tag=np.random.rand(1,tag_vector_size)
end_tag=np.random.rand(1,tag_vector_size)
tag_labels_embedding=np.zeros(shape=(len(idx2label),len(idx2label)))
np.fill_diagonal(tag_labels_embedding, 1)
input_embed=[] 
input_embedding_labels=[]


# In[264]:


print word_vectors[554]
print word_vectors[104]
print tag_vectors[126]
print tag_vectors[123]
print tag_vectors[2]
print (start_tag)
#print input_embedding.shape


# In[265]:


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
                
        else :
            input_tag_embd=tag_vectors[train_y[i][j-1]]
            input_word_embd=word_vectors[current_word_id]
            input_word_embd=input_word_embd.reshape(1,word_vector_size)
            input_tag_embd=input_tag_embd.reshape(1,tag_vector_size)
            input_embed.append(np.concatenate([input_word_embd,input_tag_embd],axis=1))
        
  


# In[266]:


input_embed=np.asarray(input_embed,dtype=np.float32)
input_embedding_labels=np.asarray(input_embedding_labels)

input_embedding=np.zeros(shape=(input_embed.shape[0],input_embed.shape[2]))
for i in range (input_embed.shape[0]):
    for j in range (input_embed.shape[2]):
        input_embedding[i][j]=input_embed[i][0][j]


# In[267]:


print input_embedding.shape

