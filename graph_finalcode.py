#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm.auto import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from itertools import combinations


# pip freeze

# 
# # 1 Prep - prediction test data
# ## 1-1 Read the data to predict @df_pre; @df_pre_list

# In[2]:


#----------------- predict
# @df_pre
# @df_pre_list
#-----------------------------
read_file = pd.read_csv ('test-public.txt')

df_pre = pd.read_csv(
    'test-public.txt', sep="\t")
df_pre.columns = ['Id','node1', 'node2']
print(df_pre.head(1))
print("-----------------")

df_pre_list = df_pre.values.tolist()
a = random.sample(df_pre_list,1)
print(a)


# ## 1-2 Use a few to start with @df_pre_list

# In[44]:


df_pre_list_pick = []
r_from = 105
r_to = 106
r_range = range(r_from, r_to+1)

for i in r_range:
    df_pre_list_pick.append(df_pre_list[i-1])

#print(df_pre_list_pick)


# ### 1-2-1 Put all the nodes in the prediction test data into @ab

# In[45]:


#--------------------all nodes
## @ab
#-----------------------------
ab = []

for x in df_pre_list_pick:
    ab.append(str(x[1]))
    ab.append(str(x[2]))

# remove duplicate items from the list
ab = list(dict.fromkeys(ab))
print(random.sample(ab,1), "len: ",len(ab))


# # 2 Training data

# ## [optional] 2-1 Training data - sample data to test @data_list_part1

# file = 'train_part1.txt'
# with open(file) as file:
#     rows = (line.split('\t') for line in file)
#     data_list_part1 = [row[0:] for row in rows]
# for x in data_list_part1:
#     x[-1] = x[-1].rstrip('\n')
# len(data_list_part1)

# ### [optional] 2-1-1 only pick short data @short_data_list

# short_data_list = []
# for x in data_list_part1:
#     if 200 < len(x) < 300 :
#         short_data_list.append(x)
# print(len(short_data_list))

# ## 2-2 Using all training data @data_list

# In[46]:


file = 'train.txt'
with open(file) as file:
    rows = (line.split('\t') for line in file)
    data_list = [row[0:] for row in rows]
for x in data_list:
    x[-1] = x[-1].rstrip('\n')
len(data_list)


# In[53]:


short_data_list = []

for x in data_list:
    if 30 < len(x) < 50:
        short_data_list.append(x)

#short_data_list_random = list(random.sample(short_data_list,20))
#print(len(short_data_list_random))
print(len(short_data_list))
print(random.sample(short_data_list,2))


# ### 2-2-1 Get sample traning data containing prediction test data @abab

# In[54]:


# 1st iteration
abab = []
x_list = []
for x in tqdm(ab):
    for y in short_data_list:
       # if (100 < len(y) <= 150) and (x in y):
        if y not in abab:
            abab.append(y)
        #if x not in x_list:
         #   x_list.append(x)

print("1st: ",len(abab))
#abab = random.sample(abab,20)
#print("sample: ",len(abab))


# 
# d# 2nd iteration
# ab2 = []
# for i in ab:
#     if i not in x_list:
#         ab2.append(i)
# 
# for x in ab2:
#     for y in data_list:
#         if (200 < len(y) <= 300) and (x in y):
#             abab.append(y)
#             x_list.append(x)
#             break
#             
# print("2nd: ",len(x_list))
# 
# d# 3rd iteration
# ab3 = []
# for i in ab2:
#     if i not in x_list:
#         ab3.append(i)
# 
# for x in ab3:
#     for y in data_list:
#         if (300 < len(y)< 1000) and (x in y):
#             abab.append(y)
#             x_list.append(x)
#             break
#             
# print("3rd: ",len(x_list))

# ## 2-3 Combine sample data from data_list_part1 (first few rows) & abab (containing prediction test nodes from all data)

# print(len(abab))
# print(len(short_data_list))

# short_data_list = short_data_list + abab
# print(len(short_data_list))

# # 3 Pair all nodes @df

# In[55]:


# dic - put into a dictionary (removes duplicates)
data_dict = {x[0]:x[1:] for x in abab}
print(len(data_dict))
#print(data_dict)


# In[56]:


pairs = []
for key in data_dict: #each row
    for val in data_dict[key]:
        pair = [key,val]
        pairs.append(pair)

node1_list = []
node2_list = []
for x in pairs:
    node1_list.append(x[0])
    node2_list.append(x[1])

df = pd.DataFrame({'node1': node1_list, 'node2': node2_list})
df.shape
df.head(2)


# # 4 Create a graph

# In[57]:


G = nx.from_pandas_edgelist(df, "node1", "node2",create_using=nx.Graph() )


# plt.figure(figsize=(10,10))
# 
# pos = nx.random_layout(G, seed=23)
# nx.draw(G, with_labels=False,  pos = pos, node_size = 4, alpha = 0.6, width = 0.1)
# 
# plt.show()

# # 5 Put all nodes into @node_list

# In[58]:


# combine all nodes in a list
node_list = node1_list + node2_list
# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))
len(node_list)


# 
# #print(abab)
# 
# node_list_long = []
# for x in abab:
#     for y in x:
#         node_list_long.append(y)
# node_list_unique = list(dict.fromkeys(node_list_long))
# 
# node_list_short = list(random.sample(node_list_unique,4000))
# node_list_short = node_list_short + ab
# print(len(node_list_short))
# node_list_short = list(dict.fromkeys(node_list_short))
# node_list = node_list_short

# # 6 all pair list

# In[43]:


df_list = df.values.tolist()
print(len(df_list))
print(random.sample(df_list,2))


# # 7 All connected pairs @pairs_all - takes long

# In[36]:


pairs_all = []
for x in tqdm(range(len(node_list))):
    for y in range(x+1,len(node_list)):
        if nx.has_path(G,node_list[x], node_list[y]) == True:
           # if nx.shortest_path_length(G,node_list[x],node_list[y]) <=2:
            pairs_all.append([node_list[x], node_list[y]])


# pairs_all = df_list

# # 8 all unconnected nodes @data (df) 

# In[ ]:


no_edge_list = []
for x in tqdm(pairs_all):
    if x not in pairs:
        no_edge_list.append(x)
        
no_edge_node1 = []
no_edge_node2 = []
for x in tqdm(no_edge_list):
    no_edge_node1.append(x[0])
    no_edge_node2.append(x[1])
         
data = pd.DataFrame({'node1':no_edge_node1, 'node2':no_edge_node2})
data['link'] = 0


# In[ ]:


print(len(no_edge_list),len(pairs))


# # 9 Find removable edges

# In[ ]:


initial_node_count = len(G.nodes)
df_temp = df.copy()
removable_edges = []

for i in tqdm(df.index.values):
    #remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(df_temp.drop(index=i), "node1", "node2", create_using=nx.Graph())
    # check if graph is still valid
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        removable_edges.append(i)
        df_temp = df_temp.drop(index=i)

        
print(len(removable_edges))


# # 10 all removable edges set to 1

# In[ ]:


# create dataframe of removable edges
df_ghost = df.loc[removable_edges]

# add the target variable 'link'
df_ghost['link'] = 1

data = data.append(df_ghost[['node1', 'node2', 'link']], ignore_index=True)


# In[ ]:


data['link'].value_counts()


# # 11 Draw a graph again without removable edges

# In[ ]:


# drop removable edges
df_partial = df.drop(index=df_ghost.index.values)

# build graph
G_data = nx.from_pandas_edgelist(df_partial, "node1", "node2", create_using=nx.Graph())


# # 12 Use node2vec to create a model

# In[ ]:


from node2vec import Node2Vec

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=3, min_count=1)


# # 13 Apply unconnected & removable edge data to the model

# In[ ]:


x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['node1'], data['node2'])]


# # 14 Train the model

# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)


# In[ ]:


lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)


# # 15 Prediction of the model

# In[ ]:


predictions = lr.predict_proba(xtest)


# In[ ]:


roc_auc_score(ytest, predictions[:,1])


# # 16 Input our data for prediction, train & predict

# In[ ]:


print(df_pre_list_pick)


# In[ ]:


a = data['node1'].to_list()
a = list(dict.fromkeys(a))
b = data['node2'].to_list()
b = list(dict.fromkeys(b))
data_node_list = a+b
data_node_list = list(dict.fromkeys(data_node_list))

aa = []
bb = []
for x in df_pre_list:
    if (str(x[1]) in a) and (str(x[2]) in b):
        print(x[0])
        aa.append(str(x[1]))
        bb.append(str(x[2]))


# In[ ]:


XX = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(aa,bb)]


# In[ ]:


predictions_pre = lr.predict_proba(XX)


# In[ ]:


i = 0
while 0 <= i < len(predictions_pre):
    print(predictions_pre[i,1])
    i += 1


# # 17 Apply parameters

# In[ ]:


from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

split = int(x_return.__len__() * 0.8)
xtrain = xtrain[:split]
xvalid = xtrain[split:]

ytrain = xtrain[:split]
yvalid = xtrain[split:]

#Train and Validation
est = make_pipeline(StandardScaler(), SGDClassifier(loss="log", penalty="l2", alpha=0.01, shuffle=True))
est.fit(xtrain,ytrain)

y_pred = est.predict(xvalid)
valid_res = accuracy_score(yvalid, y_pred) * 100
print(valid_res) #AUC : 87

#Probability for Kaggle Submission
y_test_pr = est.predict_proba(xtest)
fb_df = pd.DataFrame(y_test_pr[:,1])
fb_df.to_clipboard(sep=',', index=False)


# In[ ]:
pre_param = est.predict(df_tp_pre)

