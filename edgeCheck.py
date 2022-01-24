import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time
#import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# function to get unique values

def str_float(str):

    str.replace("'", '').replace(" ", '')
    if str == '': str = '-1'
    fl = float(str)
    return fl


def load_data(filepath):
    f = open(filepath)
    M = []
    node_list_1 = []
    node_list_2 = []


    for line in f:
        cols = line.replace('\n', '').split('\t')
        M.append(cols)

# 0|| 1234

        for i in range(len(cols)):
            if i == 0:
                pass
            else:
                node_list_1.append(cols[0])
                node_list_2.append(cols[i])


    #return_x.append([node_list_1, node_list_2])

    fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    #G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())
    #### create graph
    """G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

    # plot graph
    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)

    plt.show()"""
    #### create graph

    x_return = []
    y_return = []
    #1인 애들만 쌓임
    node_list = []
    node_list.append([node_list_1, node_list_2])

    dist_node = list(dict.fromkeys(node_list_1+ node_list_2))


    for line in M:

        for node in dist_node:
            flg = 0
            for col in line: #123

                if node == col and line.index(col) != 0:
                    flg = flg + 1
                else :
                    flg = flg + 0
            if flg == 0:
                y_return.append(0)
            if flg == 1:
                y_return.append(1)

            x_return.append([str_float(line[0]), str_float(node)])



    split = int(x_return.__len__()*0.8)
    x_train = x_return[:split]
    x_valid = x_return[split:]

    y_train = y_return[:split]
    y_valid = y_return[split:]





    return x_train,x_valid, y_train, y_valid


def load_data_test(filepath):
    f = open(filepath)

    node_list_1 = []
    node_list_2 = []
    return_x = []

    i = -1
    for line in f:
        if i == -1 : i = i+1
        else :
            cols = line.replace('\n', '').split('\t')
            node_list_1.append(str_float(cols[1]))
            node_list_2.append(str_float(cols[2]))
            return_x.append([str_float(cols[1]),str_float(cols[2])])

    # combine all nodes in a list
    #del node_list_1[0]
    #del node_list_2[0]


    return return_x


# region [Main]
#train_x = load_data('./Data/train.txt')  # 20000rows * dynamic
x_train,x_valid, y_train, y_valid = load_data('./Data/train_short.txt')  # 5rows only * dynamic
x_test = load_data_test('./Data/test-public.txt')  # 2001 rows incl header * 3


###### Neural Network Test

"""clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_valid)
y_pred_pr = clf.predict_proba(x_valid)
print(y_pred_pr)
#compare with y_valid
valid_res = accuracy_score(y_valid, y_pred) * 100
print(valid_res) #66"""



################TEST
#Normalization and Resampling
"""from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_valid = scaler.transform(x_valid)

"""

###################





# example of making multiple probability predictions
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.datasets import make_blobs
# fit final model
"""
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

y_pred=model.predict(x_valid)
y_pred_pr = model.predict_proba(x_valid)

valid_res = accuracy_score(y_valid, y_pred) * 100
print(valid_res) #76


y_test_pr = model.predict_proba(x_test)
fb_df = pd.DataFrame(y_test_pr[:,1])
#df_res = pd.DataFrame(new_res, columns=['genres'])

fb_df.to_clipboard(sep=',', index=False)"""

################LR with SGD##############################
clf = SGDClassifier(loss="squared_loss", penalty="l2", max_iter=5000)

from sklearn.pipeline import make_pipeline
est = make_pipeline(StandardScaler(), SGDClassifier(loss="log", penalty="l2", alpha=0.001, shuffle=True))
est.fit(x_train,y_train)
y_pred = est.predict(x_valid)
valid_res = accuracy_score(y_valid, y_pred) * 100
print(valid_res) #


y_test_pr = est.predict_proba(x_test)
fb_df = pd.DataFrame(y_test_pr[:,1])
#df_res = pd.DataFrame(new_res, columns=['genres'])

fb_df.to_clipboard(sep=',', index=False)


#clf = SGDRegressor(loss='squared_loss', penalty='l2')
"""clf.fit(x_train, y_train)

y_pred = clf.predict(x_valid)

valid_res = accuracy_score(y_valid, y_pred) * 100
print(valid_res) #"""


##########################################################
# show the inputs and predicted probabilities
"""for i in range(len(x_train)):
	print("X=%s, Predicted=%s" % (x_train[i], y_pred[i]))"""


###parameter tuning
#https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
#from sklearn.preprocessing import LabelEncoder
#label_encoder = LabelEncoder()


