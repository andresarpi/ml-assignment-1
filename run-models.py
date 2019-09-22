#!/usr/bin/env python
# coding: utf-8

# In[1]:


import clean_datasets as mydata


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score


# # Running the models on the breas cancer dataset

# In[2]:


bc = mydata.load_data(mydata.BREAST_COLS, mydata.BREAST_CANCER_NAME)
pipeline = mydata.get_breast_cancer_pipeline()


# In[3]:


label_binalizer = LabelBinarizer()
X = pipeline.fit_transform(bc)
y = label_binalizer.fit_transform(bc['Class'])


# In[4]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    strat_train_set = X[train_index], y[train_index]
    strat_test_set = X[test_index], y[test_index]


# In[5]:


Xtr, Ytr = strat_train_set
Xts, Yts = strat_test_set


# In[6]:


from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras

from keras import Sequential
from keras.layers import Dense

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from keras.models import model_from_json
from tensorflow import keras
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope


# In[7]:


def save_neural_network_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    model_name = MODEL_DIR + DATASET_NAME + "-" + name
    with open(model_name + "-model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + "-weights.h5")
    history = model.history.history
    with open(model_name + "-history", 'wb') as file_pi:
        pickle.dump(history, file_pi)
        
def load_neural_network(model_name):
    model_name = "neural_network"
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        json_file = open(MODEL_DIR + DATASET_NAME + "-" + model_name + "-model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(MODEL_DIR + DATASET_NAME + "-" + model_name + "-weights.h5")
        print("Loaded model from disk")

    #Load history
    with open(MODEL_DIR + DATASET_NAME + "-" + model_name + "-history", 'rb') as fid:
        loaded_history = pickle.load(fid)
        
    return loaded_model, loaded_history


# In[8]:


import pickle

MODEL_DIR ="models/"
DATASET_NAME = "breast_cancer"

# loading the classifier
estimators =[('decision_tree', 0),
             #('neural_network', best_nn_estimator.model), 
             ('k-nearest-neighbor', 0), 
             ('SVM', 0),
             ('ada-boost-classifier', 0),
            ]
models = []

for model_name, estimator in estimators:
    
    file_name = MODEL_DIR + DATASET_NAME + "-" + model_name + ".pkl"
    print(file_name)
    
    with open(file_name, 'rb') as fid:
        model_loaded = pickle.load(fid)

    models.append((model_name, model_loaded))


# In[9]:


nn, his = load_neural_network("neural_network")
nn_history = his
models.append(("neural_network", nn))


# In[12]:


for model_name, model in models:
    Ypr = model.predict(Xts)
    Ypr = (Ypr > 0.5)
    print(model_name, ": ", accuracy_score(Yts,Ypr)*100, )


# # Running models on credit dataset

# In[13]:


credit = mydata.load_data(mydata.CREDIT_COLS, mydata.CREDIT_NAME)
pipeline = mydata.get_credit_pipeline()


# In[14]:


label_binalizer = LabelBinarizer()
X = pipeline.fit_transform(credit)
y = label_binalizer.fit_transform(credit['A16'])


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    strat_train_set = X[train_index], y[train_index]
    strat_test_set = X[test_index], y[test_index]


# In[16]:


Xtr, Ytr = strat_train_set
Xts, Yts = strat_test_set


# In[19]:


MODEL_DIR ="models/"
DATASET_NAME = "credit"

# loading the classifier
estimators =[('decision_tree', 0),
             #('neural_network', best_nn_estimator.model), 
             ('k-nearest-neighbor', 0), 
             ('SVM', 0),
             ('ada-boost-classifier', 0),
            ]
models = []

for model_name, estimator in estimators:
    
    file_name = MODEL_DIR + DATASET_NAME + "-" + model_name + ".pkl"
    print(file_name)
    
    with open(file_name, 'rb') as fid:
        model_loaded = pickle.load(fid)
        
    models.append((model_name, model_loaded))


# In[20]:


nn, his = load_neural_network("neural_network")
nn_history = his
models.append(("neural_network", nn))


# In[21]:


for model_name, model in models:
    Ypr = model.predict(Xts)
    Ypr = (Ypr > 0.5)
    print(model_name, ": ", accuracy_score(Yts,Ypr)*100, )


# In[ ]:




