#!/usr/bin/env python
# coding: utf-8

# In[296]:


import os
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# In[236]:


SOURCE_DIR = 'C:\\Users\\aarpi\\Documents\\Coding Projects\\OMSCS\\ML\\classification'
DATA_PATH = "/data"

CREDIT_NAME = "crx.data"
BREAST_CANCER_NAME = "breast-cancer-wisconsin.csv"

CREDIT_COLS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
BREAST_COLS = ["ID", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion", 
               "SingleEpithelialCellSize", "BareNuclei" , "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]


# In[288]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, return_pd = False):
        self.attribute_names = attribute_names
        self.return_pd = return_pd
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.return_pd:
            return X[self.attribute_names]
        else: 
            return X[self.attribute_names].values
        
class MultipleOneHot(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.get_dummies(X).values

class Converter(BaseEstimator, TransformerMixin):
    def __init__(self, src, dst, new_type =  None):
        self.dst = dst
        self.src = src
        self.new_type = new_type
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[X == self.src] = self.dst
        if self.new_type is not None:
            return X.astype(self.new_type)
        else:
            return X

def my_join(list_of_dirs):
    return "/".join(list_of_dirs)[1:]

def load_data(cols, name, path = DATA_PATH):
    csv_path = my_join((path, name))
    print(csv_path)
    return pd.read_csv(csv_path, names=cols, header=None)


# In[326]:


def get_credit_pipeline():

    cat_attributes = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    num_attributes = ['A2', 'A3', 'A8', "A11", 'A14', 'A15']

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes, return_pd=True)),
        ('multiple_one_hot', MultipleOneHot()),
    ])

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('converter', Converter("?", "Nan", new_type="float")),
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ])

    return full_pipeline


# In[327]:


def get_breast_cancer_pipeline():

    num_attributes = ['ClumpThickness', 'UniformityCellSize', 'UniformityCellShape',
           'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei',
           'BlandChromatin', 'NormalNucleoli', 'Mitoses']


    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('converter', Converter("?", "Nan", new_type="float")),
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])
    
    return num_pipeline

"""
credit = load_data(CREDIT_COLS, CREDIT_NAME)
breast = load_data(BREAST_COLS, BREAST_CANCER_NAME)

p = get_credit_pipeline()
p.fit_transform(credit)

p = get_breast_cancer_pipeline()
p.fit_transform(breast)
"""