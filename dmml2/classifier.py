# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 07:50:26 2020

@author: ashish
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, LabelEncoder,StandardScaler
from sklearn.svm import SVC 

df = pd.read_csv("bank-data/bank-additional-full.csv", sep = ";")


def modify_df(df):
    enc = []
    for i in df.columns:
        if type(df[i][0]) is str:
            enc.append(i)
            enc[-1] = LabelEncoder()
            df[i] = enc[-1].fit_transform(df[i])

df_mod = df.copy()

modify_df(df_mod)

#X = df_mod.iloc[:,:-1].values
#y = df_mod.iloc[:,-1].values
'''
def fit_naive_bayes(df):
    
    clf = MultinomialNB()
    vec = make_union(*[
            make_pipeline(FunctionTransformer(df_mod['cons.conf.idx'],validate = False), 
                          MinMaxScaler(),clf)
        ])
    scores = cross_val_score(vec,df_mod[:,:-1],df_mod[:,-1], cv = 10)
    print(scores.mean())
'''

def fit_decision_tree(df):
    
    clf = DecisionTreeClassifier()
    
    score = cross_val_score(clf,df.iloc[:,:-1],df.iloc[:,-1], cv = 10)
    print(score.mean())
    
def fit_svm(df):
    
    clf = SVC()
    score = cross_val_score(clf,df.iloc[:,:-1],df.iloc[:,-1], cv = 10)
    print(score.mean())
   

#fit_naive_bayes(df_mod)
fit_decision_tree(df_mod)
fit_svm(df_mod)