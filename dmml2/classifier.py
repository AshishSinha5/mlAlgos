# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 07:50:26 2020

@author: ashish
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from joblib import dump

df = pd.read_csv("bank-data/bank-additional-full.csv", sep = ";")


enc = []

def modify_df(df):
    for i in df.columns:
        if type(df[i][0]) is str:
            enc.append(i)
            enc[-1] = LabelEncoder()
            df[i] = enc[-1].fit_transform(df[i])

df_mod = df.copy()

modify_df(df_mod)

#X = df_mod.iloc[:,:-1].values
#y = df_mod.iloc[:,-1].values

def fit_naive_bayes(df):
    
    clf = GaussianNB()
    scores = cross_val_score(clf,df_mod.iloc[:,:-1],df_mod.iloc[:,-1], cv = 10)
    dump(clf, 'naive_bayes.joblib')
    print(scores.mean())


def fit_decision_tree(df):
    
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf,df.iloc[:,:-1],df.iloc[:,-1], cv = 10)
    dump(clf, 'decision_tree.joblib')
    print(score.mean())
    
def fit_svm(df):
    
    clf = SVC(kernel = 'linear')
    score = cross_val_score(clf,df.iloc[:,:-1],df.iloc[:,-1], cv = 10) 
    dump(clf, 'svm.joblib')
    print(score.mean())
   

fit_naive_bayes(df_mod)
fit_decision_tree(df_mod)
fit_svm(df_mod)