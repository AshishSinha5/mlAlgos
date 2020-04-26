# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:55:22 2020

@author: ashish
"""

import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn import model_selection
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
import matplotlib.pylab as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve,auc
from numpy import interp
from joblib import dump
import os
import joblib
import timeit
from datetime import datetime

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def preprocess(df):
    df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']] = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']].replace('unknown', np.NaN)
    bank_object_data = df.select_dtypes(include="object")
    bank_non_object_data = df.select_dtypes(exclude="object")
    bank_object_data = DataFrameImputer().fit_transform(bank_object_data)
    label = LabelEncoder()
    bank_object_data = bank_object_data.apply(label.fit_transform)
    bank_final = pd.concat([bank_object_data, bank_non_object_data], axis = 1)
    return(bank_final)
    
df = pd.read_csv('bank-data/bank-additional-full.csv')
bank_final = preprocess(df)

def kfold_output(model): #function for kfold output
    start = timeit.default_timer()
    begin = datetime.now()
    scoring = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']

    kfold = KFold(n_splits=10, random_state=100, shuffle = True)
    results_kfold = model_selection.cross_validate(model, X, Y, scoring=scoring, cv=kfold)
    print("Recall: %0.2f (+/- %0.2f)" % (results_kfold['test_recall'].mean(), results_kfold['test_recall'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (results_kfold['test_precision'].mean(), results_kfold['test_precision'].std()))
    print("F1 Score: %0.2f (+/- %0.2f)" % (results_kfold['test_f1'].mean(), results_kfold['test_f1'].std()))
    print("Accuracy: %0.2f (+/- %0.2f)" % (results_kfold['test_accuracy'].mean(), results_kfold['test_accuracy'].std()))
    print("ROC_AUC: %0.2f (+/- %0.2f)" % (results_kfold['test_roc_auc'].mean(), results_kfold['test_roc_auc'].std()))
    
    print(results_kfold)
    
    print(model.priors)
    stop = timeit.default_timer()
    end = datetime.now()
    print('Start Time: ', begin,
          'Stop Time: ', end,
          'Time Taken: ', stop - start)
    
def classifier_roc(df, classifier, X_train_res, y_train_res):
    cv = KFold(n_splits=10, random_state=100, shuffle = True)
    cv_split_filenames = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10,10))
    i = 1
    file_name = ''
    if type(classifier) == type(DecisionTreeClassifier()):
        file_name = 'DT'
    elif type(classifier) == type(GaussianNB()):
        file_name = 'GNB'
    else:
        file_name = 'SVM'
    for train, test in cv.split(X_train_res, y_train_res):
        probas_ = classifier.fit(X_train_res.iloc[train], y_train_res.iloc[train]).predict_proba(X_train_res.iloc[test])
        
        cv_split_filenames = file_name + str(i)
        cv_split_filenames = os.path.abspath(cv_split_filenames)
        dump(probas_,cv_split_filenames)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_train_res[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
                        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', 
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC of Decision Tree',fontsize=14)
    plt.legend(loc="lower right", prop={'size': 10})
    plt.show()

def plot_saved(df,file_name, X_train_res,y_train_res):
    cv = KFold(n_splits=10, random_state=100, shuffle = True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10,10))
    i = 1
    for train, test in cv.split(X_train_res, y_train_res):
        probas_ = joblib.load(file_name + str(i),mmap_mode = 'c')
         # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_train_res[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)     
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC of Decision Tree',fontsize=14)
    plt.legend(loc="lower right", prop={'size': 10})
    plt.show()


