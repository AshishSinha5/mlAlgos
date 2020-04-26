# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:49:13 2020

@author: ashish
"""

# Importing Required Packages
import numpy as np
import numpy.linalg as la

def eigenspace_update(B, N, epsilon):
    ''' Takes matrix B and N (no. of image vectors) as input and outputs the updated SVD of B'''
    A1 = B[:, 0]
    U1 = A1/la.norm(A1)
    U = np.reshape(U1, (len(U1), 1))
    V = [1]
    Sigma = [la.norm(A1)]
    print(np.shape(Sigma))
    
    for i in range(1,N):
        #Sigma = np.diag(Sigma) 
        Vt = np.transpose(V) 
        #print('enter loop', i)
        #svd = U.dot(Sigma.dot(Vt))
        Ai = B[:, i]
        #print('enter loop', i)
        #M = np.c_[svd, Ai]     #M = Dummy matrix = [U.Sigma.V Ai]
        #Ui, Sigmai, Vi = la.svd(M)     # Here Sigmai is an array of singular values arranged in descending order
        Ui, Sigmai, Vi = svd_update(Ai, U, Sigma, V)
        print(np.shape(Sigmai))
        for j in range(len(Sigmai)):
            if Sigmai[j] <= epsilon:
                #print("Here")
                break
        U = Ui[:, :j]
        Sigma = Sigmai[:j]
        V = Vi[:, :j]
    return U, Sigma, V



def svd_update(Ai, U, Sigma, V):
    l = len(Ai)
    Ut = np.transpose(U)
    x = Ut.dot(Ai)
    a_perp = Ai - U.dot(x)
    #print('enter subfunc', 1)
    
    a_perp = a_perp/la.norm(a_perp)
    a_perp = np.reshape(a_perp, (len(a_perp), 1))
    #print('enter subfunc', 1)
    
    Q = np.c_[Sigma, Ut.dot(Ai)]    
    lastrowQ = np.zeros((len(Sigma),1))
    #print('enter subfunc', 1)
    
    a_perp_t = np.transpose(a_perp)
    X = a_perp_t.dot(Ai)
    #print('enter subfunc', 1)
    
    lastrowQ = np.c_[lastrowQ, X]
    Q = np.r_[Q, lastrowQ]
    Ui, Sigmai, Vi = la.svd(Q)       # Have to use SVD Calculation of Broken Arrowhead Matrix
    #print('enter subfunc', 1)
    
    Ui_extra = np.c_[U, a_perp]      # Dummy variable for intermediate step
    Ui = Ui_extra.dot(Ui)
    #print('enter subfunc', 1.1)
    #print(len(V), type(V))
    
    if len(V) > 1:
        shape = V.shape
        v = np.c_[V, np.zeros(len(V)).reshape(len(V),1)]
        #print('a', v, V, len(V), V.shape)
    else:
        #print(V)
        v = np.c_[V, 0]
        #print('b',v,V)
    #print('enter subfunc', 1.2)
    
    #lastrow_v = np.array([np.append(np.zeros(len(V)), [1])])
    #print(lastrow_v, lastrow_v.shape, len(v), v.shape)
    #lastrow_v.reshape(1,2)
    #print(lastrow_v, lastrow_v.shape)
    
    if len(V) > 1:
        Vshape = V.shape 
        lastrow_v = np.r_[np.zeros(Vshape[1]), 1]
        #print('c2',v, lastrow_v)
    else:
        lastrow_v = np.r_[np.zeros(len(V)), 1]
        #print('c1',v, lastrow_v)
        #print('enter subfunc', 1.2)
    
    v = np.r_['0,2',v, lastrow_v]
    #print('d',v)
    
    Vi = v.dot(Vi)
    #print('enter subfunc', 1)
    
    return Ui, Sigmai, Vi


import timeit

start = timeit.default_timer()
from datetime import datetime

# current date and time
begin = datetime.now()

Btest = np.array([[1,2,3,4,5], [3,6,7,5,6], [2,4,6,8,9], [7,8,4,1,4], [3,7,6,1,2]])
N = 5
k = 0.00001
print(eigenspace_update(Btest, N, k))
print(Btest.shape)

stop = timeit.default_timer()
end = datetime.now()
print('Start Time: ', begin,
      'Stop Time: ', end,
      'Time Taken: ', stop - start) 