# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:30:58 2020

@author: Marianna
"""
# we expect an energy of 0.5 H
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize

#%% general definitions
pi = np.arctan(1)*4

#%% definition of useful functions
def calculate_H(alpha):     #to calculate the matrix H_ij
    n = np.size(alpha)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] = - (5*pi**(3/2)*alpha[i]*alpha[j])/(2*(alpha[i]+alpha[j])**(7/2)) - (2*pi)/(3*(alpha[i]+alpha[j])**2)
    return H

def calculate_S(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = pi**(3/2)/(2*(alpha[i]+alpha[j])**(5/2))
    return S

def generalized_eig(alpha, only_eig = 1):
    n = np.size(alpha)
    H = calculate_H(alpha)
    S = calculate_S(alpha)
    temp_eig, U = linalg.eigh(S)                #define U
    S12 = np.zeros((n,n))                       #define S_1/2
    for i in range(n):
        S12[i, i] = 1/np.sqrt(temp_eig[i])
    V = np.dot(U, S12)                          #define V
    Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
    eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
    C = V.dot(Cprime)
    if only_eig:
        return eig[0]
    else:
        return eig[0], C[:,0]


#%% Three p-waves only case
#automatic minimization
alpha = np.array([0.33, 0.079, 0.024])        #initial alpha
sol3D = minimize(generalized_eig, alpha, bounds = ((0.001,None), (0.001, None), (0.001, None)) )

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig(alpha3D, only_eig = 0)

#%% STO-3G
alphaSTO3G = np.array([0.109818, 0.405771, 2.22776])
eigSTO3G, CSTO3G = generalized_eig(alphaSTO3G, only_eig = 0)

