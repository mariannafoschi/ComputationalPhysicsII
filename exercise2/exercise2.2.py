# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:16:15 2020

@author: Davide
"""

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


#%% Testing general matrix operation
A = np.array([[1,2],[3,5]])
B = np.array([[4,2],[1,1]])
k = 3
print(k*A)
print(np.transpose(A))
print(A+B)
print(A.dot(B))
print(B.dot(A))
        
    
#%% Testing the generalized eigenvalue problem

N = 1000
eig_our = np.zeros(N)
eig_routine = np.zeros(N)
j=0
while j<N:
    H = np.random.rand(3,3)
    H = (H + H.T)/2
    S = np.random.rand(3,3)
    S = (S + S.T)/2
    temp_check = linalg.eigh(S, eigvals_only=True)                #check if H is positive 
    temp_eig, U = linalg.eigh(S)                #define U
    if all(temp_eig>0) and all(temp_check>0):
        S12 = np.zeros((3,3))                       #define S_1/2
        for i in range(3):
            S12[i, i] = 1/np.sqrt(temp_eig[i])
        V = np.dot(U, S12)                          #define V
        Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
        eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
        C = V.dot(Cprime)
        eig_our[j] = eig[0]
        eig_routine[j] = linalg.eigh(H,S, eigvals_only = True)[0]
        j+=1

#plt.hist((eig_our-eig_routine)/eig_routine, bins="auto")
plt.figure()
ax = plt.gca()
ax.grid(True)
ax.set_yscale("log")
p, = plt.plot((abs((eig_our-eig_routine)/eig_routine)))
p.set_label("Energy relative error")
ax.legend()
plt.xlabel("Attempt number")
plt.ylabel("Energy relative error")

    
