# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:30:58 2020

@author: Davide
"""

#%% IMPORT PACKAGES
import numpy as np
import matplotlib as plt
from scipy import linalg

#%% TESTING MATRIX MULTIPLICATION
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(A)
print(B)
C = A.dot(B) #should give [[19,22],[43,50]]
print(C)

#%% SOLVING EIGENPROBLEM WITH ALREADY EXISTING ROUTINE
A = np.array([[1,0],[0,-1]])
B = np.array([[0,1],[1,0]])
print(linalg.eigvals(A))
linalg.eigh(A)


#%% Exercise 2.3

#%% general definitions
pi = np.arctan(1)*4
#%% One gaussian case
#tecnica di minimizzazione

#%% Two gaussian case
alpha1 = 0.1
alpha2 = 0.4
H = np.array([[alpha1**3*alpha1**3*pi**(3/2)/(alpha1+alpha1)**(3/2)+2*pi/(alpha1+alpha1),
               alpha1**3*alpha2**3*pi**(3/2)/(alpha1+alpha2)**(3/2)+2*pi/(alpha1+alpha2)], #fine prima riga
              [alpha1**3*alpha2**3*pi**(3/2)/(alpha1+alpha2)**(3/2)+2*pi/(alpha1+alpha2),
               alpha2**3*alpha2**3*pi**(3/2)/(alpha2+alpha2)**(3/2)+2*pi/(alpha2+alpha2)]])
S = np.array([[2*pi**(3/2)/(alpha1+alpha1)**(3/2),2*pi**(3/2)/(alpha1+alpha2)**(3/2)],
               [2*pi**(3/2)/(alpha2+alpha1)**(3/2),2*pi**(3/2)/(alpha2+alpha2)**(3/2)]])
[exact_eig, exact_vec] = linalg.eigh(H, S, type=1)    #solving with python routine

#Pederiva routine
temp_eig, U = linalg.eigh(S)                #calculate U
# calcolo S^1/2
#Stilde = (U.conj().T).dot( S.dot(U) )       # calculate Stilde so that to calculate S^1/2
#S12 = np.sqrt(np.reciprocal(Stilde, where = Stilde>=1))
S12 = np.array([[1/np.sqrt(temp_eig[0]), 0], [0, 1/np.sqrt(temp_eig[1])]])
V = np.dot(U, S12)
Hprime = np.dot(V.conj().T, np.dot(H, V) )
eig2, Cprime = linalg.eigh(Hprime)          #diagonalize Hprime
C = V.dot(Cprime)











