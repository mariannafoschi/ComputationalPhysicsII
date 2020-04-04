# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:30:58 2020

@author: Davide
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
            H[i,j] = 3*pi**(3/2)*alpha[i]*alpha[j]/(alpha[i]+alpha[j])**(5/2) - 2*pi/(alpha[i]+alpha[j])
    return H

def calculate_S(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = pi**(3/2)/(alpha[i]+alpha[j])**(3/2)
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
    
#%% One gaussian case
# plot of energy as a function of alpha
plt.figure()
x = np.arange(1000)*0.001+0.001
energie= np.zeros(1000)
for i in range(1000):
    energie[i] = generalized_eig(np.array([x[i]]))
plt.plot(x, energie)
plt.plot(x, 3/2*x-2*np.sqrt(2*x/pi))

#automatic minimization
alpha = np.array([0.4])     #initial alpha
sol1D = minimize(generalized_eig, alpha, bounds = [(0.001,None)] )

#plot of solution
alpha1D = np.array([sol1D.x])
eig1D, C1D = generalized_eig(alpha1D, only_eig = 0)

#definition of the mesh
h = 0.003
x = np.arange(2000)*h
N = np.size(x)

#compute the normalized function
final = C1D*np.e**(-alpha1D[0]*x**2)
norm = 4*pi*(np.dot(x[1:(N-1)]*final[1:(N-1)], x[1:(N-1)]*final[1:(N-1)]) + ((x[0]*final[0])**2 +(x[N-1]*final[N-1])**2)/2)*h
final = final/np.sqrt(norm)
#plot
plt.figure()
plt.plot(x, final)
plt.plot(x, 1/np.sqrt(pi)*np.e**(-x))
plt.title("Comparison with analytic solution with 1 gaussian")

#%% Two gaussian case
#automatic minimization
alpha = np.array([0.1, 0.4])        #initial alpha
sol2D = minimize(generalized_eig, alpha, bounds = ((0.001,None), (0.001, None)) )

#plot of solution
alpha2D = sol2D.x
eig2D, C2D = generalized_eig(alpha2D, only_eig = 0)

#definition of the mesh
h = 0.003
x = np.arange(2000)*h
N = np.size(x)

#compute the normalized function
final = C2D[0]*np.e**(-alpha2D[0]*x**2) + C2D[1]*np.e**(-alpha2D[1]*x**2)
norm = 4*pi*(np.dot(x[1:(N-1)]*final[1:(N-1)], x[1:(N-1)]*final[1:(N-1)]) + ((x[0]*final[0])**2 +(x[N-1]*final[N-1])**2)/2)*h
final = final/np.sqrt(norm)
#plot
plt.figure()
plt.plot(x, final)
plt.plot(x, 1/np.sqrt(pi)*np.e**(-x))
plt.title("Comparison with analytic solution with 2 gaussians")

#%% Three gaussian case
#automatic minimization
alpha = np.array([0.1, 1, 2.2])        #initial alpha
sol3D = minimize(generalized_eig, alpha, bounds = ((0.001,None), (0.001, None), (0.001, None)) )

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig(alpha3D, only_eig = 0)

#definition of the mesh
h = 0.003
x = np.arange(2000)*h
N = np.size(x)

#compute the normalized function
final = C3D[0]*np.e**(-alpha3D[0]*x**2) + C3D[1]*np.e**(-alpha3D[1]*x**2) + C3D[2]*np.e**(-alpha3D[2]*x**2)
norm = 4*pi*(np.dot(x[1:(N-1)]*final[1:(N-1)], x[1:(N-1)]*final[1:(N-1)]) + ((x[0]*final[0])**2 +(x[N-1]*final[N-1])**2)/2)*h
final = final/np.sqrt(norm)
#plot
plt.figure()
plt.plot(x, final)
plt.plot(x, 1/np.sqrt(pi)*np.e**(-x))
plt.title("Comparison with analytic solution with 3 gaussians")

#%% STO-3G
alphaSTO3G = np.array([0.109818, 0.405771, 2.22776])
eigSTO3G, CSTO3G = generalized_eig(alphaSTO3G, only_eig = 0)

