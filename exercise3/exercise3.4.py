
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors

#%% general definitions
pi = np.arctan(1)*4

#%% Implementation of finite difference method

#Defining range and number of steps

r_max = 5
N = 1000
h = r_max/N

#mash
r = np.array(range(N))*h

#check with gaussian

#potential
V = 0.5 * r**2
#diag term
U = V[1:N-1] + np.ones(N-2)/h**2

#matrix
A = np.diagflat(-0.5*np.ones(N-3)/h**2,1) +np.diagflat(-0.5*np.ones(N-3)/h**2,-1) +np.diagflat(U)

eigenvalues,eigenvectors= np.linalg.eigh(A)

mu = eigenvalues[0]
phi = np.concatenate((np.array(0.), eigenvectors[:,0] ,np.array(0.)))

print(phi)
#plt.plot(r[1:N-1],-phi/r[1:N-1] )
#plt.plot(r[1:N-1],-np.exp(-r[1:N-1]*r[1:N-1]/2)*phi[0]/r[1])

#%% Prototype of the function

def solve_GP(phi_0, r, potential, algorithm ):
    
    if algorithm == 'numerov':
        phi=phi_0
        mu = 0
        
        return mu, phi 
        
    if algorithm== 'fd_method':
        ## WARNING with N>1000 ci mette troppo
        #step
        h=r[1]-r[0]
        N= np.len(r)
        
        #matrix definition
        A = np.diagflat(-0.5*np.ones(N-3)/h**2,1) +np.diagflat(-0.5*np.ones(N-3)/h**2,-1) +np.diagflat(potential[1:N-1])
        
        #solve eigenvalue problem
        
        eigenvalues,eigenvectors= np.linalg.eigh(A)

        mu = eigenvalues[0]
        phi = np.array([0, eigenvectors[:,0] ,0])
        
        #returns mu and phi defined on the input mesh
        return mu, phi
     
    
    
