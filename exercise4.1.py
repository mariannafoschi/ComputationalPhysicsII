# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:12:48 2020

@author: zeno1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors
import time


def solve_eq(potential, r, spin, n_states):
    #input: -potential: starts from r=step, must not include the effective centrifugal one
    #       -r: is the mesh of length N and should starts starts from step
    #       -spin: look for the solution at a certain value of L
    #output:-eigenvalues: array with found eigenvalues in increasing order
    #       -wavefunctions: radial wavefunctions normalized to 1, phi[:,i] is the i-th wavefunction 


    #some useful quantuty
    h = r[1]-r[0]
    N= len(r)

    #Add the centrifugal part of the potential
    potential = potential + spin*(spin+1)/2/r**2    
    
    U = potential[0:N-1] + np.ones(N-1)/h**2
        
    #solve eigenvalue problem        
    eigenvalues,eigenvectors=linalg.eigh_tridiagonal(U,-0.5*np.ones(N-2)/h**2,select='i',select_range=(0,n_states-1))
        
    #write down the correct ground state
    mu = eigenvalues
    phi = np.vstack((eigenvectors,np.zeros((1,n_states))))
    
    for i in range(n_states):
        norm = (np.dot(phi[1:(N-1),i],phi[1:(N-1),i]) + (phi[0,i]**2 +phi[N-1,i]**2)/2)*h
        phi[:,i]= phi[:,i]/np.sqrt(norm)
    #phi =np.ones(N)
    #returns the ground state
    return mu, phi


#%% MAIN
#wiegner radius and othe parameters
r_s = 3.93
N_e = 40
rho_b = 3/4/np.pi /r_s**3
R_c = N_e**(1/3)*r_s
L_max = 4
n_states = 3
#mesh
N = 10**5
r_max= 3*R_c
h = r_max/N
r = np.array(range(N))*h +h
potential = np.zeros(N)
E_tot = 0
for i in range(N):
    if r[i]<R_c:
        potential[i]= 2*np.pi*rho_b*(1/3*r[i]**2-R_c**2)
        
    else:
        potential[i]= -4*np.pi*rho_b/3*R_c**3/r[i]
        
E = np.zeros((n_states,L_max+1))
phi = np.zeros((N,n_states,L_max+1))

for l in range(L_max+1):
    E[:,l],phi[:,:,l] = solve_eq(potential,r,l,n_states)

E_sort = np.zeros(((L_max+1)*n_states,3))
for i in range(L_max+1):
    E_sort[(i*n_states):((i+1)*n_states),0] = E[:,i]
    E_sort[(i*n_states):((i+1)*n_states),1] = i*np.ones(n_states)
    E_sort[(i*n_states):((i+1)*n_states),2] = np.array(range(n_states))

ind_sort = np.argsort(E_sort[:,0])
E_temp = E_sort
E_sort = E_temp[ind_sort,:]

#fill electrons
N_e = 40
fill=0
rho = np.zeros(N)
k = 0
while fill<N_e:
    l = int(E_sort[k,1])
    n = int(E_sort[k,2])
    rho = rho + 2*(2*l+1)*(phi[:,n,l]/r)**2 /4*np.pi
    E_tot = E_tot + 2*(2*l+1)*E_sort[k, 0]
    fill = fill + 2*(2*l+1)
    k=k+1

if fill>N_e:
    print('WARNING')
    print('shell not closed: need ' + str(fill-N_e)+ ' electrons to fill the state')
   
    rho = rho - (fill-N_e)*(phi[:,n,l]/r)**2 /4*np.pi
    E_tot = E_tot- (fill-N_e) *E_sort[k, 0]
    
plt.plot(r,rho)
plt.show()

plt.plot(r,potential)
plt.show()
#%%
