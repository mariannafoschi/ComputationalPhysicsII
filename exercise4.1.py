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
def build_density(potential, mesh,N_e):
    density = np.zeros(N)
    sum_eig = 0
    E = np.zeros((n_states,L_max+1))
    wf = np.zeros((N,n_states,L_max+1))

    for l in range(L_max+1):
        E[:,l],wf[:,:,l] = solve_eq(potential,mesh,l,n_states)

    E_sort = np.zeros(((L_max+1)*n_states,3))
    for i in range(L_max+1):
        E_sort[(i*n_states):((i+1)*n_states),0] = E[:,i]
        E_sort[(i*n_states):((i+1)*n_states),1] = i*np.ones(n_states)
        E_sort[(i*n_states):((i+1)*n_states),2] = np.array(range(n_states))

    ind_sort = np.argsort(E_sort[:,0])
    E_temp = E_sort
    E_sort = E_temp[ind_sort,:]

    #fill electrons
    fill=0
    k = 0
    while fill<N_e:
        l = int(E_sort[k,1])
        n = int(E_sort[k,2])
        density = density + 2*(2*l+1)*(wf[:,n,l]/mesh)**2 /(4*np.pi)
        sum_eig = sum_eig + 2*(2*l+1)*E[n,l]
        fill = fill + 2*(2*l+1)
        k=k+1
    warn =0
    if fill>N_e:
        print('WARNING')
        print('shell not closed: need ' + str(fill-N_e)+ ' electrons to fill the state')
        warn=1
        density = density - (fill-N_e)*(wf[:,n,l]/mesh)**2 /(4*np.pi)
        sum_eig = sum_eig - (fill-N_e)*E[n,l]
        print(n)
        print(l)
        
    density[N-1] = density[N-2]
        
    return density, sum_eig , 

def V_ext(mesh,r_s,Ne):
    # input: mesh
    # output: vext potential term vector
    rho_b = 3/4/np.pi /r_s**3
    Rc = Ne**(1/3)*r_s
    vext = np.zeros(N)
    for i in range(N):
        if mesh[i]<Rc:
            vext[i]= 2*np.pi*rho_b*(1/3*mesh[i]**2-Rc**2)
        else:
            vext[i]= -4*np.pi*rho_b/3*Rc**3/mesh[i]
            
    return vext

#%% MAIN
#wiegner radius and othe parameters
r_s = 3.93
N_e = np.array([2,8,20,40])
rho_b = 3/4/np.pi /r_s**3
R_c = np.array([N_e[i]**(1/3)*r_s for i in range(4)])
L_max = 4
n_states = 3
#mesh
N = 10**5
r_max= 3*R_c
h = r_max/N
r = np.array([np.array(range(N))*h[i]+h[i] for i in range(4)])
print(r[0,:])
density_1 = np.zeros((N,4))
E_1 = np.zeros(4)
plt.figure()
for i in range(4):
    print(i)
    density_1[:,i],E_1[i], warn = build_density(V_ext(r[i,:],r_s,N_e[i]), r[i,:],N_e[i])  
    plt.plot(r[i],density_1[:,i]/rho_b,label='N_e ='+str(N_e[i]))
plt.axis([-1,15,-1,17.5])
plt.legend()
plt.xlabel('radial distance [a_0]')
plt.ylabel('normalized density $\rho$ /\rho_b')
plt.title('Na - non interacting electrons density')
plt.grid(True)
plt.show()
print(E_1)
#%% MAIN￼
#wiegner radius and othe parameters
r_s = 4.86
N_e = np.array([2,8,20,40])
rho_b = 3/4/np.pi /r_s**3
R_c = np.array([N_e[i]**(1/3)*r_s for i in range(4)])
L_max = 4
n_states = 3
#mesh
N = 10**5
r_max= 3*R_c
h = r_max/N
r = np.array([np.array(range(N))*h[i]+h[i] for i in range(4)])
print(r[0,:])
density_2 = np.zeros((N,4))
E_2 = np.zeros(4)
plt.figure()
for i in range(4):
    print(i)
    density_2[:,i],E_2[i], warn = build_density(V_ext(r[i,:],r_s,N_e[i]), r[i,:],N_e[i])  
    plt.plot(r[i],density_2[:,i]/rho_b,label='N_e ='+str(N_e[i]))
plt.axis([-1,15,-1,17.5])
plt.legend()
plt.xlabel("radial distance [a_0]")
plt.ylabel("normalized density $\rho$ /\rho_b")
plt.title('K - non interacting electrons density')
plt.grid(True)
plt.show()
print(E_2)
#%%￼