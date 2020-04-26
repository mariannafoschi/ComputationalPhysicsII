# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:40:36 2020

@author: Davide
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:37:09 2020

@author: Davide
"""


#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.special as sp
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors
import time

#start = time.time()
#print("hello")
#end = time.time()
#print(end - start)

# =============================================================================
# UNITS: the energy is expressed in hbar*omega, while a is expressed in terms of a_HO
# =============================================================================


#%% Define the functions

def numerov(energy,potential,r): 
    
    phi = np.ones(len(r))
    step = r[1]-r[0]
    K2 = 2*energy-2*potential #evaluate the k^2 parameter   
    i=2
    phi[0] =step #initial point, at infinity y=0
    phi[1] =2*step #second point, correct derivative to be set with normalization
    while i<len(r):
        phi[i]= (2*phi[i-1]*(1-5/12*step**2 *K2[i-1])-phi[i-2]*(1+step**2 /12*K2[i-2]))/(1+step**2/12*K2[i])
        i += 1 
    
    return phi

def solve_GP(potential, r, algorithm ):
    #input: -potential: starts from r=step
    #       -r: is the mesh of length N and should starts starts from step
    #       -algotithm: can be 'numerov' or 'fd_method'
    #output:-eigenvalue: ground state eigenvalue
    #       -wavefunction: ground state radial wavefunction normalized to 1, length of the input mesh r 
    #performance: finite difference method works faster by almost a factor of 100    
    #some useful quantuty

    step = r[1]-r[0]
    h=step
    N= len(r)
     
    #solution using numerov
    if algorithm == 'numerov':
    
        #initialize eigenenergy mu 
        mu = 0;
        mu_step = 0.1;
        
        #to check if you find the ground state
        check = 0
        
        #zeroth step 
        phi=numerov(mu,potential,r)
        control_0= np.sign(phi[N-1])
        
        #look for the groun state
        while check==0:
            #increase energy
            mu = mu+mu_step
            phi=numerov(mu,potential,r)
            
            #check the divergence
            control_1 = np.sign(phi[N-1])
            
            #did the function cross 0 while increaseing the energy? 
            if control_0 != control_1:
                #you found the ground state                       
                check=1
                
                #initialize variables for tangent method
                mu_0 = mu-mu_step 
                phi_0 = numerov(mu_0,potential,r)
                mu_1 = mu
                phi_1 = phi
                delta = 1;
                acc = 10**-5
                
                #tangent method
                
                # to count the number of iterations of the tangent method
                counter = 0
                
                while delta > acc and counter < 14:
                    counter = counter + 1
                
                    mu_2 = mu_0 - phi_0[N-1]*(mu_1-mu_0)/(phi_1[N-1]-phi_0[N-1])
                    phi_2 = numerov(mu_2,potential,r)
                
                    control_2 = np.sign(phi_2[N-1])
                
                    if control_2 == control_1:
                        mu_1 = mu_2;
                        phi_1=phi_2
                        delta = mu_1-mu_0
                    else:
                        mu_0 = mu_2
                        phi_0 =phi_2
                        delta = mu_1-mu_0
                
                # bisection method (if needed)
                while delta > acc:
                    mu_2 = (mu_0 + mu_1)/2
                    phi_2 = numerov(mu_2,potential,r)
                    
                    control_2 = np.sign(phi_2[N-1])
                    
                    if control_2 == control_1:
                        mu_1 = mu_2;
                        phi_1=phi_2
                        delta = mu_1-mu_0
                    else:
                        mu_0 = mu_2
                        phi_0 =phi_2
                        delta = mu_1-mu_0     
                        
                #write down the correct ground state
                mu = mu_2               
                phi = phi_2
                norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
                phi= phi/np.sqrt(norm)
            
            #Didn't find the ground state?
            else:    
                control_0 = control_1
            #Repeat
        

        #return the ground state
        return mu, phi 
   
    #finite difference method     
    if algorithm== 'fd_method':
        ## WARNING with N>1000 ci mette troppo
        
        U = potential[0:N-1] + np.ones(N-1)/h**2
        #matrix definition
        #A = np.diagflat(-0.5*np.ones(N-2)/h**2,1) +np.diagflat(-0.5*np.ones(N-2)/h**2,-1) +np.diagflat(U[0:N-1])

        #solve eigenvalue problem        
        eigenvalues,eigenvectors=linalg.eigh_tridiagonal(U,-0.5*np.ones(N-2)/h**2,select='i',select_range=(0,0))
        
        #write down the correct ground state
        mu = eigenvalues[0]
        phi = np.append(eigenvectors[:,0] ,[0])
        norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
        phi= -phi/np.sqrt(norm)
        #phi =np.ones(N)
        #returns the ground state
        return mu, phi


def calc_energy(r, phi, Na):
    step = r[1]-r[0]
    N= len(r)
    
    #cinetic energy
    der2_phi = (45*phi[0:N-5] - 154*phi[1:N-4] + 214*phi[2:N-3] - 156*phi[3:N-2] + 61*phi[4:N-1] - 10*phi[5:N])/(12*step**2) #error of O(step) because non centered derivative
    energy_cin = - 1/2 * (np.dot(phi[1:(N-6)],der2_phi[1:(N-6)]) + (phi[0]*der2_phi[0] +phi[N-6]*der2_phi[N-6])/2)*step
     
    #external potential energy
    energy_ext = 1/2 * (np.dot(phi[1:(N-1)]*r[1:(N-1)],phi[1:(N-1)]*r[1:(N-1)]) + ((phi[0]*r[0])**2 + (phi[N-1]*r[N-1])**2)/2)*step
    
    #interaction potenetia energy
    energy_int = Na/2 * (np.dot(phi[1:(N-1)]**2/r[1:(N-1)],phi[1:(N-1)]**2/r[1:(N-1)]) + ((phi[0]**2/r[0])**2 + (phi[N-1]**2/r[N-1])**2)/2)*step
    
    return (energy_cin + energy_ext + energy_int), energy_int



#%% main
# definition of main variables
r_max = 7
N = 8000
h = r_max/N
Na = [0.01, 0.1, 1, 10, 100] # this is N_particles * a
alpha_mix = [0.05, 0.3, 0.6, 0.9]

# variables where to store the energy and solutions after 1 iteration
energy_archive = np.zeros((len(Na), len(alpha_mix)))
phi_archive = np.zeros((len(Na), len(alpha_mix), N))

#mash
r = np.array(range(N))*h+h

for i in range(len(Na)):
    for j in range(len(alpha_mix)):
        #initial potential guess
        Vext = 0.5 * r**2
        phi_guess = r*np.exp(-1/2*r**2)*np.sqrt( 1/np.sqrt(4*np.pi)*2**(3) )
        Vint = alpha_mix[j]*Na[i]*(phi_guess/r)**2
        V0 = Vext + Vint
        #first iteration
        mu_final, phi_final = solve_GP(V0,r,'numerov')  
        energy, energy_int = calc_energy(r, phi_final, Na[i])
        energy_archive[i, j] = energy
        phi_archive[i, j, :] = phi_final


#%%
#plot eigenfunctions
plt.close("all")
#normalization cutting the tail that can be wrong due to numerov
cut_function = np.zeros((len(Na), len(alpha_mix), N))
for i in range(len(Na)):
    for j in range(len(alpha_mix)):
        for k in range(round(4.5/7*N)):
            cut_function[i,j,k] = 1
#no cut for Na = 100
for j in range(len(alpha_mix)):
        for k in range(N):
            cut_function[4,j,k] = 1
phi_archive = phi_archive * cut_function
for i in range(len(Na)):
    for j in range(len(alpha_mix)):
        norm = (np.dot(phi_archive[i,j,1:(N-1)],phi_archive[i,j,1:(N-1)]) + (phi_archive[i,j,0]**2 +phi_archive[i,j,N-1]**2)/2)*h
        phi_archive[i,j,:]= phi_archive[i,j,:]/np.sqrt(norm)

colors = ["#1f77b4ff", "#ff7f0eff", "#2ca02cff", "#d62728ff"]
weights = [1/20, 1/2, 2, 20, 60]
for i in range(len(Na)):
    plt.figure()
    ax_pot = plt.gca()
    ax_pot.plot(r,Vext, "--", label="Harmonic potential", color = "k", linewidth = "1")
    for j in range(len(alpha_mix)):
        V = Vext + alpha_mix[j]*Na[i]*(phi_guess/r)**2
        ax_pot.plot(r, V, "--", color = colors[j], linewidth = "1")
    ax_phi = plt.twinx()
    for j in range(len(alpha_mix)):
        plt.plot(r, phi_archive[i, j, :]/r, label="alpha="+str(alpha_mix[j]), color = colors[j])
    ax_phi.legend()
    ax_pot.grid(True)
    ax_pot.set_xlabel("r in $a_0$ units")
    ax_phi.set_ylabel("Amplitude in units of (1/$a_0)^{3/2}$")
    ax_pot.set_ylabel("Potential [V]")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped






