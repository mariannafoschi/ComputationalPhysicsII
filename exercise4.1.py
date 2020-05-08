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


def numerov(energy,potential,spin,r): 
    
    phi = np.ones(len(r))
    step = r[1]-r[0]
    K2 = 2*energy-2*potential #evaluate the k^2 parameter   
    i=2
    phi[0] =(step)**(spin+1) #near zero wf goes like r^l+1
    phi[1] =(2*step)**(spin+1) 
    while i<len(r):
        phi[i]= (2*phi[i-1]*(1-5/12*step**2 *K2[i-1])-phi[i-2]*(1+step**2 /12*K2[i-2]))/(1+step**2/12*K2[i])
        i += 1 
    
    return phi

def solve_eq(potential, r, spin ):
    #input: -potential: starts from r=step, must not include the effective centrifugal one
    #       -r: is the mesh of length N and should starts starts from step
    #       -spin: look for the solution at a certain value of L
    #output:-eigenvalues: array with found eigenvalues in increasing order
    #       -wavefunctions: radial wavefunctions normalized to 1, phi[:,i] is the i-th wavefunction 


    #some useful quantuty
    step = r[1]-r[0]
    h=step
    N= len(r)
    #define output
    eig = [0]
    eigfun = np.zeros((N,1))
    #Add the centrifugal part of the potential
    potential = potential + spin*(spin+1)/2/r**2

    #Energy from which to start to look for eignestates
    E = np.min(potential)
    E_step = 0.2
    #zeroth step 
    phi=numerov(E,potential,spin,r)
    control_0= np.sign(phi[N-1])

    #look for the groun state
    while E<np.max(potential- spin*(spin+1)/2/r**2):
        #increase energy
        E = E+E_step
        phi=numerov(E,potential,spin,r)          
        #check the divergence
        control_1 = np.sign(phi[N-1])
        
        #did the function cross 0 while increaseing the energy? 
        if control_0 != control_1:
            #you found the ground state                       
            check=1
                
            #initialize variables for tangent method
            mu_0 = E-E_step 
            phi_0 = numerov(mu_0,potential,spin,r)
            mu_1 = E
            phi_1 = phi
            delta = 1;
            acc = 10**-12             
            #tangent method
               
            # to count the number of iterations of the tangent method
            counter = 0
            
            while delta > acc and counter < 20:
                counter = counter + 1
            
                mu_2 = mu_0 - phi_0[N-1]*(mu_1-mu_0)/(phi_1[N-1]-phi_0[N-1])
                phi_2 = numerov(mu_2,potential,spin,r)
            
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
                phi_2 = numerov(mu_2,potential,spin,r)
                
                control_2 = np.sign(phi_2[N-1])
                
                if control_2 == control_1:
                    mu_1 = mu_2;
                    phi_1=phi_2
                    delta = mu_1-mu_0
                else:
                    mu_0 = mu_2
                    phi_0 =phi_2
                    delta = mu_1-mu_0     
                        
            #write down the correct state
            eig = np.append(eig,mu_2) 
                          
            phi = phi_2
            norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
            phi= phi/np.sqrt(norm)
            eigfun = np.append(eigfun,np.resize(phi,(N,1)),1)
            
        control_0 = control_1
        #Repeat
        
    E_out = eig[1:]
    phi_out =eigfun[:,1:]
    #return the eigenstates state
    return E_out,phi_out

#%% MAIN
#wiegner radius and othe parameters
r_s = 3.92
N_e = 8
rho_b = 3/4/np.pi /r_s**3
R_c = N_e**(1/3)*r_s


#mesh
N = 10**4
r_max= 3*R_c
h = r_max/N
r = np.array(range(N))*h +h
potential = np.zeros(N)
for i in range(N):
    if r[i]<R_c:
        potential[i]= 2*np.pi*rho_b*(1/3*r[i]**2-R_c**2)
        
    else:
        potential[i]= -4*np.pi*rho_b/3*R_c**3/r[i]
#plt.plot(r,potential+1/r**2)
E_0,phi_0=solve_eq(potential,r,0)      
E_1,phi_1=solve_eq(potential,r,1)      
E_2,phi_2=solve_eq(potential,r,2)      
E_3,phi_3=solve_eq(potential,r,3)      
    
#%%
plt.plot([0, 0,0,0],E_0,marker='.',markersize=10,linestyle='none')
plt.plot([1,1,1],E_1,marker='.',markersize=10,linestyle='none')
plt.plot([2,2,2,2],E_2,marker='.',markersize=10,linestyle='none')
plt.plot([3,3,3],E_3,marker='.',markersize=10,linestyle='none')
plt.show()
plt.plot(r,phi_3[:,2])
pl
E=np.concatenate([E_0,E_1,E_2,E_3])
E=np.sort(E)
plt.plot(E)