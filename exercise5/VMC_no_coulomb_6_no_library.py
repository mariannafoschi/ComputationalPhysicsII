# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:48:48 2020

@author: Davide
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:22:35 2020

@author: Davide
"""

#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg
import numba
from numba import jit

#%% FUNCTION DEFINITION
""" Mean field part """
@jit(nopython=True)    
def eval_mf_matrix(r, n, levels):
    """ Calculates the matrix with mf functions evaluated in a certain point r """
    temp = np.zeros((n,n))
    a_0 = 1/omega
    # fill the first level (always occupied)
    for i in range(n):
        temp[i,0] = np.exp(-np.sum(r[:,i]**2)/2/a_0)
    # fill other levels
    if n > 1:
        for i in np.arange(n-1)+1:
            if levels[i] == 1:
                for j in range(n):
                    temp[j,i] = r[0,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
            elif levels[i] == 2:
                for j in range(n):
                    temp[j,i] = r[1,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
    return temp.reshape((n,n))

@jit(nopython=True)            
def eval_mf_matrix_grad(r, n, levels):
    """ Calculates the gradient matrices of mf functions evaluated in a certain point r """
    temp_gradx = np.zeros((n,n))
    temp_grady = np.zeros((n,n))
    temp_grad2x = np.zeros((n,n))
    temp_grad2y = np.zeros((n,n))
    a_0 = 1/omega

    # fill the first level (always occupied)
    for i in range(n):
        temp_gradx[i,0] = -r[0,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grady[i,0] = -r[1,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2x[i,0] = 1/a_0*(r[0,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2y[i,0] = 1/a_0*(r[1,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        
    # fill other levels
    for i in np.arange(n-1)+1:
        for j in range(n):
            if levels[i] == 1:
                temp_gradx[j,i] = -a_0 * temp_grad2x[j,0]
                temp_grady[j,i] =  r[0,j] * temp_grady[j,0]
                temp_grad2x[j,i] = (3-r[0,j]**2/a_0) * temp_gradx[j,0]
                temp_grad2y[j,i] = r[0,j] * temp_grad2y[j,0]
            if levels[i] == 2:
                temp_gradx[j,i] = r[1,j] * temp_gradx[j,0]
                temp_grady[j,i] = -a_0 * temp_grad2y[j,0]
                temp_grad2x[j,i] = r[1,j] * temp_grad2x[j,0]
                temp_grad2y[j,i] = (3-r[1,j]**2/a_0) * temp_grady[j,0]
    return temp_gradx, temp_grady, temp_grad2x, temp_grad2y


# EVERYTHING THAT HAS TO DO WITH ENERGY
@jit(nopython=True)    
def kinetic_energy(r, A_up, A_down):
    """ Calculates the local kinetic energy
        Inputs:
            r = position of which you want to calculate the local kinetic energy.
                Each column is a particle
            A_up = matrix of single particle functions for spin-up particles
            A_down = matrix of single particles functions for spin-down particles
    """
    
    # Mean field part (N=2)
    mf_grad = np.zeros((2,num))
    mf_lap = 0
    [Agradx_up, Agrady_up, Agrad2x_up, Agrad2y_up] = eval_mf_matrix_grad(r[:,:N_up], N_up, [0,1,2])
    A_inv_up = np.linalg.inv(A_up)   # inverse of matrix A (useful for calculating gradient)
    [Agradx_down, Agrady_down, Agrad2x_down, Agrad2y_down] = eval_mf_matrix_grad(r[:,N_up:], N_down, [0,1,2])
    A_inv_down = np.linalg.inv(A_down)   # inverse of matrix A (useful for calculating gradient)
    
    # gradient of first N_plus particles (spin up)
    for l in range(N_up):
        for i in range(N_up):
                mf_grad[0,l] = mf_grad[0,l] + Agradx_up[l,i]*A_inv_up[i,l]
                mf_grad[1,l] = mf_grad[1,l] + Agrady_up[l,i]*A_inv_up[i,l]
    # gradient of last N_minus particles (spin down)
    for l in range(N_down):
        for i in range(N_down):
                mf_grad[0,N_up+l] = mf_grad[0,N_up+l] + Agradx_down[l,i]*A_inv_down[i,l]    #NOTA GLI INDICI INVERTITI
                mf_grad[1,N_up+l] = mf_grad[1,N_up+l] + Agrady_down[l,i]*A_inv_down[i,l]
    
    
    # laplacian
    for l in range(N_up):
        for i in range(N_up):
            mf_lap = mf_lap + Agrad2x_up[l,i]*A_inv_up[i,l] + Agrad2y_up[l,i]*A_inv_up[i,l]
    for l in range(N_down):
        for i in range(N_down):
            mf_lap = mf_lap + Agrad2x_down[l,i]*A_inv_down[i,l] + Agrad2y_down[l,i]*A_inv_down[i,l]
        
    kin_en = -1/2*mf_lap
    
    #feenberg energy
    feenberg_en = 0
    for i in range(num):
        feenberg_en = feenberg_en + np.sum(mf_grad[:,i]**2)
    feenberg_en = -1/4*(mf_lap-feenberg_en)
    return kin_en, feenberg_en


@jit(nopython=True)    
def potential_energy(r):
    return 1/2 * omega**2 * np.sum(r**2)
    



@jit(nopython=True)    
def density(r):
    """ Return the probability density of point r.
        Inputs:
            b = parameters of the Jastrow factors
            r = point of which we want the probability density """
    A_up = eval_mf_matrix(r[:,:N_up], N_up, [0,1,2]) 
    A_down = eval_mf_matrix(r[:,N_up:], N_down, [0,1,2])
    #print(np.shape(A_up), np.shape(A_down))
    psi = np.linalg.det(A_up) * np.linalg.det(A_down)
    return psi**2, A_up, A_down

@jit(nopython=True)    
def generate_pos(r, delta, mode):
    new_r = np.zeros(np.shape(r))
    if mode==1:
        new_r = r + (np.random.rand(2, num)-1/2)*delta
#    elif mode==2:
#        return 0
    return new_r

@jit(nopython=True)        
def sampling_function(r, delta, N_s, mode, cut):
    """ This function performs Metropolis algorithm: it samples "num" points from distribution "p" using mode "mode".
        Inputs:
            r = matrix with all initial positions: each column represent a particle [x; y; z]
            b = distribution function to sample (RIVEDI, MI SA CHE BASTANO I PARAMETRI)
            delta = max width of each jump (same along each direction)
            N_s = number of total samples
            mode = 1: moves all particles at each step (better for few particle)
                   2: moves one particle at each step
            cut = number of initial points of the sampling to delete """
    count = 0
    pos = np.zeros((2,num,N_s))
    pot_energy = np.zeros(N_s)
    kin_energy = np.zeros(N_s)
    feenberg_energy = np.zeros(N_s)
    
    prev_density, A_up, A_down = density(r)
    pos[:,:,0] = r
    kin_energy[0], feenberg_energy[0] = kinetic_energy(r, A_up, A_down)
    pot_energy[0] = potential_energy(r)
    count = count + 1
    
    if mode==1:
        n = 1
        while n < N_s:
            if n%10000 == 0:
                print(n/10000)
            pos_temp = generate_pos(pos[:,:,n-1], delta, mode)
            new_density, A_up, A_down = density(pos_temp)
            w = new_density/prev_density   # VEDI COMMENTO QUADERNO, PUO ESSERE IMPORTANTE
            if random.random() <= w:
                pos[:,:,n] = pos_temp
                pot_energy[n] = potential_energy(pos_temp)
                kin_energy[n], feenberg_energy[n] = kinetic_energy(pos_temp, A_up, A_down)
                prev_density = new_density
                count = count + 1
            else:
                pos[:,:,n] = pos[:,:,n-1]
                pot_energy[n] = pot_energy[n-1]
                kin_energy[n] = kin_energy[n-1]
                feenberg_energy[n] = feenberg_energy[n-1]
            n = n + 1
#    elif mode==2:
#        return 0
    print("Accepted steps (%):")
    print(count/N_s*100)
    return pos[:,:,cut:], pot_energy[cut:], kin_energy[cut:], feenberg_energy[cut:]

#%% MAIN PARAMETERS DEFINITION
# for unit conversion at the end
hartree = 11.86 #meV
bohr_radius = 9.793 #nm

omega = 1.000
N_up = 3 # number of particles with spin up
N_down = 3 #number of particles with spin down
num = N_up + N_down

r_init = np.random.rand(2, num)     # initial position NOTE: FIRST 2 PARTICLES MUST BE IN DIFFERENT POSITIONS OTHERWISE DENSITY IS ZERO (E NOI DIVIDIAMO PER LA DENSITà)
delta = 0.8                  # width of movement
N_s = 1000000                     # number of samples
cut = 3000
samples, pot_energy, kin_energy, feenberg_energy = sampling_function(r_init, delta, N_s, 1, cut)



#%% DATA ELABORATION

tot_energy = pot_energy + kin_energy
tot_energy_feenberg = pot_energy + feenberg_energy
fin_energy = sum(tot_energy/(N_s - cut))
fin_energy_feenberg = sum(tot_energy_feenberg/(N_s - cut))
fin_error = np.sqrt(1/(N_s - cut - 1))*np.sqrt(sum(tot_energy**2/(N_s - cut)) - sum(tot_energy/(N_s - cut))**2) # il primo fattore moltiplica tutto vero?
fin_error_feenberg = np.sqrt(1/(N_s - cut - 1))*np.sqrt(sum(tot_energy_feenberg**2/(N_s - cut)) - sum(tot_energy_feenberg/(N_s - cut))**2) # il primo fattore moltiplica tutto vero?
print("Ground state energy +- 3 sigma=\n" + str(fin_energy/omega) + " x omega"+ "+-" + str(3 * fin_error) + "\n")
print("Ground state energy con feenberg +- 3 sigma=\n" + str(fin_energy_feenberg/omega) + " x omega"+ "+-" + str(3 * fin_error_feenberg))

# Non cancellare (da capire)
print(np.sqrt(1/(N_s - cut - 1))*np.sqrt(1/(N_s - cut)*sum(tot_energy**2) - (1/(N_s - cut)*sum(tot_energy))**2)) 
print(np.sqrt(1/(N_s - cut - 1))*np.sqrt(1/(N_s - cut)*sum(tot_energy_feenberg**2) - (1/(N_s - cut)*sum(tot_energy_feenberg))**2))



#%% PLOT
plt.hist(samples[0,0,:], bins=50, density= True)
#xx = (np.arange(1000)-500)/100
#plt.plot(xx, 1/np.sqrt(np.pi)*np.exp(-(xx)**2) + 2*xx**2/np.sqrt(np.pi)*np.exp(-(xx)**2))

#%%
#correlation between kinetic and potential energy
plt.figure()
plt.plot(kin_energy, label="Kinetic energy")
plt.plot(pot_energy, label = "Potential energy")
plt.plot(kin_energy + pot_energy)
plt.legend()

# difference between kinetic and feenberg energy
plt.figure()
plt.plot(kin_energy-feenberg_energy)
