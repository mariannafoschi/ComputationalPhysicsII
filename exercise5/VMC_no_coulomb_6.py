# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:22:35 2020

@author: Davide
"""

#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import mf_functions as my
import global_variables as gv

#%% FUNCTION DEFINITION








#%% MAIN PARAMETERS DEFINITION
temp_omega = 1
temp_N_up = 3
temp_N_down = 3
gv.initialize_variables(temp_omega, temp_N_up, temp_N_down)

r_init = np.random.rand(2, gv.num)     # initial position NOTE: FIRST 2 PARTICLES MUST BE IN DIFFERENT POSITIONS OTHERWISE DENSITY IS ZERO (E NOI DIVIDIAMO PER LA DENSITà)
delta = 1                  # width of movement
N_s = 10**6                     # number of samples
cut = 10**3
samples, pot_energy, kin_energy, feenberg_energy = my.sampling_function(r_init, delta, N_s, 1, cut)



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
