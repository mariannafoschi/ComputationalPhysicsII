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
temp_L4 = 0

gv.initialize_variables(temp_omega, temp_N_up, temp_N_down,temp_L4)
gv.occ_levels(temp_L4)

r_init = np.random.rand(2, gv.num)     # initial position NOTE: FIRST 2 PARTICLES MUST BE IN DIFFERENT POSITIONS OTHERWISE DENSITY IS ZERO (E NOI DIVIDIAMO PER LA DENSITà)
delta = 1                  # width of movement
N_s = 10**6                     # number of samples
cut = 10**3

b= np.zeros((gv.num,gv.num))
samples, pot_energy, kin_energy, feenberg_energy = my.sampling_function(r_init, delta, N_s, 1, cut,b)



#%% DATA ELABORATION
# calculation of energy and errors
tot_energy = pot_energy + kin_energy
tot_energy_feenberg = pot_energy + feenberg_energy
fin_energy = sum(tot_energy/(N_s - cut)) # se la calcoli così: sum(tot_energy)/(N_s - cut)   viene 10 giusto
fin_energy_feenberg = sum(tot_energy_feenberg/(N_s - cut))
fin_error = np.sqrt(1/(N_s - cut - 1))*np.sqrt(sum(tot_energy**2/(N_s - cut)) - sum(tot_energy/(N_s - cut))**2) # il primo fattore moltiplica tutto vero?
fin_error_feenberg = np.sqrt(1/(N_s - cut - 1))*np.sqrt(sum(tot_energy_feenberg**2/(N_s - cut)) - sum(tot_energy_feenberg/(N_s - cut))**2) # il primo fattore moltiplica tutto vero?
print("Ground state energy +- 3 sigma=\n" + str(fin_energy/gv.omega) + " x omega"+ "+-" + str(3 * fin_error) + "\n")
print("Ground state energy con feenberg +- 3 sigma=\n" + str(fin_energy_feenberg/gv.omega) + " x omega"+ "+-" + str(3 * fin_error_feenberg))

# Calcolo errore in 4 modi uguali, ma riscritti con formule leggermente diverse
print("First attemp: " + str(np.sqrt(1/(N_s - cut - 1))*np.sqrt(sum(tot_energy**2/(N_s - cut)) - sum(tot_energy/(N_s - cut))**2)))
print("Second attemp: " + str(np.sqrt(1/(N_s - cut - 1))*np.sqrt(abs(sum(tot_energy**2/(N_s - cut)) - sum(tot_energy/(N_s - cut))**2))) )  #uguale a sopra, ma con valore assoluto sotto radice
                                                                                                                                            #teoricamente non dovrebbe servire
print("Third attemp (best): " + str(np.sqrt(sum(tot_energy**2/(N_s - cut)/(N_s - cut - 1)) - sum(tot_energy/(N_s - cut)/np.sqrt(N_s - cut - 1))**2   )))
print("Fourth attemp: " + str(np.sqrt(1/(N_s - cut - 1))*np.sqrt(1/(N_s - cut)*sum(tot_energy**2) - (1/(N_s - cut)*sum(tot_energy))**2)))

#no problem with feenberg energy. This is a check
print(3*np.sqrt(1/(N_s - cut - 1))*np.sqrt(1/(N_s - cut)*sum(tot_energy_feenberg**2) - (1/(N_s - cut)*sum(tot_energy_feenberg))**2))



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
