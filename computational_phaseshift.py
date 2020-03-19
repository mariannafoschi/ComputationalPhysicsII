# --------------------------------
# ------- Import libraries -------
# --------------------------------
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# ------- Define variables -------
# --------------------------------

pi = np.arctan(1)*4                             # definition of pi
epsilon = 1.0                                   # potential parameter energy
sigma = 1.0                                     # potential parameter radius
E = 0.3*epsilon                                 # energy
N_E = 100                                       # number of energy steps
h_E = 1.65*epsilon/(N_E)                        # energy step (to get just a bit below E_max = 5.9 meV)
E_range = (np.array(range(N_E))+1)*h_E          # array with energy values
crossect_range = np.zeros(N_E)                  # initialize a vector where we write the cross sections
h = 0.001                                       # step
r_low = sigma*0.1                               # lowest mesh point
r1_ix = int(sigma*6/h)                          # index in position array of r1 > r_max
r2_ix = int(sigma*7/h)                          # index in position array of r2 > r_max
N = int(sigma*7.5/h)                            # number of steps (just a bit more than those needed for r=sigma*7)
x = (np.array(range(N)))*h + r_low              # define the mesh (note: first point is r_low)
prefac = 1#/(3.48*10**(-5))                     # prefactor 2m/hbar^2 (deve restare 1 o metto il valore calcolato da Zeno?)
b = (4*epsilon*sigma**(12)*prefac/25)**(1/10)   # parametro per le bc
ell_max = 6                                     # number of values for the angular momentum
phase = np.zeros(ell_max+1)                     # initialize a vector where we write the phase shifts
    
# --------------------------------
# ------- Define functions -------
# --------------------------------

# Function that define the k^2 parameter in numerov for Lennard Jones potential 
def K2_LJ(energy,position,spin): 
    return prefac*(energy - 4*epsilon*((sigma/position)**(12)-(sigma/position)**(6))) - spin*(spin+1)/position**2

# Numerov algorithm set for Lennard Jones, gives a vector as output
def numerov_LJ(energy,spin,position,step): 
    y_p = np.ones(len(position))
    K2 = K2_LJ(energy,position,spin)            # evaluate the k^2 parameter   
    y_p[0] = np.exp(-(b/position[0])**5)        # first point, set 1st boundary condition
    y_p[1] = np.exp(-(b/position[1])**5)        # second point, set 2nd boundary condition
    i=2
    while i<N:
        y_p[i]= (2*y_p[i-1]*(1-5/12*step**2 *K2[i-1])-y_p[i-2]*(1+step**2 /12*K2[i-2]))/(1+step**2/12*K2[i])
        i += 1 
    
    return y_p

# Calulation of Bessel and Neumann function
def bess(ell,point):
    bess1 = np.cos(point)/point
    bess2 = np.sin(point)/point
    jj = 0
    for jj in range(ell) :
        bess3 = bess2*(2*jj+1)/point - bess1
        bess1 = bess2
        bess2 = bess3
    
    return bess2

def neumm(ell,point):
    neumm1 = np.sin(point)/point
    neumm2 = -np.cos(point)/point
    jj = 0
    for jj in range(ell) :
        neumm3 = neumm2*(2*jj+1)/point - neumm1
        neumm1 = neumm2
        neumm2 = neumm3
    
    return neumm2

# --------------------------------
# ------- Actual main code -------
# --------------------------------

# --- Point 6 ---

k = np.sqrt(prefac*E)                           # calculate wave vector from energy
ell=0

# Calculate phase shifts
for ell in range(ell_max+1):                    # cycle over the values of angular momenta
    y = numerov_LJ(E,ell,x,h)                   # calculate w.f. for a specific ell
    kappa = (y[r1_ix]*x[r2_ix])/(y[r2_ix]*x[r1_ix])
    phase[ell] = np.arctan((kappa*bess(ell,(k*x[r2_ix]))-bess(ell,(k*x[r1_ix])))/(kappa*neumm(ell,(k*x[r2_ix]))-neumm(ell,(k*x[r1_ix]))))
    
# Print phase shifts
plt.plot(np.array(range(ell_max+1)),phase)
plt.show()

# --- Point 7 ---

# cycle over different energies 
j = 0
for j in range(N_E):                                      
    k = np.sqrt(prefac*E_range[j])              # calculate wave vector from energy
    ell=0

    # Calculate phase shifts
    for ell in range(ell_max+1):                # cycle over the values of angular momenta
        y = numerov_LJ(E_range[j],ell,x,h)      # calculate w.f. for a specific ell
        kappa = (y[r1_ix]*x[r2_ix])/(y[r2_ix]*x[r1_ix])
        phase[ell] = np.arctan((kappa*bess(ell,(k*x[r2_ix]))-bess(ell,(k*x[r1_ix])))/(kappa*neumm(ell,(k*x[r2_ix]))-neumm(ell,(k*x[r1_ix]))))

    # Calculate cross section
    crossect = 0
    ell = 0                                    
    for ell in range(ell_max+1):
        crossect = crossect + (4*pi/(k**2))*(2*ell+1)*np.sin(phase[ell])**2
    crossect_range[j] = crossect

# Plot cross section as function of energy
plt.figure()
plt.plot(E_range,crossect_range)
plt.show()
