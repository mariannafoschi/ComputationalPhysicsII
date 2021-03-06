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
N_E = 1                                    # number of energy steps
h_E = 0.65*epsilon/(N_E)                        # energy step (to get just a bit below E_max = 5.9 meV)
E_range = (np.array(range(N_E))+1)*h_E          # array with energy values
crossect_range = np.zeros(N_E)                  # initialize a vector where we write the cross sections
h = 0.001                                       # step
r_low = sigma*0.5
r_max = sigma*8                # lowest mesh point
N = int((r_max-r_low)/h)                            # number of steps (just a bit more than those needed for r=sigma*7)
x = (np.array(range(N)))*h + r_low            # define the mesh (note: first point is r_low)
prefac = (1.05*10**(-34))**2 / (2*1.67*10**(-27)) /(3.18*10**(-10))**2 /(5.9*1.6*10**(-22))      # prefactor hbar^2/2m last two factor to trasform Jm^2 dimensions
prefac = 1/prefac                               # prefactor 2m/hbar^2 
b = (4*epsilon*sigma**(12)*prefac/25)**(1/10)   # parametro per le bc
ell_max = 6                          # number of values for the angular momentum
phase = np.zeros([ell_max+1,5])                     # initialize a vector where we write the phase shifts
crossect_partial = np.zeros([ell_max+1,5])   
wf= np.zeros([ell_max+1,N])
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

k = np.sqrt(prefac*E)                               # calculate wave vector from energy

for j in range(5):

    ell=0
    r1_ix = int(sigma*(5-r_low)/h)                          # index in position array of r1 > r_max
    r2_ix = int(sigma*(5.5-j/10-r_low)/h)                          # index in position array of r2 > r_max

    # Calculate phase shifts
    for ell in range(ell_max+1):                    # cycle over the values of angular momenta
        y = numerov_LJ(E,ell,x,h)                   # calculate w.f. for a specific ell
        kappa = (y[r1_ix]*x[r2_ix])/(y[r2_ix]*x[r1_ix])
        phase[ell,j] = np.arctan((kappa*bess(ell,(k*x[r2_ix]))-bess(ell,(k*x[r1_ix])))/(kappa*neumm(ell,(k*x[r2_ix]))-neumm(ell,(k*x[r1_ix]))))
        crossect_partial[ell,j]= (4*pi/(prefac*E))*(2*ell+1)*np.sin(phase[ell,j])**2
        norm = (np.dot(y[1:(N-1)],y[1:(N-1)]) + (y[0]**2 +y[N-1]**2)/2)*h
        y =  y/np.sqrt(norm) 
    
        wf[ell,:]=y
    

    #plt.plot(range(ell_max+1),crossect_partial)
    #plt.show()
    # Print phase shifts
    plt.plot(np.array(range(ell_max+1)),phase[:,j])
plt.show()
