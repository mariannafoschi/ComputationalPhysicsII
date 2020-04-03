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
N_E = 1000                                # number of energy steps
h_E = 3.5/5.9*epsilon/(N_E)                     # energy step (to get just a bit below E_max = 5.9 meV)
E_range = (np.array(range(N_E))+1)*h_E          # array with energy values
E_peak = [0.05,0.0795,0.15,0.1,0.2391,0.3,0.3,0.4615,0.6]                  # initialize a vector where we write the cross sections
N_peak =len(E_peak)
l_peak = [4,4,4,5,5,5,6,6,6]
h = 0.001                                       # step
r_low = sigma*0.5
r_max = sigma*7.6             # lowest mesh point
N = 10**4
h = (r_max-r_low)/N                            # number of steps (just a bit more than those needed for r=sigma*7)
x = (np.array(range(N)))*h + r_low            # define the mesh (note: first point is r_low)
prefac = (1.05*10**(-34))**2 / (2*1.67/(1+1.008/83.8)*10**(-27)) /(3.18*10**(-10))**2 /(5.9*1.6*10**(-22))      # prefactor hbar^2/2m last two factor to trasform Jm^2 dimensions
prefac = 1/prefac                               # prefactor 2m/hbar^2 
b = (4*epsilon*sigma**(12)*prefac/25)**(1/10)   # parametro per le bc
ell_max = 10                                # number of values for the angular momentum
phase = np.zeros([N_E,ell_max+1])   
phase_peak = np.zeros([N_peak,ell_max+1])                   # initialize a vector where we write the phase shifts
crossect_partial = np.zeros([N_E,ell_max+1])   
crossect_partial_peak = np.zeros([N_E,ell_max+1])
wf_peak= np.zeros([N,N_peak])
crossect = np.zeros(N_E)                          # initialize a vector where we write the cross sections
r1_ix = int(sigma*(7-r_low)/h)                          # index in position array of r1 > r_max
r2_ix = int(sigma*(7.5-r_low)/h)  


    
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

# --- Point 7 ---

# cycle over different energies 
j = 0
for j in range(N_E):                                      
    k = np.sqrt(prefac*E_range[j])              # calculate wave vector from energy
    ell=0
    print(j)

    # Calculate phase shifts
    for ell in range(ell_max+1):                # cycle over the values of angular momenta
        y = numerov_LJ(E_range[j],ell,x,h)      # calculate w.f. for a specific ell
        kappa = (y[r1_ix]*x[r2_ix])/(y[r2_ix]*x[r1_ix])
        phase[j,ell] = np.arctan((kappa*bess(ell,(k*x[r2_ix]))-bess(ell,(k*x[r1_ix])))/(kappa*neumm(ell,(k*x[r2_ix]))-neumm(ell,(k*x[r1_ix]))))
        crossect_partial[j,ell] = (4*pi/(prefac*E_range[j]))*(2*ell+1)*np.sin(phase[j,ell])**2

    # Calculate cross section
    crossect[j]=sum(crossect_partial[j,:])

# Plot cross section as function of energy
plt.figure()
plt.plot(E_range,crossect)
plt.show()


for j in range(N_peak):                                      
    k = np.sqrt(prefac*E_peak[j])              # calculate wave vector from energy
    ell=0

    wf_peak[:,j] = numerov_LJ(E_peak[j],l_peak[j],x,h)      # calculate w.f. for a specific ell

