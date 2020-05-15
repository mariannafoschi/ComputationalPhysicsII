# ============================================================================
# Import packages
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time
# ============================================================================
# Define functions
# ============================================================================

def solve_eq(potential, mesh, spin):
    # input:  - potential: starts from r=step, must not include the effective centrifugal one
    #         - mesh: the mesh of length N and should start from step
    #         - spin: look for the solution at a certain value of L
    # output: - eigenvalues: array with found eigenvalues in increasing order
    #         - wavefunctions: radial wavefunctions normalized to 1, wf[:,i] is the i-th wavefunction 

    # some useful quantity
    h = mesh[1]-mesh[0]
    N= len(mesh)

    # add the centrifugal part of the potential
    potential = potential + spin*(spin+1)/2/mesh**2    
    
    U = potential[0:N-1] + np.ones(N-1)/h**2
        
    # solve eigenvalue problem        
    eigenvalues,eigenvectors=linalg.eigh_tridiagonal(U,-0.5*np.ones(N-2)/h**2,select='i',select_range=(0,n_states-1))
        
    # write down the correct ground state
    wf = np.vstack((eigenvectors,np.zeros((1,n_states))))
    
    for i in range(n_states):
        norm = (np.dot(wf[1:(N-1),i],wf[1:(N-1),i]) + (wf[0,i]**2 +wf[N-1,i]**2)/2)*h
        wf[:,i]= wf[:,i]/np.sqrt(norm)
    
    # returns the ground state
    return eigenvalues, wf

def V_ext(mesh):
    # input: mesh
    # output: vext potential term vector
    vext = np.zeros(N)
    for i in range(N):
        if mesh[i]<R_c:
            vext[i]= 2*np.pi*rho_b*(1/3*mesh[i]**2-R_c**2)
        else:
            vext[i]= -4*np.pi*rho_b/3*R_c**3/mesh[i]
            
    return vext

def V_xc(density):
    v_exchange = -(3/np.pi)**(1/3)*density**(1/3)
    v_correlation = gamma*(1+7/6*beta1*np.sqrt(r_s)+4/3*beta2*r_s)/(1+beta1*np.sqrt(r_s)+beta2*r_s)**2
    vxc= v_exchange + v_correlation
    
    return vxc

def V_int(mesh,density): 
    vint1 = np.zeros(N)
    vint2 = np.zeros(N)
    density = density/4/np.pi/mesh**2
    # trapezoidal method integration
    for i in range(N): 
        vint1[i] = 4*np.pi/mesh[i]*h*(np.dot(mesh[0:i]**2,density[0:i])-0.5*mesh[i]**2*density[i]) # CHECK estremi credo siano giusti ma un check non guasta
        vint2[i] = 4*np.pi*h*(np.dot(mesh[i:],density[i:]) - 0.5*mesh[i]*density[i] - 0.5*mesh[N-1]*density[N-1])
        vint = vint1 + vint2
    
    return vint

def build_density(mesh, potential):
    density = np.zeros(N)
    sum_eig = 0
    E = np.zeros((n_states,L_max+1))
    wf = np.zeros((N,n_states,L_max+1))

    for l in range(L_max+1):
        E[:,l],wf[:,:,l] = solve_eq(potential,mesh,l)

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
        density = density + 2*(2*l+1)*(wf[:,n,l]/mesh)**2 /4*np.pi
        sum_eig = sum_eig + 2*(2*l+1)*E[n,l]
        fill = fill + 2*(2*l+1)
        k=k+1

    if fill>N_e:
        print('WARNING')
        print('shell not closed: need ' + str(fill-N_e)+ ' electrons to fill the state')
        density = density - (fill-N_e)*(wf[:,n,l]/mesh)**2 /4*np.pi
        sum_eig = sum_eig - (fill-N_e)*E[n,l]
        
    return density, sum_eig

def weighted_integ(function, density):
    integral = h*(np.dot(function,density) - 0.5*function[N-1]*density[N-1])
    
    return integral

# ============================================================================
# Main code
# ============================================================================

# DEFINE VARIABLES
# ----------------------------------------------------------------------------
# system parameters
r_s = 3.92
gamma = -0.103756 # che unità di misura è H?????? Hertree?
beta1 = 0.56371
beta2 = 0.27358
N_e = 40
rho_b = 3/4/np.pi /r_s**3
R_c = N_e**(1/3)*r_s
L_max = 4
n_states = 3

# simulation parametes
acc = 10**(-4)
<<<<<<< Updated upstream
alpha = 0.3
=======
alpha = 0.002
>>>>>>> Stashed changes
N = 10**4
r_max= 2.5*R_c
h = r_max/N
r = np.array(range(N))*h +h

# CREATE ANSATZ DENSITY IN n_states v ext OUT rho
# ----------------------------------------------------------------------------
basic_pot = V_ext(r)
rho, sum_mu = build_density(r,basic_pot)
plt.plot(r,rho)
plt.show() 

# %% START SELF CONSISTENT PROCEDURE
# ----------------------------------------------------------------------------
energy = 0
energy_previous = 0
energy_diff = 2*acc

k=0

<<<<<<< Updated upstream
while energy_diff > acc:
    
    # save values of previous iteration
    rho_previous = rho
    energy_previous = energy
=======
while energy_diff > acc and k<10:
    
    # calculate total potential (dependent on density)
    potential_new = v_ext + V_int(r, rho) + V_xc(rho)
    
    plot=1
    if plot==1:
        plt.plot(r,v_ext)
        plt.plot(r,V_int(r,rho))
        plt.plot(r,V_xc(rho))
        plt.plot(r,potential_new)
        plt.show()
    #plt.plot(r,v_ext)
    #plt.plot(r,V_int(r, rho))
    #plt.plot(r,V_xc(rho))
    #plt.show()
    if k == 0:
        potential_previous = potential_new
    tot_pot = (1-alpha)*potential_previous + alpha*potential_new
>>>>>>> Stashed changes
    
    start = time.time()
    # calculate total potential (dependent on density)
    tot_pot = V_ext(r) + V_int(r, rho) + V_xc(rho)
    end = time.time()
    print(end-start)
    start = time.time()
    # solve KS equation and calculate density and the sum of the energy eigenvalues mu
<<<<<<< Updated upstream
    rho, sum_mu = build_density(r,tot_pot)
=======
    rho, sum_mu = build_density(tot_pot, r)

    if plot==1:
        plt.plot(r,rho)
        plt.show()

>>>>>>> Stashed changes
    # compute energy
    start = time.time()
    energy = sum_mu - 0.5*weighted_integ(V_int(r,rho), rho) - weighted_integ(V_xc(rho), rho) + weighted_integ(-3/4*(3/np.pi)**(1/3)*rho**(1/3), rho) + weighted_integ(gamma/(1+beta1*np.sqrt(r)+beta2*r), rho)
    energy_diff = np.abs(energy - energy_previous)
    end = time.time()
    print(end-start)
    print(energy_diff)
    # compute mix between old and new density
    rho = (1-alpha)*rho_previous + alpha*rho
    
    k=k+1
    print(k)

# PLOT DENSITY
<<<<<<< Updated upstream
# ----------------------------------------------------------------------------
plt.plot(r,rho)
plt.show()    
=======
# ----------------------------------------------------------------------------   

# %%


>>>>>>> Stashed changes
    



