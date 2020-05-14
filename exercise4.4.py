# ============================================================================
# Import packages
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

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
    rs = (3/(4*np.pi*density))**(1/3)
    v_exchange = -(3/np.pi)**(1/3)*density**(1/3)
    v_correlation = gamma*(1+7/6*beta1*np.sqrt(rs)+4/3*beta2*rs)/(1+beta1*np.sqrt(rs)+beta2*rs)**2
    vxc= v_exchange + v_correlation
    
    return vxc

def V_int(mesh,density): 
    int1 = np.zeros(N)
    int2 = np.zeros(N)
    
    # trapezoidal method integration
    int1[0] = 0.5*h*mesh[0]**2*density[0]
    int2[0] = h*(np.dot(mesh[:],density[:]) - 0.5*mesh[0]*density[0] - 0.5*mesh[N-1]*density[N-1])
    for i in range(N-1):
        int1[i+1] = int1[i] + 0.5*h*(mesh[i+1]**2*density[i+1] + mesh[i]**2*density[i])
        int2[i+1] = int2[i] - 0.5*h*(mesh[i+1]*density[i+1] + mesh[i]*density[i])
    vint = 4*np.pi*[int1/mesh + int2]
    
    return vint

def build_density(potential, mesh):
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
        
    density[N-1] = density[N-2]
        
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
r_s = 4.86
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
alpha = 0.2
N = 10**4
r_max= 3*R_c
h = r_max/N
r = np.array(range(N))*h +h

# CREATE ANSATZ DENSITY IN n_states v ext OUT rho
# ----------------------------------------------------------------------------
v_ext = V_ext(r)
rho, sum_mu = build_density(v_ext,r)
#plt.plot(r,rho)
#plt.show() 
#plt.plot(r,v_ext)
#plt.show()

# %% START SELF CONSISTENT PROCEDURE
# ----------------------------------------------------------------------------
energy = 0
energy_previous = 0
energy_diff = 2*acc
potential_previous = v_ext

k=0

while energy_diff > acc and k<100:
    
    # calculate total potential (dependent on density)
    potential_new = v_ext + V_int(r, rho) + V_xc(rho)
    
    #plt.plot(r,v_ext)
    #plt.plot(r,V_int(r, rho))
    #plt.plot(r,V_xc(rho))
    #plt.show()
    
    tot_pot = (1-alpha)*potential_previous + alpha*potential_new
    
    # solve KS equation and calculate density and the sum of the energy eigenvalues mu
    rho, sum_mu = build_density(tot_pot, r)
    
    # compute energy
    energy = sum_mu - 0.5*weighted_integ(V_int(r,rho), rho) - weighted_integ(V_xc(rho), rho) + weighted_integ(-3/4*(3/np.pi)**(1/3)*rho**(1/3), rho) + weighted_integ(gamma/(1+beta1*np.sqrt(r)+beta2*r), rho)
    energy_diff = np.abs(energy - energy_previous)
    
    # save values of previous iteration
    energy_previous = energy
    potential_previous = tot_pot
    
    k=k+1
    print(k)
    
    #plt.plot(r,rho)
    #plt.show()

# PLOT DENSITY
# ----------------------------------------------------------------------------   
plt.plot(r,rho)
plt.show()
plt.plot(r,tot_pot)
plt.show()
# %%


    



