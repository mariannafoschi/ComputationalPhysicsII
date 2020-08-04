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
        density = density + 2*(2*l+1)*(wf[:,n,l]/mesh)**2 /4/np.pi
        sum_eig = sum_eig + 2*(2*l+1)*E[n,l]
        fill = fill + 2*(2*l+1)
        k=k+1

    if fill>N_e:
        print('WARNING')
        print('shell not closed: need ' + str(fill-N_e)+ ' electrons to fill the state')
        density = density - (fill-N_e)*(wf[:,n,l]/mesh)**2 /4/np.pi
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

<<<<<<< Updated upstream
# CREATE ANSATZ DENSITY IN n_states v ext OUT rho
=======
# Define variables for the different number of shells that we consider
Nee = np.array([8, 20, 40])
densities = np.zeros((N,len(Nee)))
E_fin = np.zeros(len(Nee))
#E_fin2 = np.zeros(len(Nee))
deltaN = np.zeros(len(Nee))
polariz = np.zeros(len(Nee))

# %% CYCLE OVER SHELLS ---> SELF CONSISTENT PROCEDURE 
>>>>>>> Stashed changes
# ----------------------------------------------------------------------------
basic_pot = V_ext(r)
rho, sum_mu = build_density(r,basic_pot)
plt.plot(r,rho)
plt.show() 

<<<<<<< Updated upstream
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
    

=======
# Cycle over the number of shells considered
for j in range(len(Nee)):
    k=0
    N_e = Nee[j]
    rho_b = 3/4/np.pi /r_s**3
    R_c = N_e**(1/3)*r_s
    
    # define initial density
    rho = 1/(1+np.exp(1*(r-R_c)))
    norm = h*sum(rho*r**2)*4*np.pi
    rho = N_e* rho/norm
    
    # initialise energy variables for self consistent procedure
    energy = 0
    energy_previous = 0
    energy_diff = 2*acc
    v_ext = V_ext(r)
    potential_previous = v_ext
    
    # self consistent procedure
    while energy_diff > acc and k<10000:
        
        # calculate total potential (dependent on density)
        potential_new = v_ext + V_int(r, rho) + V_xc( rho)
        #print(V_int(r,rho)[-1]/v_ext[-1])
        plot=0
        if plot ==1:
            plt.plot(r,v_ext,color='b')
            plt.plot(r,V_int(r, rho),color='g')
            plt.plot(r,V_int(r, rho)-V_xc(rho),color='k')
    
            plt.plot(r,V_xc(rho),color='r')
            plt.show()
        
        tot_pot = (1-alpha)*potential_previous + alpha*potential_new
        
        # solve KS equation and calculate density and the sum of the energy eigenvalues mu
        if plot ==1:
            plt.plot(r,rho)
            
        rho, sum_mu,warn = build_density(tot_pot, r)
    
        if plot ==1:
           plt.plot(r,rho)
           plt.show()
        
        # compute energy
        erres = (3/(4*np.pi*rho))**(1/3)
        energy = sum_mu - 0.5*weighted_integ(V_int(r,rho), rho) - weighted_integ(V_xc(rho), rho) + weighted_integ(-3/4*(3/np.pi)**(1/3)*rho**(1/3), rho) + weighted_integ(gamma/(1+beta1*np.sqrt(erres)+beta2*erres), rho)
        energy_diff = np.abs(energy - energy_previous)
        
        # save values of previous iteration
        energy_previous = energy
        potential_previous = tot_pot
        
        k=k+1
        print(k)
        #print(energy)
    print(energy)
    
    E_fin[j] = energy
    densities[:,j] = rho   
    
    # alternative energy
    #e_kin = E_kin(r,tot_pot,j) 
    #E_fin2[j] = e_kin + weighted_integ(v_ext, rho) + 0.5*weighted_integ(V_int(r,rho), rho) + weighted_integ(-3/4*(3/np.pi)**(1/3)*rho**(1/3), rho) + weighted_integ(gamma/(1+beta1*np.sqrt(erres)+beta2*erres), rho)
  
    
    # polarizability
    ii=0
    while r[ii] < R_c:
        ii = ii + 1
    deltaN[j] = 4*np.pi*h*np.dot(r[ii:]**2,rho[ii:]) - 0.5*h*4*np.pi*r[-1]**2*rho[-1]
    polariz[j] = R_c**3*(1+deltaN[j]/N_e)
    
    # Plot del potenziale efficace
    if j>0:
        plt.plot(r,potential_previous,label='N ='+str(Nee[j]))
        plt.legend()
ax = plt.gca()
ax.legend()
plt.ylabel('effective potential [H]')
plt.xlabel('radial distance [a_0]')
plt.grid('True')
plt.show()

# PLOT DENSITY
# %%----------------------------------------------------------------------------   
for i in range(3):
    plt.plot(r,densities[:,i]/rho_b,label='N ='+str(Nee[i]))
    plt.legend()
    #plt.show()
ax = plt.gca()
ax.legend()
plt.ylabel('normalized density ρ/ρ_b')
plt.xlabel('radial distance [a_0]')
plt.grid('True')
plt.show()
#plt.plot(r,tot_pot)
#plt.show()

#%%
R_cc = np.array([Nee[i]**(1/3) for i in range(len(Nee))])*r_s
E_fin = E_fin + 3/5 *np.array(Nee)**2 /R_cc
print("Energy per particle in H: ", E_fin/Nee)
print("Energy per particle in eV: ", E_fin/Nee*27.2)
    
#%%
#E_fin2 = E_fin2 + 3/5 *np.array(Nee)**2 /R_cc
>>>>>>> Stashed changes


