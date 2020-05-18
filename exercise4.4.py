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
    vint1 = np.zeros(N)
    vint2 = np.zeros(N)
    
    # trapezoidal method integration
    vint1[0] = 0.5*h*mesh[0]**2*density[0]
    vint2[0] = h*(np.dot(mesh[:],density[:]) - 0.5*mesh[0]*density[0] - 0.5*mesh[N-1]*density[N-1])
    for i in range(N-1):
        vint1[i+1] = vint1[i] + 0.5*h*(mesh[i+1]**2*density[i+1] + mesh[i]**2*density[i])
        vint2[i+1] = vint2[i] - 0.5*h*(mesh[i+1]*density[i+1] + mesh[i]*density[i])
    vint = 4*np.pi*(vint1/mesh + vint2)
    
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
        density = density + 2*(2*l+1)*(wf[:,n,l]/mesh)**2 /(4*np.pi)
        sum_eig = sum_eig + 2*(2*l+1)*E[n,l]
        fill = fill + 2*(2*l+1)
        k=k+1
    warn =0
    if fill>N_e:
        print('WARNING')
        print('shell not closed: need ' + str(fill-N_e)+ ' electrons to fill the state')
        warn=1
        density = density - (fill-N_e)*(wf[:,n,l]/mesh)**2 /(4*np.pi)
        sum_eig = sum_eig - (fill-N_e)*E[n,l]
        print(n)
        print(l)
        
    density[N-1] = density[N-2]
        
    return density, sum_eig , warn

def weighted_integ(function, density):
    integral = 4*np.pi*h*(np.dot(function,density) - 0.5*function[N-1]*density[N-1])
    
    return integral

def E_kin(mesh, potential, n_shells):
    
    kinetic = 0
    integrand = np.zeros(N)
    
    E = np.zeros((n_states,L_max+1))
    wf = np.zeros((N,n_states,L_max+1))
    wf_der2 = np.zeros((N,n_states,L_max+1))
    
    # find eigenfunctions R(r)
    for l in range(L_max+1):
        E[:,l],wf[:,:,l] = solve_eq(potential,mesh,l)
        
    # calculate second derivative of R(r) ?????
    wf_der2[0:N-5,:,:] = (45*wf[0:N-5,:,:] - 154*wf[1:N-4,:,:] + 214*wf[2:N-3,:,:] - 156*wf[3:N-2,:,:] + 61*wf[4:N-1,:,:] - 10*wf[5:N,:,:])/(12*h**2)
    
    # function to be integrated
    n_levels = 1
    for ll in range(n_shells):
        if ll>1:
            n_levels = 0
        for nn in range(n_levels):
            integrand = integrand + 2*(2*ll+1)*wf[:,nn,ll]*(-0.5*wf_der2[:,nn,ll] + 0.5*ll*(ll+1)/(mesh**2)*wf[:,nn,ll])
    
    # integrate function to get kinetic term
    kinetic = h*4*np.pi*np.dot(mesh**2,integrand) - 0.5*h*4*np.pi*(mesh[0]**2*integrand[0] + mesh[-1]**2*integrand[-1])
     
    return kinetic

# ============================================================================
# Main code
# ============================================================================

# DEFINE VARIABLES
# ----------------------------------------------------------------------------
    
# system parameters
r_s = 3.93 # 3.93 or 4.86
gamma = -0.103756 
beta1 = 0.56371
beta2 = 0.27358
rho_b = 3/4/np.pi /r_s**3
L_max = 4
n_states = 4

# simulation parametes
acc = 10**(-7)
alpha = 0.1
N = 10**4
r_max= 3*40**(1/3)*r_s # note: 40 è il valore massimo di Nee.
h = r_max/N
r = np.array(range(N))*h +h

# Define variables for the different number of shells that we consider
Nee = [2,8,20,40]
densities = np.zeros((N,4))
E_fin = np.zeros(4)
E_fin2 = np.zeros(4)
polariz = np.zeros(4)

# %% CYCLE OVER SHELLS ---> SELF CONSISTENT PROCEDURE 
# ----------------------------------------------------------------------------

# Cycle over the number of shells considered
for j in range(4):
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
    e_kin = E_kin(r,tot_pot,j) 
    E_fin2[j] = e_kin + weighted_integ(v_ext, rho) + 0.5*weighted_integ(V_int(r,rho), rho) + weighted_integ(-3/4*(3/np.pi)**(1/3)*rho**(1/3), rho) + weighted_integ(gamma/(1+beta1*np.sqrt(erres)+beta2*erres), rho)
  
    
    # polarizability
    ii=0
    while r[ii] < R_c:
        ii = ii + 1
    deltaN = 4*np.pi*h*np.dot(r[ii:]**2,rho[ii:]) - 0.5*h*rho[-1]
    polariz[j] = R_c**3*(1+deltaN/N_e)
    
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
    plt.plot(r,densities[:,i+1]/rho_b,label='N ='+str(Nee[i+1]))
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


    



