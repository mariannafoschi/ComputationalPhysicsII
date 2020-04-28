#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.special as sp
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors
import time

#start = time.time()
#print("hello")
#end = time.time()
#print(end - start)

# =============================================================================
# UNITS: the energy is expressed in hbar*omega, while a is expressed in terms of a_HO
# =============================================================================


#%% Define the functions

def numerov(energy,potential,r): 
    
    phi = np.ones(len(r))
    step = r[1]-r[0]
    K2 = 2*energy-2*potential #evaluate the k^2 parameter   
    i=2
    phi[0] =step #initial point, at infinity y=0
    phi[1] =2*step #second point, correct derivative to be set with normalization
    while i<len(r):
        phi[i]= (2*phi[i-1]*(1-5/12*step**2 *K2[i-1])-phi[i-2]*(1+step**2 /12*K2[i-2]))/(1+step**2/12*K2[i])
        i += 1 
    
    return phi

def solve_GP(potential, r, algorithm ):
    #input: -potential: starts from r=step
    #       -r: is the mesh of length N and should starts starts from step
    #       -algotithm: can be 'numerov' or 'fd_method'
    #output:-eigenvalue: ground state eigenvalue
    #       -wavefunction: ground state radial wavefunction normalized to 1, length of the input mesh r 
    #performance: finite difference method works faster for mash with less than ~300 points
    
    #some useful quantuty

    step = r[1]-r[0]
    N= len(r)
    #for big mesh the finite difference method is really slow

        
    #solution using numerov
    if algorithm == 'numerov':
    
        #initialize eigenenergy mu 
        mu = 0;
        mu_step = 0.1;
        
        #to check if you find the ground state
        check = 0
        
        #zeroth step 
        phi=numerov(mu,potential,r)
        control_0= np.sign(phi[N-1])
        
        #look for the groun state
        while check==0:
            #increase energy
            mu = mu+mu_step
            phi=numerov(mu,potential,r)
            
            #check the divergence
            control_1 = np.sign(phi[N-1])
            
            #did the function cross 0 while increaseing the energy? 
            if control_0 != control_1:
                #you found the ground state                       
                check=1
                
                #initialize variables for tangent method
                mu_0 = mu-mu_step 
                phi_0 = numerov(mu_0,potential,r)
                mu_1 = mu
                phi_1 = phi
                delta = 1;
                acc = 10**-10
                
                #tangent method
               
                # to count the number of iterations of the tangent method
                counter = 0
                
                while delta > acc and counter < 25:
                    counter = counter + 1
                
                    mu_2 = mu_0 - phi_0[N-1]*(mu_1-mu_0)/(phi_1[N-1]-phi_0[N-1])
                    phi_2 = numerov(mu_2,potential,r)
                
                    control_2 = np.sign(phi_2[N-1])
                
                    if control_2 == control_1:
                        mu_1 = mu_2;
                        phi_1=phi_2
                        delta = mu_1-mu_0
                    else:
                        mu_0 = mu_2
                        phi_0 =phi_2
                        delta = mu_1-mu_0
                
                # bisection method (if needed)
                while delta > acc:
                    
                    mu_2 = (mu_0 + mu_1)/2
                    phi_2 = numerov(mu_2,potential,r)
                    
                    control_2 = np.sign(phi_2[N-1])
                    
                    if control_2 == control_1:
                        mu_1 = mu_2;
                        phi_1=phi_2
                        delta = mu_1-mu_0
                    else:
                        mu_0 = mu_2
                        phi_0 =phi_2
                        delta = mu_1-mu_0     
                        
                #write down the correct ground state
                mu = mu_2               
                phi = phi_2
                norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
                phi= phi/np.sqrt(norm)
            
            #Didn't find the ground state?
            else:    
                control_0 = control_1
            #Repeat
        

        #return the ground state
        return mu, phi 
   
    #finite difference method     
    if algorithm== 'fd_method':
        ## WARNING with N>1000 ci mette troppo
        
        U = potential[0:N-1] + np.ones(N-1)/h**2
        #matrix definition
        #A = np.diagflat(-0.5*np.ones(N-2)/h**2,1) +np.diagflat(-0.5*np.ones(N-2)/h**2,-1) +np.diagflat(U[0:N-1])
        
        #solve eigenvalue problem        
        eigenvalues,eigenvectors=linalg.eigh_tridiagonal(U,-0.5*np.ones(N-2)/h**2, select = "i", select_range = (0,0))
        
        #write down the correct ground state
        mu = eigenvalues[0]
        phi = np.append(eigenvectors[:,0] ,[0])
        norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
        phi= -phi/np.sqrt(norm)
        #phi =np.ones(N)
        #returns the ground state
        return mu, phi
    


def calc_energy(r, phi, Na):
    step = r[1]-r[0]
    N= len(r)
    
    #cinetic energy
    der2_phi = (45*phi[0:N-5] - 154*phi[1:N-4] + 214*phi[2:N-3] - 156*phi[3:N-2] + 61*phi[4:N-1] - 10*phi[5:N])/(12*step**2) #error of O(step) because non centered derivative
    energy_cin = - 1/2 * (np.dot(phi[1:(N-6)],der2_phi[1:(N-6)]) + (phi[0]*der2_phi[0] +phi[N-6]*der2_phi[N-6])/2)*step
    
    #external potential energy
    energy_ext = 1/2 * (np.dot(phi[1:(N-1)]*r[1:(N-1)],phi[1:(N-1)]*r[1:(N-1)]) + ((phi[0]*r[0])**2 + (phi[N-1]*r[N-1])**2)/2)*step
    
    #interaction potenetia energy
    energy_int = Na/2 * (np.dot(phi[1:(N-1)]**2/r[1:(N-1)],phi[1:(N-1)]**2/r[1:(N-1)]) + ((phi[0]**2/r[0])**2 + (phi[N-1]**2/r[N-1])**2)/2)*step
    
    return (energy_cin + energy_ext + energy_int), energy_int



#%% main
# definition of main variables
pi = np.arctan(1)*4
r_max = 7
N = 1000
h = r_max/N
Na = [-0.6,-0.59,-0.58,-0.57,-0.56,-0.55,-0.5,-0.4,-0.3,-0.2,-0.1] #[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1] #[0.01, 0.1, 1, 10, 100] # this is N_particles * a
Nmix = 30
alpha_mix = [0.3]

#mash
r = np.array(range(N))*h+h

#self consistency constant parameters
Vext = 0.5 * r**2
phi_guess = r*np.exp(-1/2*r**2)*np.sqrt( 2**3/np.sqrt(4*np.pi) )
error = 10**(-4)
convergence = np.zeros((len(Na), len(alpha_mix)))
end_energy = np.zeros((len(Na), len(alpha_mix)))
end_phi = np.zeros((len(Na), len(alpha_mix), N))


for i in range(len(Na)):
    for j in range(len(alpha_mix)):
        print('')
        print(i,j)
        #initial potential guess
        Vint = alpha_mix[j]*Na[i]*(phi_guess/r)**2
        V = Vext + Vint

        #self consistency first step
        mu_final, phi_final = solve_GP(V,r,'fd_method')  
        energy, energy_int = calc_energy(r, phi_final, Na[i])
        difference = (energy - (mu_final - energy_int))
        cont = 1
        
        while abs(difference) > error:
            if cont > 2000:
                difference = 0 
                
            else:                
                Vint = alpha_mix[j]*Na[i]*phi_final**2/(r**2) + (1-alpha_mix[j])*(V-Vext)
                V = Vext + Vint
                mu_final, phi_final = solve_GP(V,r,'fd_method')
                energy, energy_int = calc_energy(r, phi_final, Na[i])
                difference = (energy - (mu_final - energy_int))
                cont +=1
                #phi_archive[cont-1,:] = phi_final
                
        end_energy[i,j]= energy
        end_phi[i,j,:]=phi_final
        convergence[i,j] = cont
        print('')
        print(i,j)
        print(mu_final)
        print(cont)
        

# %%
#plt.figure()
#for i in range(len(Na)):
#    plt.plot(alpha_mix, convergence[i,:], color = [0,1-i/len(Na),1],label = "Na = " + str(10**(i-2)))
#plt.legend()
#plt.xlabel('alpha')
#plt.ylabel('iteration')
#plt.grid(True)     
        

# plot potentials # ((grafico no log visibile))
plt.figure()
for i in range(len(Na)):
    Vint = Na[i]*end_phi[i,0,:]**2/(r**2)
    V = Vext + Vint   
    #plt.plot(r, Vint,linestyle = '--',color =[0,1-i/len(Na),1] )
    ii = 0
    if i < 3:
        ii = 1
    plt.plot(r,V, label = 'Na ='+ str(Na[i]),color = [0+ii,1-i/len(Na)+ii*i/len(Na)-ii,1-ii])
    plt.legend()
    plt.xlabel('position')
    plt.ylabel('potential')
    plt.axis([-0.5,7,-6,5])
    plt.grid(True)
plt.plot(r, Vext, label="Harmonic potential",linestyle = '--',color = 'k')
    
    
# plot energies
plt.figure()
plt.plot(Na[0:],end_energy[0:,0]-1.5,label= 'ground state energy corrections' ,color='b',marker='.',markersize='10',markeredgecolor='b')
plt.legend()
plt.xlabel('Na')
plt.ylabel('energy')
plt.axis([-0.8,0,-0.5,0])
plt.grid(True)


# plot densities
plt.figure()
plt.plot(r, ((phi_guess/r)**2)/(4*pi), label="non-interacting",color = 'k',linestyle ='--')
for i in range(len(Na)):
    ii = 0
    if i < 3:
        ii = 1
    plt.plot(r, ((end_phi[i,0,:]/r)**2)/(4*pi),color =[0+ii,1-i/len(Na)+ii*i/len(Na)-ii,1-ii],label = 'Na='+ str(Na[i]))
    plt.legend()
    plt.xlabel('position')
    plt.ylabel('density')
    plt.axis([-0.5,4,0,1])
    plt.grid(True)


# plot density at r=0
plt.figure()
plt.plot(Na,np.abs(((end_phi[:,0,0]/r[0])**2)/(4*pi)),label= 'central density corrections' ,color='b',marker='.',markersize='10',markeredgecolor='b')
plt.plot(Na,np.ones(len(Na))*((phi_guess[0]/r[0])**2)/(4*pi),label= 'central density for non-interacting case' ,color='k',linestyle='--')
plt.legend()
plt.xlabel('Na')
plt.ylabel('density')
plt.axis([-0.8,0,0.001,2.0])
plt.grid(True)

#plt.plot(r, phi_final, label = "Mean field potential")
#plt.plot(r,V, label="Total potential")
#plt.legend()
#plt.grid(True)
#
##plot convergence of energy
#plt.figure()
#plt.semilogy(np.arange(len(energy_archive))+1,abs(energy_archive-energy_archive[-1]), label="Energy of single particle")
#plt.legend()
#plt.grid(True)

#%%
#plt.figure()
#for i in range(len(energy_archive)):
#    plt.plot(r, phi_archive[i,:]/r)



