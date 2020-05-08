
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)
#%% Implementation of finite difference method

#Defining range and number of steps

r_max = 7
N = 700
h = r_max/N

#mash
r = np.array(range(N))*h+h

V = 0.5* r**2

#%% Prototype of the function

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
    #performance: finite difference method works faster by almost a factor of 100    
    #some useful quantuty

    step = r[1]-r[0]
    h=step
    N= len(r)
     
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
                acc = 10**-5
                
                #tangent method
               
                # to count the number of iterations of the tangent method
                counter = 0
                
                while delta > acc and counter < 20:
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
        eigenvalues,eigenvectors=linalg.eigh_tridiagonal(U,-0.5*np.ones(N-2)/h**2,select='i',select_range=(0,0))
        
        #write down the correct ground state
        mu = eigenvalues[0]
        phi = np.append(eigenvectors[:,0] ,[0])
        norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
        phi= -phi/np.sqrt(norm)
        #phi =np.ones(N)
        #returns the ground state
        return mu, phi
    
    
mu,phi = solve_GP(V,r,'fd_method')   
print(mu)

#%% Measuring efficiency
n = [100,300,600,750,1000,2000,5000,10000]
timing_numerov = np.zeros(len(n))
timing_fd = np.zeros(len(n))
acc_numerov = np.zeros(len(n))
acc_fd = np.zeros(len(n))
#Numerov
for i in range(len(n)):
    r_max = 7
    N = n[i]
    h = r_max/N
    #mash
    r = np.array(range(N))*h+h
    V = 0.5* r**2
    start = time.time()
    for j in range(10):
        mu,phi = solve_GP(V,r,'numerov')
    end = time.time()
    acc_numerov[i]= np.abs(1.5-mu)
    timing_numerov[i]=(end-start)/10
    
for i in range(len(n)):
    r_max = 7
    N = n[i]
    h = r_max/N
    #mash
    r = np.array(range(N))*h+h
    V = 0.5* r**2
    start = time.time()
    for j in range(100):
        mu,phi = solve_GP(V,r,'fd_method')
    end = time.time()
    acc_fd[i]= np.abs(1.5-mu)
    timing_fd[i]=(end-start)/100

#%%    
plt.loglog(n,timing_numerov,label='numerov')
plt.loglog(n,timing_fd,label='finite difference')
ax = plt.gca()
ax.legend()
plt.ylabel('time for 1 execution [s]')
plt.xlabel('numer of points in the mesh')
plt.grid('True')
plt.show()


plt.figure()
plt.loglog(n,acc_numerov,label='numerov')
plt.loglog(n,acc_fd,label='finite difference')
ax = plt.gca()
ax.legend()
plt.ylabel('energy error')
plt.xlabel('numer of points in the mesh')
plt.grid('True')
plt.show()
