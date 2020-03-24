# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:45:26 2020

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Definition of fundamental parameters
x_max = 7 # boundary of the mesh are -x_max, +x_max-h
# definition of a vector with the number of steps for each mesh
N_mesh = 10**2*np.array([1, 4, 8, 10, 50, 80, 100, 400, 700, 1000])#[1, 4, 7, 10, 50, 80, 100, 400, 700, 1000]

E_max = 8 #maximum energy
E_step= 0.09 #step in energy
E=0 #starting energy
acc = 10**(-10) # accuracy in energy

eig_0 = np.zeros((len(N_mesh), 3)) #initialize a vector where we write the eigenvalues for l=0
eig_1 = np.zeros((len(N_mesh), 3)) #initialize a vector where we write the eigenvalues for l=0
eig_2 = np.zeros((len(N_mesh), 3)) #initialize a vector where we write the eigenvalues for l=0
y_eig_0 = np.zeros((len(N_mesh), 3, np.max(N_mesh))) #initialize a vector where we write the eigenfunctions with l=0
y_eig_1 = np.zeros((len(N_mesh), 3, np.max(N_mesh))) #initialize a vector where we write the eigenfunctions with l=1
y_eig_2 = np.zeros((len(N_mesh), 3, np.max(N_mesh))) #initialize a vector where we write the eigenfunctions with l=2


#%% Definition of useful functions

# Function that define the k^2 parameter in numerov for 3D ho
def K2(energy,position,spin):
    return 2*energy-position**2- spin*(spin+1)/position**2

# Numerov algorithm, give as output a vector
def numerov(energy,spin,position,step): 
    y_p = np.ones(len(position))
    K2_E = K2(energy,position,spin) #evaluate the k^2 parameter   
    i=2
    y_p[0] =(h)**(spin+1) #initial point, at infinity y=0
    y_p[1] = (2*h)**(spin+1) #second point, correct derivative to be set with normalization
    while i<len(position):
        y_p[i]= (2*y_p[i-1]*(1-5/12*step**2 *K2_E[i-1])-y_p[i-2]*(1+step**2 /12*K2_E[i-2]))/(1+step**2/12*K2_E[i])
        i += 1 
    
    return y_p


#%% Main body
for i in range(len(N_mesh)):
    print(i, "out of", len(N_mesh))
    N = N_mesh[i]                           # define the number of steps in this mesh
    h = x_max/N                          # define the step
    x= (np.array(range(N)))*h + h          #define the mesh
    
    for ell in range(3):
        E = 0                                   # reset energy at zero
        found_E = 0                             # reset found_E
        y_0 = numerov(E,ell,x,h)                  # calculate the function on initial energy 
        control_0 = np.sign(y_0[N-1])           # where does it diverge at +infinity
        while E<E_max:                          # find eigenvalues by requiring y(+infinity)=0
            y = numerov(E,ell,x,h)                # compute y 
            control_1 = np.sign(y[N-1])         # where does it diverge at +infinity
        
            if control_0 != control_1 : #if the sign changes then y(+inifinity)=0 has just passed
                
                #initialize variables for tangent method
                E_0 = E-E_step 
                y_0 = numerov(E-E_step,ell,x,h)
                E_1 = E
                y_1 = y
                delta = 1;
                #tangent method
                while delta > acc:
                    
                    E_2 = E_0 - y_0[N-1]*(E_1-E_0)/(y_1[N-1]-y_0[N-1])
                    
                    y_2 = numerov(E_2,ell,x,h)
                    
                    control_2 = np.sign(y_2[N-1])
                    
                    if control_2 == control_1:
                        E_1 = E_2;
                        y_1=y_2
                        delta = E_1-E_0
                    else:
                        E_0 = E_2
                        y_0 =y_2
                        delta = E_1-E_0
                
            
                #compute eigenfunction
                y_new = numerov(np.mean([E_0, E_1]),ell,x,h)
                norm = (np.dot(y_new[1:(N-1)],y_new[1:(N-1)]) + (y_new[0]**2 +y_new[N-1]**2)/2)*h
                y_new =  y_new/np.sqrt(norm) 
                #copy eigenfunction and eigenvalue
                if ell==0:
                    y_eig_0[i, found_E, :] = np.resize(y_new,(1,np.max(N_mesh)))
                    eig_0[i, found_E] = np.mean([E_0, E_1])
                if ell==1:
                    y_eig_1[i, found_E, :] = np.resize(y_new,(1,np.max(N_mesh)))
                    eig_1[i, found_E] = np.mean([E_0, E_1])
                if ell==2:
                    y_eig_2[i, found_E, :] = np.resize(y_new,(1,np.max(N_mesh)))
                    eig_2[i, found_E] = np.mean([E_0, E_1])
                
                #increase the counter found_E
                found_E += 1
                if found_E>2:
                    E = 10
            
                #plot the eigenfunction
                plt.plot(x,y_new/x)
                
            #increase energy and new sign to be checked
            E += E_step
            y_0 = y
            control_0 = np.sign(y_0[N-1])

    
#%% plot the relative difference between the computed and exact eigenvalue.
plt.figure()
ax = plt.gca()
ax.grid(True)
ax.set_xscale("Log")
ax.set_yscale("Log")
#plot with l=0
for i in range(eig_0.shape[1]):
    t, = plt.loglog(N_mesh, np.abs(eig_0[:,i]-(2*i+1.5))/(2*i+1.5), marker = ".", markersize = 6)
    t.set_label("n="+str(2*i)+"; l=0")
#plot with l=0
for i in range(eig_1.shape[1]):
    t, = plt.loglog(N_mesh, np.abs(eig_1[:,i]-((2*i+1)+1.5))/((2*i+1)+1.5), marker = "x", markersize = 6)
    t.set_label("n="+str(2*i+1)+"; l=1")
#plot with l=0
for i in range(eig_2.shape[1]):
    t, = plt.loglog(N_mesh, np.abs(eig_2[:,i]-(2*(i+1)+1.5))/(2*(i+1)+1.5), marker = "*", markersize = 6)
    t.set_label("n="+str(2*(i+1))+"; l=2")
ax.legend()
plt.errorbar(N_mesh, np.abs(eig_0[:,1]-(3+0.5))/(3+0.5), yerr=acc/2/(3+0.5)*np.ones(len(N_mesh)), fmt = "none", ecolor = "green")
plt.errorbar(N_mesh, np.abs(eig_1[:,1]-(4+0.5))/(4+0.5), yerr=acc/2/(4+0.5)*np.ones(len(N_mesh)), fmt = "none", ecolor = "green")
plt.errorbar(N_mesh, np.abs(eig_2[:,1]-(5+0.5))/(5+0.5), yerr=acc/2/(5+0.5)*np.ones(len(N_mesh)), fmt = "none", ecolor = "green")
plt.xlabel("Number of point in the mesh")
plt.ylabel("Relative difference between eigenvalues")
plt.show()

#%% plot of the wavefunctions with N = N_mesh[6] = 10^4
N = N_mesh[6]+1
h = 2*x_max/(N-1)
x = (np.array(range(N))-(N-1)/2)*h
plt.figure()
ax = plt.gca()
ax.grid(True)
for j in range(5): #loop on the energies
    t,= plt.plot(x, y_eig[6,j,:N_mesh[6]+1])
    t.set_label("n="+str(j)+"; E="+str(2*j+1)+"/2")
ax.legend()
plt.xlabel("Position UNITA NATURALI")
plt.ylabel("Wavefunction amplitude RADICE(1/UNITA NATURALI)")
plt.show()

#%% Plot of the eigenvalue with N = N_mesh[6] = 10^4
plt.figure()
ax = plt.gca()
ax.grid(True)
t, = plt.plot(range(5), eig[6,:], marker = ".", markersize = 6)
t.set_label("Eigenvalues for a mesh of $10^4$ points")
ax.legend()
plt.xlabel("Quantum number n")
plt.ylabel("Eigenvalue [UNITA]")
plt.show()

#%% plot the difference between the computed eigenfunction and the exact one
y_exact = np.zeros((6,N_mesh[6]+1))
pi = np.arctan(1)*4
y_exact[0,:] = pi**(-1/4)*np.exp(-1/2*x**2)
y_exact[1,:] = -pi**(-1/4)*np.exp(-x**2/2)*np.sqrt(2)*x
y_exact[2,:] = pi**(-1/4)*np.exp(-x**2/2)/np.sqrt(2)*(2*x**2-1)
y_exact[3,:] = -pi**(-1/4)*np.exp(-x**2/2)/np.sqrt(3)*(2*x**3-3*x)
y_exact[4,:] = pi**(-1/4)*np.exp(-x**2/2)/2/np.sqrt(6)*(4*x**4-12*x**2+3)
plt.figure()
ax = plt.gca()
ax.grid(True)
for j in range(5): #loop on the energies
    t,= plt.semilogy(x, abs(y_eig[6,j,:N_mesh[6]+1]-y_exact[j,:]))#y_eig[6,j,:N_mesh[6]+1]-
    t.set_label("n="+str(j)+"; E="+str(2*j+1)+"/2")
ax.legend()
plt.xlabel("Position UNITA NATURALI")
plt.ylabel("Wavefunction amplitude RADICE(1/UNITA NATURALI)")
plt.show()

#%% plot the relative difference between the computed eigenfunction and the exact one
plt.figure()
ax = plt.gca()
ax.grid(True)
for j in range(5): #loop on the energies
    t,= plt.semilogy(x, abs((y_eig[6,j,:N_mesh[6]+1]-y_exact[j,:])/y_exact[j,:]))
    t.set_label("n="+str(j)+"; E="+str(2*j+1)+"/2")
ax.legend()
plt.xlabel("Position UNITA NATURALI")
plt.ylabel("Wavefunction amplitude RADICE(1/UNITA NATURALI)")
plt.show()
            
       

            






       
    
    
    