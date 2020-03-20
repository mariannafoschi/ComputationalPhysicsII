# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:45:26 2020

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Definition of fundamental parameters
h=0.001 #step 
N = int(0.7*10**4) # number of steps
x= np.array(range(N))*h+h  #define the mesh

E_max = 8 #maximum energy
E_step= 0.09 #step in energy
E=0 #starting energy
acc = 0.001 # accuracy in energy

eig_0 = 0 #initialize a vector where we write the eigenvalues for l=0
eig_1 = 0 #initialize a vector where we write the eigenvalues for l=0
eig_2 = 0 #initialize a vector where we write the eigenvalues for l=0
y_eig_0 = np.zeros([1,N]) #initialize a vector where we write the eigenfunctions with l=0
y_eig_1 = np.zeros([1,N]) #initialize a vector where we write the eigenfunctions with l=1
y_eig_2 = np.zeros([1,N]) #initialize a vector where we write the eigenfunctions with l=2


#%% Definition of useful functions

# Function that define the k^2 parameter in numerov for 3D ho
def K2(energy,position,spin):
    return 2*energy-position**2- spin*(spin+1)/position**2

# Numerov algorithm, give as output a vector
def numerov(energy,spin,position,step): 
    y_p = np.ones(len(x))
    K2_E = K2(energy,position,spin) #evaluate the k^2 parameter   
    i=2
    y_p[0] =(h)**(spin+1) #initial point, at infinity y=0
    y_p[1] = (2*h)**(spin+1) #second point, correct derivative to be set with normalization
    while i<N:
        y_p[i]= (2*y_p[i-1]*(1-5/12*step**2 *K2_E[i-1])-y_p[i-2]*(1+step**2 /12*K2_E[i-2]))/(1+step**2/12*K2_E[i])
        i += 1 
    
    return y_p


#%% Main body
for ell in range(3):
    E=0
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
                y_eig_0 = np.append(y_eig_0,np.reshape(y_new,(1,N)),axis=0)
                eig_0 = np.append(eig_0,np.mean([E_0, E_1]))
            if ell==1:
                y_eig_1 = np.append(y_eig_1,np.reshape(y_new,(1,N)),axis=0)
                eig_1 = np.append(eig_1,np.mean([E_0, E_1]))
            if ell==2:
                y_eig_2 = np.append(y_eig_2,np.reshape(y_new,(1,N)),axis=0)
                eig_2 = np.append(eig_2,np.mean([E_0, E_1]))
            
            
            #plot the eigenfunction
            plt.plot(x,y_new/x)
            
        #increase energy and new sign to be checked
        E += E_step
        y_0 = y
        control_0 = np.sign(y_0[N-1])

#%% Part in which we make a nice plot
    #DA FARE
    
#%% Plot solution for each l
plt.figure()
for i in range(y_eig_0.shape[0]):
    plt.plot(x,y_eig_0[i,:]/x)

plt.figure()
for i in range(y_eig_1.shape[0]):
    plt.plot(x,y_eig_1[i,:]/x)
    
plt.figure()
for i in range(y_eig_2.shape[0]):
    plt.plot(x,y_eig_2[i,:]/x)


#plot the eigenvalue/analytical ones
plt.figure()
plt.plot(np.array(range(len(eig_0[1:]))),eig_0[1:]/(2*np.array(range(len(eig_0[1:])))+1.5))
plt.plot(np.array(range(len(eig_1[1:]))),eig_1[1:]/(2*np.array(range(len(eig_1[1:])))+1+1.5))
plt.plot(np.array(range(len(eig_2[1:]))),eig_2[1:]/(2*np.array(range(len(eig_2[1:])))+2+1.5))
plt.show()



            
       
    
    
    