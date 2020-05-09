# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:49:22 2020

@author: zeno1
"""

def solve_eq1(potential, r, spin ):
    #input: -potential: starts from r=step, must not include the effective centrifugal one
    #       -r: is the mesh of length N and should starts starts from step
    #       -spin: look for the solution at a certain value of L
    #output:-eigenvalues: array with found eigenvalues in increasing order
    #       -wavefunctions: radial wavefunctions normalized to 1, phi[:,i] is the i-th wavefunction 


    #some useful quantuty
    step = r[1]-r[0]
    h=step
    N= len(r)
    #define output
    eig = [0]
    eigfun = np.zeros((N,1))
    #Add the centrifugal part of the potential
    potential = potential + spin*(spin+1)/2/r**2

    #Energy from which to start to look for eignestates
    E = np.min(potential)
    E_step = 0.05
    #zeroth step 
    phi=numerov(E,potential,spin,r)
    control_0= np.sign(phi[N-1])

    #look for the groun state
    while E<np.max(potential- spin*(spin+1)/2/r**2):
        #increase energy
        E = E+E_step
        phi=numerov(E,potential,spin,r)          
        #check the divergence
        control_1 = np.sign(phi[N-1])
        
        #did the function cross 0 while increaseing the energy? 
        if control_0 != control_1:
            #you found the ground state                       
            check=1
                
            #initialize variables for tangent method
            mu_0 = E-E_step 
            phi_0 = numerov(mu_0,potential,spin,r)
            mu_1 = E
            phi_1 = phi
            delta = 1;
            acc = 10**-12             
            #tangent method
               
            # to count the number of iterations of the tangent method
            counter = 0
            
            while delta > acc and counter < 20:
                counter = counter + 1
            
                mu_2 = mu_0 - phi_0[N-1]*(mu_1-mu_0)/(phi_1[N-1]-phi_0[N-1])
                phi_2 = numerov(mu_2,potential,spin,r)
            
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
                phi_2 = numerov(mu_2,potential,spin,r)
                
                control_2 = np.sign(phi_2[N-1])
                
                if control_2 == control_1:
                    mu_1 = mu_2;
                    phi_1=phi_2
                    delta = mu_1-mu_0
                else:
                    mu_0 = mu_2
                    phi_0 =phi_2
                    delta = mu_1-mu_0     
                        
            #write down the correct state
            eig = np.append(eig,mu_2) 
                          
            phi = phi_2
            norm = (np.dot(phi[1:(N-1)],phi[1:(N-1)]) + (phi[0]**2 +phi[N-1]**2)/2)*h
            phi= phi/np.sqrt(norm)
            eigfun = np.append(eigfun,np.resize(phi,(N,1)),1)
            
        control_0 = control_1
        #Repeat
        
    E_out = eig[1:]
    phi_out =eigfun[:,1:]
    #return the eigenstates state
    return E_out,phi_out
