"""
IMPORTANT: for each mesh I calculate ONLY 5 solutions. In order to change it
            you also have to change the dimension of few variables (can be modified)
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Definition of fundamental parameters
x_max = 7 # boundary of the mesh are -x_max, +x_max-h
num_mesh = 10 #how many mesh between 10**2 and 10**4 (included), equidistanti logaritmicamente
# definition of a vector with the number of steps for each mesh
#N_mesh = np.floor( (10**2)*np.power(10*np.ones(num_mesh), 2/(num_mesh-1)*np.array(range(num_mesh))) )
N_mesh = 10**2*np.array([2, 5, 7, 10, 50, 80, 100, 400, 700, 1000])-1#[1, 4, 7, 10, 50, 80, 100, 400, 700, 1000]
E_max = 5 #maximum energy
E_step= 0.09 #step in energy
E=0 #starting energy
acc = 10**(-10) # accuracy in energy
found_E = 0 #counter: number of solutions already found

eig = np.zeros((len(N_mesh), 5)) #initialize a matrix where we write the eigenvalues for each mesh
                                #the first index refers to the mesh, the second to the eigenvalue
                                
y_eig = np.zeros((len(N_mesh), 5, np.max(N_mesh))) #initialize a 3D matrix where we write the eigenfunctions
                                    #the first index is the mesh, the second the eigenvalue, the third the points

#%% Definition of useful functions

# Function that define the k^2 parameter in numerov for 1D ho
def K2(energy,position,spin):
    return 2*energy-position**2

# Numerov algorithm, give as output a vector
def numerov(energy,spin,position,step): 
    y_p = np.ones(len(position))
    K2_E = K2(energy,position,0) #evaluate the k^2 parameter   
    i=2
    y_p[0] =0 #initial point, at infinity y=0
    y_p[1] = 1 #second point, correct derivative to be set with normalization
    while i<len(position):
        y_p[i]= (2*y_p[i-1]*(1-5/12*step**2 *K2_E[i-1])-y_p[i-2]*(1+step**2 /12*K2_E[i-2]))/(1+step**2/12*K2_E[i])
        i += 1 
    
    return y_p


#%% Main body
for i in range(len(N_mesh)):
    print(i, "out of", len(N_mesh))
    N = N_mesh[i]+1                           # define the number of steps in this mesh
    E = 0                                   # reset energy at zero
    found_E = 0                             # reset found_E
    h = 2*x_max/(N-1)                           # define the step
    x= (np.array(range(N))-(N-1)/2)*h           #define the mesh
    y_0 = numerov(E,0,x,h)                  # calculate the function on initial energy 
    control_0 = np.sign(y_0[N-1])           # where does it diverge at +infinity
    while E<E_max:                          # find eigenvalues by requiring y(+infinity)=0
        #print(E)
        y = numerov(E,0,x,h)                # compute y 
        control_1 = np.sign(y[N-1])         # where does it diverge at +infinity
    
        if control_0 != control_1 : #if the sign changes then y(+inifinity)=0 has just passed
            
            #initialize variables for tangent method
            E_0 = E-E_step 
            y_0 = numerov(E_0,0,x,h)
            E_1 = E
            y_1 = y
            delta = 1;
            #tangent method
            while delta > acc:
                
                E_2 = E_0 - y_0[N-1]*(E_1-E_0)/(y_1[N-1]-y_0[N-1])
                y_2 = numerov(E_2,0,x,h)
                
                control_2 = np.sign(y_2[N-1])
                #print(E_0, " ", E_1, " ", E_2)
                #print(control_0, " ", control_1, " ", control_2)
                if control_2 == control_1:
                    E_1 = E_2;
                    y_1=y_2
                    delta = E_1-E_0
                else:
                    E_0 = E_2
                    y_0 =y_2
                    delta = E_1-E_0
            
            #copy the eigenvalue
            eig[i, found_E] = np.mean([E_0, E_1])
            print(np.mean([E_0, E_1]))
        
            #compute eigenfunction
            y_new = numerov(np.mean([E_0, E_1]),0,x,h)
            norm = (np.dot(y_new[1:(N-1)],y_new[1:(N-1)]) + (y_new[0]**2 +y_new[N-1]**2)/2)*h
            y_new =  y_new/np.sqrt(norm) 
            #copy eigenfunction
            y_eig[i, found_E, :] = np.resize(y_new,(1,np.max(N_mesh)))
            
            #increase the counter found_E
            found_E += 1
            
            #plot the eigenfunction
            plt.plot(x,y_new)
            
        #increase energy and new sign to be checked
        E += E_step
        y_0 = y
        control_0 = np.sign(y_0[N-1])

#%% Part in which we make a nice plot
    #DA FARE
    
##%% Check with analytical solution. N=6  (if you want)   
#y_an = np.exp(-x**2 /2)*(64*x**6-480*x**4+720*x**2-120) # in this case the sixth eigenfunction
#norm = (np.dot(y_an[1:(N-1)],y_an[1:(N-1)]) + (y_an[0]**2 +y_an[N-1]**2)/2)*h
#y_an= y_an/np.sqrt(norm)
#
#plt.figure()
#plt.title("Difference between our solution and analytic solution")
##plt.plot(x,y_an)
#plt.plot(x,y_eig[7,:])
#plt.plot(x,y_an-y_eig[7])

#plot the relative difference between the computed and exact eigenvalue.
plt.figure()
ax = plt.gca()
ax.grid(True)
ax.set_xscale("Log")
ax.set_yscale("Log")
for i in range(eig.shape[1]):
    t, = plt.loglog(N_mesh, np.abs(eig[:,i]-(i+0.5))/(i+0.5), marker = ".", markersize = 6)
    t.set_label("n="+str(i))#+"; E="+str(2*i+1)+"/2"
ax.legend()
plt.errorbar(N_mesh, np.abs(eig[:,2]-(2+0.5))/(2+0.5), yerr=acc/2/(2+0.5)*np.ones(len(N_mesh)), fmt = "none", ecolor = "green")
plt.xlabel("Number of point in the mesh")
plt.ylabel("Relative difference between eigenvalues")
plt.show()

#plot of the wavefunctions with N = N_mesh[6] = 10^4
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
plt.xlabel("Position")
plt.ylabel("Wavefunction amplitude")
plt.show()

#Plot of the eigenvalue with N = N_mesh[6] = 10^4
plt.figure()
ax = plt.gca()
ax.grid(True)
t, = plt.plot(range(5), eig[6,:], marker = ".", markersize = 6)
t.set_label("Eigenvalues")
ax.legend()
plt.xlabel("Quantum number n")
plt.ylabel("Energy")
plt.show()

#plot the difference between the computed eigenfunction and the exact one
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
    t.set_label("n="+str(j))
ax.legend()
plt.xlabel("Position")
plt.ylabel("Wavefunction amplitude")
plt.show()

#plot the relative difference between the computed eigenfunction and the exact one
plt.figure()
ax = plt.gca()
ax.grid(True)
for j in range(5): #loop on the energies
    t,= plt.semilogy(x, abs((y_eig[6,j,:N_mesh[6]+1]-y_exact[j,:])/y_exact[j,:]))
    t.set_label("n="+str(j))
ax.legend()
plt.xlabel("Position")
plt.ylabel("Wavefunction amplitude")
plt.show()
            
       
    