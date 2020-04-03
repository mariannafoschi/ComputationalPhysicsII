import numpy as np
import matplotlib.pyplot as plt

<<<<<<< Updated upstream

h=0.001 #step 
N =10**4 # number of steps
=======
#%% Definition of fundamental parameters
h=0.01 #step 
N = int(1.4*10**3) # number of steps
>>>>>>> Stashed changes
x= (np.array(range(N))-N/2)*h  #define the mesh

E_max = 5 #maximum energy
E_step= 0.09 #step in energy
E=0 #starting energy
acc = 0.00001 # accuracy in energy

eig = 0. #initialize a vector where we write the eigenvalues
y_eig = np.zeros([N,1]) #initialize a vector where we write the eigenfunctions


def K2(energy,position,spin): #functions that define the k^2 parameter in numerov for 1D ho
    if spin==0:
        return 2*energy-position**2  
    else:
        return 2*energy-position**2 - 2*spin*(spin+1)/position
def numerov(energy,spin,position,step): #numerov algorithm, give as output a vector
    y_p = np.ones(len(x))
    K2_E = K2(energy,position,spin) #evaluate the k^2 parameter   
    i=2
    y_p[0] =0 #initial point, at infinity y=0
    y_p[1] = 1/100 #second point, correct derivative to be set with normalization
    while i<N:
        y_p[i]= (2*y_p[i-1]*(1-5/12*step**2 *K2_E[i-1])-y_p[i-2]*(1+step**2 /12*K2_E[i-2]))/(1+step**2/12*K2_E[i])
        i += 1 
    
    return y_p


y_0 = numerov(E,0,x,h) #calculate the function on initial energy 
control_0 = np.sign(y_0[N-1]) #where does it diverge at +infinity
while E<E_max: #find eigenvalues by requiring y(+infinity)=0
    y = numerov(E,0,x,h)  #compute y 
    control_1 = np.sign(y[N-1]) #where does it diverge at +infinity
    
    if control_0 != control_1 : #if the sign changes then y(+inifinity)=0 has just passed
        
        #initialize variables for tangent method
        E_0 = E-E_step 
        y_0 = numerov(E-E_step,0,x,h)
        E_1 = E
        y_1 = y
        delta = 1;
        #tangent method
        while delta > acc:
            
            E_2 = E_0 - y_0[N-1]*(E_1-E_0)/(y_1[N-1]-y_0[N-1])
            
            y_2 = numerov(E_2,0,x,h)
            
            control_2 = np.sign(y_2[N-1])
            
            if control_2 == control_1:
                E_1 = E_2;
                y_1=y_2
                delta = E_1-E_0
            else:
                E_0 = E_2
                y_0 =y_2
                delta = E_1-E_0
        
        #copy the eigenvalue
        eig = np.append(eig,np.mean([E_0, E_1]))
    
        #compute eigenfunction
        y_new = numerov(np.mean([E_0, E_1]),0,x,h)
        norm = np.dot(y_new[1:(N-1)],y_new[1:(N-1)])*h + (y_new[0]**2 +y_new[N-1]**2)/2
        y_new =  y_new/np.sqrt(norm) 
        #copy eigenfunction
        y_eig = np.append(y_eig,y_new)
        
        #plot the eigenfunction
        plt.plot(x,y_new)
        
    #increase energy and new sign to be checked
    E += E_step
    y_0 = y
    control_0 = np.sign(y_0[N-1])

# check with analytical solution  if you want   
y_an = np.exp(-x**2 /2)*(64*x**6-480*x**4+720*x**2-120) # in this case the sixth eigenfunction
norm = np.dot(y_an[1:(N-1)],y_an[1:(N-1)])*h + (y_an[0]**2 +y_an[N-1]**2)/2
y_an= y_an/np.sqrt(norm)

plt.plot(x,y_an)

#plot the eigenvalue/analytical ones
plt.figure()
plt.plot(np.array(range(len(eig[1:]))),eig[1:]/(np.array(range(len(eig[1:])))+0.5))
plt.show()



            
       
    
    
    