import numpy as np
import matplotlib.pyplot as plt


h=0.001
N =10**4
E_max = 5
E_step= 0.1
E=0
acc = 0.000001
x= (np.array(range(N))-N/2)*h 
eig = 0.

y_eig = np.zeros([N,1])


def K2(energy,position,spin):
    if spin==0:
        return 2*energy-position**2  
    else:
        return 2*energy-position**2 - 2*spin*(spin+1)/position
def numerov(energy,spin,position,step):
    y_p = np.ones(len(x))
    K2_E = K2(energy,position,0)
    i=2
    y_p[0] =0
    y_p[1] = 1
    while i<N:
        y_p[i]= (y_p[i-1]*(2-5/12*step**2 *K2_E[i-1])-y_p[i-2]*(1+step**2/12*K2_E[i-2]))/(1+step**2/12*K2_E[i])
        i += 1 
    
    return y_p


y_0 = numerov(0,0,x,h)
control_0 = np.sign(y_0[N-1])
while E<E_max: 
    y = numerov(E,0,x,h)   
    control_1 = np.sign(y[N-1])
    
    if control_0 != control_1 :
        
        E_0 = E-E_step
        y_0 = numerov(E-E_step,0,x,h)
        E_1 = E
        y_1 = y
        delta = 1;
        
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
       
        eig = np.append(eig,np.mean([E_0, E_1]))
        y_new = numerov(np.mean([E_0, E_1]),0,x,h)
        norm = np.dot(y_new[1:(N-1)],y_new[1:(N-1)])*h + (y_new[0]**2 +y_new[N-1]**2)/2
        y_eig = np.append(y_eig, y_new/np.sqrt(norm) )
        plt.plot(x,y_new/np.sqrt(norm))
        
    E += E_step
    y_0 = y
    control_0 = np.sign(y_0[N-1])
    
plt.figure()
plt.plot(range(len(eig)-1),eig[1:])
plt.show()  
            
       
    
    
    