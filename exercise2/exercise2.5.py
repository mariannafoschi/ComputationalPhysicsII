# EXERCISE 2.5
# import packages
import numpy as np
from scipy import linalg
from scipy.optimize import minimize

# general definitions
pi = np.arctan(1)*4

#%% definition of useful functions
def calculate_H(alpha):     #to calculate the matrix H_ij
    n = np.size(alpha)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] = + (5*pi**(3/2)*alpha[i]*alpha[j])/(2*(alpha[i]+alpha[j])**(7/2)) - (2*pi)/(3*(alpha[i]+alpha[j])**2)
    return H

def calculate_S(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = pi**(3/2)/(2*(alpha[i]+alpha[j])**(5/2))
    return S

def generalized_eig(alpha, only_eig = 1):
    n = np.size(alpha)
    H = calculate_H(alpha)
    S = calculate_S(alpha)
    temp_eig, U = linalg.eigh(S)                #define U
    S12 = np.zeros((n,n))                       #define S_1/2
    for i in range(n):
        S12[i, i] = 1/np.sqrt(temp_eig[i])
    V = np.dot(U, S12)                          #define V
    Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
    eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
    C = V.dot(Cprime)
    if only_eig:
        return eig[0]
    else:
        return eig[0], C[:,0]

def calculate_Hsym(alpha):     #to calculate the matrix H_ij
    n = np.size(alpha)
    H = np.zeros((n,n))
    for i in range(n):
        H[i,i] = + (5*pi**(3/2)*alpha[i]*alpha[i])/(2*(alpha[i]+alpha[i])**(7/2)) - (2*pi)/(3*(alpha[i]+alpha[i])**2)
    return H

def calculate_Ssym(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        S[i,i] = pi**(3/2)/(2*(alpha[i]+alpha[i])**(5/2))
    return S

def generalized_eig_sym(alpha, only_eig = 1):
    n = np.size(alpha)
    H = calculate_Hsym(alpha)
    S = calculate_Ssym(alpha)
    temp_eig, U = linalg.eigh(S)                #define U
    S12 = np.zeros((n,n))                       #define S_1/2
    for i in range(n):
        S12[i, i] = 1/np.sqrt(temp_eig[i])
    V = np.dot(U, S12)                          #define V
    Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
    eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
    C = V.dot(Cprime)
    if only_eig:
        return eig[0]
    else:
        return eig[0], C[:,0]

def calculate_H3x3(alpha):     #to calculate the matrix H_ij
    nn = np.size(alpha)
    n = np.size(alpha,0)
    H = np.zeros((nn,nn))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                H[i+3*k,j+3*k] = + (5*pi**(3/2)*alpha[k,i]*alpha[k,j])/(2*(alpha[k,i]+alpha[k,j])**(7/2)) - (2*pi)/(3*(alpha[k,i]+alpha[k,j])**2)
    return H

def calculate_S3x3(alpha):
    nn = np.size(alpha)
    n = np.size(alpha,0)
    S = np.zeros((nn,nn))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                S[i+3*k,j+3*k] = pi**(3/2)/(2*(alpha[k,i]+alpha[k,j])**(5/2))
    return S

def generalized_eig_3x3(alpha_vec, only_eig = 1):
    alpha = np.array([alpha_vec[0:3],alpha_vec[3:6],alpha_vec[6:9]])
    nn = np.size(alpha)
    H = calculate_H3x3(alpha)
    S = calculate_S3x3(alpha)
    temp_eig, U = linalg.eigh(S)                #define U
    S12 = np.zeros((nn,nn))                       #define S_1/2
    for i in range(nn):
        S12[i, i] = 1/np.sqrt(temp_eig[i])
    V = np.dot(U, S12)                          #define V
    Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
    eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
    C = V.dot(Cprime)
    if only_eig:
        return eig[0]
    else:
        return eig[0], C[:,0]
    
def calculate_Hp(alpha):     #to calculate the matrix H_ij
    n = np.size(alpha)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] = + (5*pi**(3/2)*alpha[i]*alpha[j])/(2*(alpha[i]+alpha[j])**(7/2)) - (2*pi)/(3*(alpha[i]+alpha[j])**2)
    return H

def calculate_Sp(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = pi**(3/2)/(2*(alpha[i]+alpha[j])**(5/2))
    return S

def calculate_Hs(alpha):     #to calculate the matrix H_ij
    n = np.size(alpha)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] = 3*pi**(3/2)*alpha[i]*alpha[j]/(alpha[i]+alpha[j])**(5/2) - 2*pi/(alpha[i]+alpha[j])
    return H

def calculate_Ss(alpha):
    n = np.size(alpha)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = pi**(3/2)/(alpha[i]+alpha[j])**(3/2)
    return S

def generalized_eig_sp(alpha_all, only_eig = 1):
    n = np.size(alpha_all)
    alpha_p = np.array(alpha_all[0:3])
    alpha_s = np.array(alpha_all[3:6])
    Hp = calculate_Hp(alpha_p)
    Sp = calculate_Sp(alpha_p)
    Hs = calculate_Hs(alpha_s)
    Ss = calculate_Ss(alpha_s)
    H = np.zeros((n,n))
    S = np.zeros((n,n))
    H[0:3,0:3] = Hp[:,:]
    H[3:6,3:6] = Hs[:,:]
    S[0:3,0:3] = Sp[:,:]
    S[3:6,3:6] = Ss[:,:]
    temp_eig, U = linalg.eigh(S)                #define U
    S12 = np.zeros((n,n))                       #define S_1/2
    for i in range(n):
        S12[i, i] = 1/np.sqrt(temp_eig[i])
    V = np.dot(U, S12)                          #define V
    Hprime = np.dot(V.conj().T, np.dot(H, V) )  #define H'
    eig, Cprime = linalg.eigh(Hprime)           #diagonalize Hprime
    C = V.dot(Cprime)
    if only_eig:
        return eig[0]
    else:
        return eig[0], C[:,0]

#%% Three p-waves and 3 s-wves
#automatic minimization
alpha1 = np.array([0.5, 0.4, 0.6, 0.1, 1, 2.2])        #initial alpha
sol3D = minimize(generalized_eig_sp, alpha1, bounds = ((0.0001,None), (0.0001, None), (0.0001, None), (0.0001,None), (0.0001, None), (0.0001, None)) )

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig_sp(alpha3D, only_eig = 0)
#%% Nine p-waves
#automatic minimization
alpha1 = np.array([0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6])        #initial alpha
sol3D = minimize(generalized_eig_3x3, alpha1, bounds = ((0.0001,None), (0.0001, None), (0.0001, None), (0.0001,None), (0.0001, None), (0.0001, None), (0.0001,None), (0.0001, None), (0.0001, None)))

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig_3x3(alpha3D, only_eig = 0)

#%% Three p-waves only
#automatic minimization
alpha1 = np.array([0.5, 0.4, 0.6])        #initial alpha
sol3D = minimize(generalized_eig, alpha1, bounds = ((0.0001,None), (0.0001, None), (0.0001, None)) )

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig(alpha3D, only_eig = 0)

#%% Three p-waves only - symmetric matrices
#automatic minimization
alpha1 = np.array([0.5, 0.4, 0.6])        #initial alpha
sol3D = minimize(generalized_eig_sym, alpha1, bounds = ((0.0001,None), (0.0001, None), (0.0001, None)) )

#plot of solution
alpha3D = sol3D.x
eig3D, C3D = generalized_eig(alpha3D, only_eig = 0)

