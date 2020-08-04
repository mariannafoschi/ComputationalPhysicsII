# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:22:35 2020

@author: Davide
"""

#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg

#%% FUNCTION DEFINITION
""" Mean field part """
def mf_function():
    return 0

#def eval_mf_matrix(r, n, levels):
#    """ Calculates the matrix with mf functions evaluated in a certain point r """
#    temp = np.zeros((n,n))
#    a_0 = 1/omega
#    # fill the first level (always occupied)
#    for i in range(n):
#        temp[i,0] = np.exp(-sum(r[:,i]**2)/2/a_0)
#    # fill other levels
#    for i in np.arange(n-1)+1:
#        for j in range(n):
#            if levels[i] == 1:
#                temp[j,i] = r[0,j] * np.exp(-sum(r[:,j]**2)/2/a_0)
#            if levels[i] == 2:
#                temp[j,i] = r[1,j] * np.exp(-sum(r[:,j]**2)/2/a_0)
#    return temp
#        
#def eval_mf_matrix_grad(r, n, levels):
#    """ Calculates the gradient matrices of mf functions evaluated in a certain point r """
#    temp_gradx = np.zeros((n,n))
#    temp_grady = np.zeros((n,n))
#    temp_grad2x = np.zeros((n,n))
#    temp_grad2y = np.zeros((n,n))
#    a_0 = 1/omega
#
#    # fill the first level (always occupied)
#    for i in range(n):
#        temp_gradx[i,0] = -r[0,i]/a_0 * np.exp(-sum(r[:,i]**2)/2/a_0)
#        temp_grady[i,0] = -r[1,i]/a_0 * np.exp(-sum(r[:,i]**2)/2/a_0)
#        temp_grad2x[i,0] = 1/a_0*(r[0,i]**2/a_0-1) * np.exp(-sum(r[:,i]**2)/2/a_0)
#        temp_grad2y[i,0] = 1/a_0*(r[1,i]**2/a_0-1) * np.exp(-sum(r[:,i]**2)/2/a_0)
#        
#    # fill other levels
#    for i in np.arange(n-1)+1:
#        for j in range(n):
#            if levels[i] == 1:
#                temp_gradx[j,i] = -a_0 * temp_grad2x[j,0]
#                temp_grady[j,i] =  r[0,j] * temp_grady[j,0]
#                temp_grad2x[j,i] = (3-r[0,j]**2/a_0) * temp_gradx[j,0]
#                temp_grad2y[j,i] = r[0,j] * temp_grad2y[j,0]
#            if levels[i] == 2:
#                temp_gradx[j,i] = r[1,j] * temp_gradx[j,0]
#                temp_grady[j,i] = -a_0 * temp_grad2y[j,0]
#                temp_grad2x[j,i] = r[1,j] * temp_grad2x[j,0]
#                temp_grad2y[j,i] = (3-r[1,j]**2/a_0) * temp_grady[j,0]
#    return temp_gradx, temp_grady, temp_grad2x, temp_grad2y

    
    
""" Jastrow factor part """
def jastrow_function(b, r):
    out = 0
    for i in range(num):
        for j in np.arange(i+1, num):
             -1/2 * a_param[i,j] * np.sqrt(sum(r[:,i]-r[:,j])* (r[:,i]-r[:,j])) / ( 1 + b[i,j] *np.sqrt(sum(r[:,i]-r[:,j])* (r[:,i]-r[:,j])) )
    return np.exp(out)

def Udiff(b, r, l, i):
    return -a_param[l,i]/(2(1+b[l,i]*r)**2)

def Udiff2(b, r, l, i):
    return a_param[l,i]*b[l,i]/(1+b[l,i]*r)**3




def kinetic_energy(r, b):
    """ Calculates the local kinetic energy
        Inputs:
            r = position of which you want to calculate the local kinetic energy.
                Each column is a particle
            b = parameters of Jastrow functions
    """
    # Jastrow part
    Ulap = 0
    Ugrad = np.zeros((2,num))
    Ugrad2 = 0
    for l in np.arange(num):
        for i in np.arange(num):
            if i != l:
                dx = r[0,l] - r[0,i]
                dy = r[1,l] - r[1,i]
                drr = dx**2 + dy**2
                dr = np.sqrt(drr)
                
                Up = Udiff(b, dr,l,i) # calculates U'(r)/r
                Upp = Udiff2(b, dr,l,i) #calculates U''(r)
                
                #gradient
                Ugrad[0, l] = Ugrad[0, l] + Up * dx
                Ugrad[1, l] = Ugrad[1, l] + Up * dy
                
                #first part of laplacian
                Ulap = Ulap + Upp + Up
                
        # second part of laplacian
        for k in np.arange(2):
            Ugrad2 = Ugrad2 + Ugrad[k,l]*Ugrad[k,l]
    Ulap = Ulap + Ugrad2
    
    
    
#    # Mean field part (N=3)
#    MF_grad = np.zeros((2,num))
#    MF_lap = 0
#    A_plus = eval_mf_matrix(r, [0,1])                   # matrix with single-particle functions
#    [Agradx_plus, Agrady_plus, Agrad2x_plus, Agrad2y_plus] = eval_mf_matrix_grad(r, [1,2])
#    A_inv_plus = linalg.inv(A_plus)   # inverse of matrix A (useful for calculating gradient)
#    A_minus = eval_mf_matrix(r, [0])                   # matrix with single-particle functions
#    [Agradx_minus, Agrady_minus, Agrad2x_minus, Agrad2y_minus] = eval_mf_matrix_grad(r, [1])
#    A_inv_minus = linalg.inv(A_minus)   # inverse of matrix A (useful for calculating gradient)
#    
#    # gradient of first N_plus particles (spin up)
#    for l in range(N_plus):
#        for i in range(N_plus):
#                MF_grad[0,l] = MF_grad[0,l] + Agradx_plus[l,i]*A_inv_plus[l,i]
#                MF_grad[1,l] = MF_grad[1,l] + Agrady_plus[l,i]*A_inv_minus[l,i]
#    # gradient of last N_minus particles (spin down)
#    for i in range(N_minus):
#            MF_grad[0,N_plus] = MF_grad[0,N_plus] + Agradx_minus*A_inv_minus
#            MF_grad[1,N_plus] = MF_grad[1,N_plus] + Agrady_minus*A_inv_minus
    
    
    # laplacian
    #DA FINIRE

#def density(b, r):
#    """ Return the probability density of point r.
#        Inputs:
#            b = parameters of the Jastrow factors
#            r = point of which we want the probability density """
#    return mf_function
#
#def generate_pos(r, delta, mode):
#    new_r = np.zeros(np.shape(r))
#    if mode==1:
#        for i in range(num):
#            for j in range(2):
#                new_r[j,i] = r[j,i] + (random.random()-1/2)*delta
#    elif mode==2:
#        return 0
#    return new_r
    
#def sampling_function(r, b, delta, N_s, mode, cut):
#    """ This function performs Metropolis algorithm: it samples "num" points from distribution "p" using mode "mode".
#        Inputs:
#            r = matrix with all initial positions: each column represent a particle [x; y; z]
#            b = distribution function to sample (RIVEDI, MI SA CHE BASTANO I PARAMETRI)
#            delta = max width of each jump (same along each direction)
#            N_s = number of total samples
#            mode = 1: moves all particles at each step (better for few particle)
#                   2: moves one particle at each step
#            cut = number of initial points of the sampling to delete """
#    pos = np.array((3,num,N_s))
#    pos[:,:,0] = r
#    if mode==1:
#        n = 1
#        while n < N_s:
#            pos_temp = generate_pos(pos[:,:,n-1], delta, mode)
#            w = density(b, pos_temp)/density(b, pos[:,:,n-1])   # VEDI COMMENTO QUADERNO
#            if w > 1:
#                pos[:,:,n] = pos_temp
#                n = n + 1
#            else:
#                if random.random() <= w:
#                    pos[:,:,n] = pos_temp
#                    n = n + 1
#    elif mode==2:
#        return 0
#    return pos[:,:,cut:]

#%% MAIN PARAMETERS DEFINITION
# for unit conversion at the end
hartree = 11.86 #meV
bohr_radius = 9.793 #nm

omega = 1
a_param = [[0, -2/3 , -2], [ -2/3, 0, -2], [ -2/3, -2/3, 0]] # "a" parameter of the Jastrow factor 
N_up # number of particles with spin up
N_down = 1 #number of particles with spin down
num = N_up + N_down




