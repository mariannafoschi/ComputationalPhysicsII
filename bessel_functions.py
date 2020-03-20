# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:59:10 2020

@author: zeno1
"""

import numpy as np
import matplotlib.pyplot as plt

#compute bessel functions
l = 3; #index of the particular bessel function
a = 'n'; #or 'n', depends if you want j_l or n_l

x_max = 20
N = 10**4
h = x_max / N

x = np.array(range(N-1))*h +h #x è lungo N-1. parte da h e non da 0 perchè non si può dividere per zero

if a=='j':

    j = np.zeros([N,l+2]) #le j valgono 0 in 0
    j[0,0] = 1 #eccetto per j_0 (0)=1
    j_neg = np.cos(x)/x # j_-1
    j[1:,0] =  np.sin(x)/x

    j[1:,1] = j[1:,0]/x-j_neg     
    i=1   
    while i<l:
        j[1:,i+1] = (2*i+1)/x *j[1:,i]-j[1:,i-1]    
        i +=1
        
else:
    n = np.ones([N-1,l+2]) #le n non possono essere calcolate in 0. lunghezza N-1
    n_neg = np.sin(x)/x 
    n[:,0] = -np.cos(x)/x
    
    n[:,1] = n[:,0]/x-n_neg
        
    i=1   
    while i<l:       
        n[:,i+1]=(2*i+1)/x *n[:,i]-n[:,i-1]
        i +=1
 




