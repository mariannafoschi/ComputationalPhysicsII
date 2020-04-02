
# %% Code that computes recursively bessel and neumann functions up to order l.
# -----------------------------------------------------------------------------

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt


l = 6;          # index of the particular bessel function
a = 'n';        # or 'n', depends if you want j_l or n_l

x_max = 20
N = 10**4
h = x_max / N

x = np.array(range(N-1))*h +h   # x è lungo N-1. parte da h e non da 0 perchè non si può dividere per zero
j = np.zeros([N,l+1])       # le j valgono 0 in 0
n = np.ones([N-1,l+1])      # le n non possono essere calcolate in 0. lunghezza N-1

# bessel functions
j[0,0] = 1                  # eccetto per j_0 (0)=1
j_neg = np.cos(x)/x         # j_-1
j[1:,0] =  np.sin(x)/x      # j_0
j[1:,1] = j[1:,0]/x-j_neg     
i=1   
while i<l:
    j[1:,i+1] = (2*i+1)/x *j[1:,i]-j[1:,i-1]    
    i +=1
        
# neumann function
n_neg = np.sin(x)/x 
n[:,0] = -np.cos(x)/x
n[:,1] = n[:,0]/x-n_neg  
k=1   
while k<l:       
    n[:,k+1]=(2*k+1)/x *n[:,k]-n[:,k-1]
    k +=1

# difference with library functions
diff_j = np.zeros([N-1,l+1])
diff_n = np.zeros([N-1,l+1])
for kk in range(l+1):
    diff_j[:,kk] = abs((j[1:,kk]-sp.spherical_jn(kk,x))/sp.spherical_jn(kk,x))
    diff_n[:,kk] = abs((n[:,kk]-sp.spherical_yn(kk,x))/sp.spherical_yn(kk,x))

# %% Plot functions
# -----------------------------------------------------------------------------

# plot bessel
fig_bessel = plt.figure(figsize=(10, 6), dpi=150)                
ax = plt.gca()                      # get handle to axes
ax.grid(True)                       # show grid
for ii in range(l+1):
    graph, = plt.plot(x[10:],j[11:,ii], linewidth =1.5)
    graph.set_label("l = "+str(ii)+" ")
ax.legend()                         # show legend
plt.xlabel("x")
plt.ylabel("j(x)")
plt.show()

# plot neumann
fig_neumann = plt.figure(figsize=(10, 6), dpi=150)                
ax = plt.gca()                      # get handle to axes
ax.grid(True)                       # show grid
ax.set(xlim=(-1, 21), ylim=(-1, 1))
for ii in range(l+1):
    graph, = plt.plot(x[0:],n[0:,ii], linewidth =1.5)
    graph.set_label("l = "+str(ii)+" ")
ax.legend()                         # show legend
plt.xlabel("x")
plt.ylabel("n(x)")
plt.show()

# Plot difference bessel
fig_bessel_diff = plt.figure(figsize=(10, 6), dpi=150)                
ax = plt.gca()                      # get handle to axes
ax.grid(True)                       # show grid
for ii in range(l+1):
    graph, = plt.plot(x[100:],diff_j[100:,ii], linewidth =1)
    graph.set_label("l = "+str(ii)+" ")
ax.set_yscale("Log")
ax.legend()                         # show legend
plt.xlabel("x")
plt.ylabel("j(x): relative difference between our code and the built in functions")
plt.show()

fig_neumann_diff = plt.figure(figsize=(10, 6), dpi=150)                
ax = plt.gca()                      # get handle to axes
ax.grid(True)                       # show grid
for ii in range(l+1):
    graph, = plt.plot(x,diff_n[:,ii], linewidth =1)
    graph.set_label("l = "+str(ii)+" ")
ax.set_yscale("Log")
ax.legend()                         # show legend
plt.xlabel("x")
plt.ylabel("j(x): relative difference between our code and the built in functions")
plt.show()






