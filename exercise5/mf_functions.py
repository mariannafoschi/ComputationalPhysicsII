import global_variables as gv
from numba import jit
import numpy as np

#%% FUNCTION DEFINITION
""" Mean field part """
    
def eval_mf_matrix(r, n, levels):
    """ Calculates the matrix with mf functions evaluated in a certain point r """
    temp = np.zeros((n,n))
    a_0 = 1/gv.omega
    # fill the first level (always occupied)
    for i in range(n):
        temp[i,0] = np.exp(-np.sum(r[:,i]**2)/2/a_0)
    # fill other levels
    if n > 1:
        for i in np.arange(n-1)+1:
            if levels[i] == 1:
                for j in range(n):
                    temp[j,i] = r[0,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
            elif levels[i] == 2:
                for j in range(n):
                    temp[j,i] = r[1,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
    return temp.reshape((n,n))


            
def eval_mf_matrix_grad(r, n, levels):
    """ Calculates the gradient matrices of mf functions evaluated in a certain point r """
    temp_gradx = np.zeros((n,n))
    temp_grady = np.zeros((n,n))
    temp_grad2x = np.zeros((n,n))
    temp_grad2y = np.zeros((n,n))
    a_0 = 1/gv.omega

    # fill the first level (always occupied)
    for i in np.arange(n):
        temp_gradx[i,0] = -r[0,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grady[i,0] = -r[1,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2x[i,0] = 1/a_0*(r[0,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2y[i,0] = 1/a_0*(r[1,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        
    # fill other levels
    for i in np.arange(n-1)+1:
        for j in np.arange(n):
            if levels[i] == 1:
                temp_gradx[j,i] = -a_0 * temp_grad2x[j,0]
                temp_grady[j,i] =  r[0,j] * temp_grady[j,0]
                temp_grad2x[j,i] = (3-r[0,j]**2/a_0) * temp_gradx[j,0]
                temp_grad2y[j,i] = r[0,j] * temp_grad2y[j,0]
            if levels[i] == 2:
                temp_gradx[j,i] = r[1,j] * temp_gradx[j,0]
                temp_grady[j,i] = -a_0 * temp_grad2y[j,0]
                temp_grad2x[j,i] = r[1,j] * temp_grad2x[j,0]
                temp_grad2y[j,i] = (3-r[1,j]**2/a_0) * temp_grady[j,0]
    return temp_gradx, temp_grady, temp_grad2x, temp_grad2y

    
def kinetic_energy_nojastrow(r, A_up, A_down):
    """ Calculates the local kinetic energy
        Inputs:
            r = position of which you want to calculate the local kinetic energy.
                Each column is a particle
            A_up = matrix of single particle functions for spin-up particles
            A_down = matrix of single particles functions for spin-down particles
    """
    
    # Mean field part (N=2)
    mf_grad = np.zeros((2,gv.num))
    mf_lap = 0
    [Agradx_up, Agrady_up, Agrad2x_up, Agrad2y_up] = eval_mf_matrix_grad(r[:,:gv.N_up], gv.N_up, [0,1,2])
    A_inv_up = np.linalg.inv(A_up)   # inverse of matrix A (useful for calculating gradient)
    [Agradx_down, Agrady_down, Agrad2x_down, Agrad2y_down] = eval_mf_matrix_grad(r[:,gv.N_up:], gv.N_down, [0,1,2])
    A_inv_down = np.linalg.inv(A_down)   # inverse of matrix A (useful for calculating gradient)
    
    # gradient of first N_plus particles (spin up)
    for l in range(gv.N_up):
        for i in range(gv.N_up):
                mf_grad[0,l] = mf_grad[0,l] + Agradx_up[l,i]*A_inv_up[i,l]
                mf_grad[1,l] = mf_grad[1,l] + Agrady_up[l,i]*A_inv_up[i,l]
    # gradient of last N_minus particles (spin down)
    for l in range(gv.N_down):
        for i in range(gv.N_down):
                mf_grad[0,gv.N_up+l] = mf_grad[0,gv.N_up+l] + Agradx_down[l,i]*A_inv_down[i,l]    #NOTA GLI INDICI INVERTITI
                mf_grad[1,gv.N_up+l] = mf_grad[1,gv.N_up+l] + Agrady_down[l,i]*A_inv_down[i,l]
    
    
    # laplacian
    for l in range(gv.N_up):
        for i in range(gv.N_up):
            mf_lap = mf_lap + Agrad2x_up[l,i]*A_inv_up[i,l] + Agrad2y_up[l,i]*A_inv_up[i,l]
    for l in range(gv.N_down):
        for i in range(gv.N_down):
            mf_lap = mf_lap + Agrad2x_down[l,i]*A_inv_down[i,l] + Agrad2y_down[l,i]*A_inv_down[i,l]
        
    kin_en = -1/2*mf_lap
    
    #feenberg energy
    feenberg_en = np.sum(mf_grad**2)
    feenberg_en = -1/4*(mf_lap-feenberg_en)
    return kin_en, feenberg_en


    
def kinetic_energy(r, A_up, A_down,det_mf,b):
    """ Calculates the local kinetic energy
        Inputs:
            r = position of which you want to calculate the local kinetic energy.
                Each column is a particle
            A_up = matrix of single particle functions for spin-up particles
            A_down = matrix of single particles functions for spin-down particles
    """
    
    # Mean field part (N=2)
    mf_grad = np.zeros((2,gv.num,gv.num_det))
    mf_lap = np.zeros(gv.num_det)
    
    Agradx_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    Agrady_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    Agrad2x_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    Agrad2y_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    Agradx_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))
    Agrady_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))
    Agrad2x_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))
    Agrad2y_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))
    A_inv_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    A_inv_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))

    [Agradx_up[:,:,0], Agrady_up[:,:,0], Agrad2x_up[:,:,0], Agrad2y_up[:,:,0]] = eval_mf_matrix_grad(r[:,:gv.N_up], gv.N_up, gv.level_up[:,0])
    A_inv_up[:,:,0] = np.linalg.inv(A_up[:,:,0])   # inverse of matrix A (useful for calculating gradient)
    [Agradx_down[:,:,0], Agrady_down[:,:,0], Agrad2x_down[:,:,0], Agrad2y_down[:,:,0]] = eval_mf_matrix_grad(r[:,gv.N_up:], gv.N_down, gv.level_down[:,0])
    A_inv_down[:,:,0] = np.linalg.inv(A_down[:,:,0])   # inverse of matrix A (useful for calculating gradient)
    
    if gv.num_det==2:
            [Agradx_up[:,:,1], Agrady_up[:,:,1], Agrad2x_up[:,:,1], Agrad2y_up[:,:,1]] = eval_mf_matrix_grad(r[:,:gv.N_up], gv.N_up, gv.level_up[:,1])
            A_inv_up[:,:,1] = np.linalg.inv(A_up[:,:,1])   # inverse of matrix A (useful for calculating gradient)
            [Agradx_down[:,:,1], Agrady_down[:,:,1], Agrad2x_down[:,:,1], Agrad2y_down[:,:,1]] = eval_mf_matrix_grad(r[:,gv.N_up:], gv.N_down, gv.level_down[:,1])
            A_inv_down[:,:,1] = np.linalg.inv(A_down[:,:,1])   # inverse of matrix A (useful for calculating gradient)

    # gradient of first N_plus particles (spin up)
    for k in range(gv.num_det): 
        
        for l in range(gv.N_up):
            for i in range(gv.N_up):
                    mf_grad[0,l,k] = mf_grad[0,l,k] + Agradx_up[l,i,k]*A_inv_up[i,l,k]
                    mf_grad[1,l,k] = mf_grad[1,l,k] + Agrady_up[l,i,k]*A_inv_up[i,l,k]
        # gradient of last N_minus particles (spin down)
        for l in range(gv.N_down):
            for i in range(gv.N_down):
                mf_grad[0,gv.N_up+l,k] = mf_grad[0,gv.N_up+l,k] + Agradx_down[l,i,k]*A_inv_down[i,l,k]    #NOTA GLI INDICI INVERTITI
                mf_grad[1,gv.N_up+l,k] = mf_grad[1,gv.N_up+l,k] + Agrady_down[l,i,k]*A_inv_down[i,l,k]
    
    
    # laplacian
    for k in range(gv.num_det):
        for l in range(gv.N_up):
            for i in range(gv.N_up):
                mf_lap[k] = mf_lap[k] + Agrad2x_up[l,i,k]*A_inv_up[i,l,k] + Agrad2y_up[l,i,k]*A_inv_up[i,l,k]
        for l in range(gv.N_down):
            for i in range(gv.N_down):
                mf_lap[k] = mf_lap[k] + Agrad2x_down[l,i,k]*A_inv_down[i,l,k] + Agrad2y_down[l,i,k]*A_inv_down[i,l,k]
    
    
    # Jastrow part
    Ulap = 0
    Ugrad = np.zeros((2,gv.num))
    Ugrad2 = 0
    for l in np.arange(gv.num):
        for i in np.arange(gv.num):
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
    
    if gv.num_det ==1:
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf_grad[0,:]+Ugrad[1,:]*mf_grad[1,:])
    
        #summing the contributions
        kin_en = -1/2*mf_lap -1/2*Ulap -  mf_U_grad2
    
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        feenberg_en = np.sum(mf_grad**2)
        feenberg_en = -1/4*(mf_lap-feenberg_en)
#       return kin_en, feenberg_en
    
        feenberg_en = np.sum((Ugrad +mf_grad)**2)
        feenberg_en = -1/4*(mf_lap-feenberg_en)
    if gv.num_det==2 :
        mf2_grad= np.zeros((2,gv.num))
        mf2_grad[0,:] = mf_grad[0,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[0,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_grad[1,:] = mf_grad[1,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[1,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_lap = mf_lap[0]*(1/1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_lap[1]*(1/1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
         
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf2_grad[0,:]+Ugrad[1,:]*mf2_grad[1,:])
    
        #summing the contributions
        kin_en = -1/2*mf2_lap -1/2*Ulap -  mf_U_grad2
        
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        feenberg_en = np.sum((mf2_grad+Ugrad)**2)
        
    return kin_en, feenberg_en

""" Jastrow factor part """
def jastrow_function(r,b):
    out = 0
    for i in range(gv.num):
        for j in np.arange(i+1, gv.num):
             out = out-1/2 * gv.a_param[i,j] * np.sqrt(np.sum(r[:,i]-r[:,j])* (r[:,i]-r[:,j])) / ( 1 + b[i,j] *np.sqrt(np.sum(r[:,i]-r[:,j])* (r[:,i]-r[:,j])) )
    return np.exp(out)

def Udiff(b, r, l, i):
    out = -gv.a_param[l,i]/(2(1+b[l,i]*r)**2)
    return out

def Udiff2(b, r, l, i):
    out = gv.a_param[l,i]*b[l,i]/(1+b[l,i]*r)**3
    return out



# EVERYTHING THAT HAS TO DO WITH ENERGY
   
def potential_energy(r):
    return 1/2 * gv.omega**2 * np.sum(r**2)


    
def density(r,b):
    """ Return the probability density of point r.
        Inputs:
            b = parameters of the Jastrow factors
            r = point of which we want the probability density """
    A_up = np.zeros((gv.N_up,gv.N_up,gv.num_det))
    A_down = np.zeros((gv.N_down,gv.N_down,gv.num_det))   
    
    A_up[:,:,0] = eval_mf_matrix(r[:,:gv.N_up], gv.N_up, gv.level_up[:,0])
    A_down[:,:,0] = eval_mf_matrix(r[:,gv.N_up:], gv.N_down, gv.level_down[:,0])    
    if gv.num_det ==2 :
        A_up[:,:,1] = eval_mf_matrix(r[:,:gv.N_up], gv.N_up, gv.level_up[:,1])
        A_down[:,:,1] = eval_mf_matrix(r[:,gv.N_up:], gv.N_down, gv.level_down[:,1])
    
    #print(np.shape(A_up), np.shape(A_down))
    
    det_mf = np.zeros((2,gv.num_det))
    det_mf[0,0] = np.linalg.det(A_up[:,:,0])
    det_mf[1,0] = np.linalg.det(A_down[:,:,0])
    psi_mf = det_mf[0,0]*det_mf[1,0]
    if gv.num_det ==2 :
        det_mf[0,1] = np.linalg.det(A_up[:,:,1])
        det_mf[1,1] = np.linalg.det(A_down[:,:,1])
        psi_mf = psi_mf + det_mf[0,1]*det_mf[1,1]
    
    psi_jastrow = jastrow_function(r,b)
    
    psi = psi_mf*psi_jastrow
    return psi**2, A_up, A_down, det_mf



def generate_pos(r, delta, mode):
    new_r = np.zeros(np.shape(r))
    if mode==1:
        new_r = r + (np.random.rand(2, gv.num)-1/2)*delta
#    elif mode==2:
#        return 0
    return new_r


    
def sampling_function(r, delta, N_s, mode, cut,b):
    """ This function performs Metropolis algorithm: it samples "num" points from distribution "p" using mode "mode".
        Inputs:
            r = matrix with all initial positions: each column represent a particle [x; y; z]
            b = distribution function to sample (RIVEDI, MI SA CHE BASTANO I PARAMETRI)
            delta = max width of each jump (same along each direction)
            N_s = number of total samples
            mode = 1: moves all particles at each step (better for few particle)
                   2: moves one particle at each step
            cut = number of initial points of the sampling to delete """
    count = 0
    pos = np.zeros((2,gv.num,N_s))
    pot_energy = np.zeros(N_s)
    kin_energy = np.zeros(N_s)
    feenberg_energy = np.zeros(N_s)
    
    prev_density, A_up, A_down,det_mf = density(r,b)
    pos[:,:,0] = r
    kin_energy[0], feenberg_energy[0] = kinetic_energy(r, A_up, A_down,det_mf,b)
    pot_energy[0] = potential_energy(r)
    count = count + 1
    
    if mode==1:
        n = 1
        while n < N_s:
            if n%10000 == 0:
                print(n/10000)
            pos_temp = generate_pos(pos[:,:,n-1], delta, mode)
            new_density, A_up, A_down,det_mf = density(pos_temp,b)
            w = new_density/prev_density   # VEDI COMMENTO QUADERNO, PUO ESSERE IMPORTANTE
            if np.random.rand(1) <= w:
                pos[:,:,n] = pos_temp
                pot_energy[n] = potential_energy(pos_temp)
                kin_energy[n], feenberg_energy[n] = kinetic_energy(pos_temp, A_up, A_down,det_mf,b)
                prev_density = new_density
                count = count + 1
            else:
                pos[:,:,n] = pos[:,:,n-1]
                pot_energy[n] = pot_energy[n-1]
                kin_energy[n] = kin_energy[n-1]
                feenberg_energy[n] = feenberg_energy[n-1]
            n = n + 1
#    elif mode==2:
#        return 0
    print("Accepted steps (%):")
    print(count/N_s*100)
    return pos[:,:,cut:], pot_energy[cut:], kin_energy[cut:], feenberg_energy[cut:]

