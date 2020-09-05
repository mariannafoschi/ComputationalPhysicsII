from numba import jit, prange
import numpy as np
import time
import matplotlib.pyplot as plt

#%% FUNCTION DEFINITION
""" Mean field part """

@jit(nopython=True, cache = True)
def eval_mf_matrix(r, n, levels):
    """ Calculates the matrix with mf functions evaluated in a certain point r """
    temp = np.zeros((n,n))
    a_0 = 1/omega
    # fill the first level (always occupied)
    for i in np.arange(n):
        temp[i,0] = np.exp(-np.sum(r[:,i]**2)/2/a_0)
    # fill other levels
    if n > 1:
        for i in np.arange(n-1)+1:
            if levels[i] == 1:
                for j in np.arange(n):
                    temp[j,i] = r[0,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
            elif levels[i] == 2:
                for j in np.arange(n):
                    temp[j,i] = r[1,j] * np.exp(-np.sum(r[:,j]**2)/2/a_0)
    return temp.reshape((n,n))


@jit(nopython=True, cache = True)              
def eval_mf_matrix_grad(r, n, levels):
    """ Calculates the gradient matrices of mf functions evaluated in a certain point r """
    temp_gradx = np.zeros((n,n))
    temp_grady = np.zeros((n,n))
    temp_grad2x = np.zeros((n,n))
    temp_grad2y = np.zeros((n,n))
    a_0 = 1/omega

    # fill the first level (always occupied)
    for i in np.arange(n):
        temp_gradx[i,0] = -r[0,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grady[i,0] = -r[1,i]/a_0 * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2x[i,0] = 1/a_0*(r[0,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        temp_grad2y[i,0] = 1/a_0*(r[1,i]**2/a_0-1) * np.exp(-np.sum(r[:,i]**2)/2/a_0)
        
    # fill other levels
    for i in np.arange(1,n):
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

@jit(nopython=True, cache = True)
def kinetic_energy(r, A_up, A_down,det_mf,b):
    """ Calculates the local kinetic energy
        Inputs:
            r = position of which you want to calculate the local kinetic energy.
                Each column is a particle
            A_up = matrix of single particle functions for spin-up particles
            A_down = matrix of single particles functions for spin-down particles
    """
    
    # Mean field part (N=2)
    mf_grad = np.zeros((2,num,num_det))
    mf_lap = np.zeros(num_det)
    
    Agradx_up = np.zeros((N_up,N_up,num_det))
    Agrady_up = np.zeros((N_up,N_up,num_det))
    Agrad2x_up = np.zeros((N_up,N_up,num_det))
    Agrad2y_up = np.zeros((N_up,N_up,num_det))
    Agradx_down = np.zeros((N_down,N_down,num_det))
    Agrady_down = np.zeros((N_down,N_down,num_det))
    Agrad2x_down = np.zeros((N_down,N_down,num_det))
    Agrad2y_down = np.zeros((N_down,N_down,num_det))
    A_inv_up = np.zeros((N_up,N_up,num_det))
    A_inv_down = np.zeros((N_down,N_down,num_det))
    
    [Agradx_up[:,:,0], Agrady_up[:,:,0], Agrad2x_up[:,:,0], Agrad2y_up[:,:,0]] = eval_mf_matrix_grad(r[:,:N_up], N_up, level_up[:,0])
    A_inv_up[:,:,0] = np.linalg.inv(A_up[:,:,0])   # inverse of matrix A (useful for calculating gradient)
    [Agradx_down[:,:,0], Agrady_down[:,:,0], Agrad2x_down[:,:,0], Agrad2y_down[:,:,0]] = eval_mf_matrix_grad(r[:,N_up:], N_down, level_down[:,0])
    A_inv_down[:,:,0] = np.linalg.inv(A_down[:,:,0])   # inverse of matrix A (useful for calculating gradient)
    
    if num_det==2:
            [Agradx_up[:,:,1], Agrady_up[:,:,1], Agrad2x_up[:,:,1], Agrad2y_up[:,:,1]] = eval_mf_matrix_grad(r[:,:N_up], N_up, level_up[:,1])
            A_inv_up[:,:,1] = np.linalg.inv(A_up[:,:,1])   # inverse of matrix A (useful for calculating gradient)
            [Agradx_down[:,:,1], Agrady_down[:,:,1], Agrad2x_down[:,:,1], Agrad2y_down[:,:,1]] = eval_mf_matrix_grad(r[:,N_up:], N_down, level_down[:,1])
            A_inv_down[:,:,1] = np.linalg.inv(A_down[:,:,1])   # inverse of matrix A (useful for calculating gradient)

    # gradient of first N_plus particles (spin up)
    for k in np.arange(num_det):         
        for l in np.arange(N_up):
            for i in np.arange(N_up):
                    mf_grad[0,l,k] = mf_grad[0,l,k] + Agradx_up[l,i,k]*A_inv_up[i,l,k]
                    mf_grad[1,l,k] = mf_grad[1,l,k] + Agrady_up[l,i,k]*A_inv_up[i,l,k]
        # gradient of last N_minus particles (spin down)
        for l in np.arange(N_down):
            for i in np.arange(N_down):
                mf_grad[0,N_up+l,k] = mf_grad[0,N_up+l,k] + Agradx_down[l,i,k]*A_inv_down[i,l,k]    #NOTA GLI INDICI INVERTITI
                mf_grad[1,N_up+l,k] = mf_grad[1,N_up+l,k] + Agrady_down[l,i,k]*A_inv_down[i,l,k]
    
    
    # laplacian
    for k in np.arange(num_det):
        for l in np.arange(N_up):
            for i in np.arange(N_up):
                mf_lap[k] = mf_lap[k] + (Agrad2x_up[l,i,k] + Agrad2y_up[l,i,k])*A_inv_up[i,l,k]
        for l in np.arange(N_down):
            for i in np.arange(N_down):
                mf_lap[k] = mf_lap[k] + (Agrad2x_down[l,i,k] + Agrad2y_down[l,i,k])*A_inv_down[i,l,k]
    
    
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
                    
                Up = Udiff(b, dr,l,i) # calculates U'(r)/r with U = ar/(1+br)
                Upp = Udiff2(b, dr,l,i) #calculates U''(r) with U = ar/(1+br)
                
                #gradient
                Ugrad[0, l] = Ugrad[0, l] + Up * dx
                Ugrad[1, l] = Ugrad[1, l] + Up * dy
                
                #first part of laplacian
                Ulap = Ulap + Upp + Up
                
        # second part of laplacian
        for k in np.arange(2):
            Ugrad2 = Ugrad2 + Ugrad[k,l]*Ugrad[k,l]
    Ulap = Ulap + Ugrad2 

    #FEENBERG DA CONTROLLARE
    if num_det ==1:
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf_grad[0,:,0]+Ugrad[1,:]*mf_grad[1,:,0])
        #summing the contributions
        kin_en = -1/2*mf_lap[0] -1/2*Ulap -  mf_U_grad2
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        #feenberg_en = np.sum(np.sum(mf_grad[:,:,0]**2,axis=0))
        #feenberg_en = -1/4*(mf_lap-feenberg_en)
#       return kin_en, feenberg_en
    
        a = np.sum(np.sum((Ugrad +mf_grad[:,:,0])**2,axis=0))
        feenberg_en = -1/4*(mf_lap[0]-a)
        #feenberg_en = 0
    if num_det==2 :
        mf2_grad= np.zeros((2,num))
        mf2_grad[0,:] = mf_grad[0,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[0,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_grad[1,:] = mf_grad[1,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[1,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_lap = mf_lap[0]*(1/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0])) + mf_lap[1]*(1/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1]))
         
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf2_grad[0,:]+Ugrad[1,:]*mf2_grad[1,:])
    
        #summing the contributions
        kin_en = -1/2*mf2_lap -1/2*Ulap -  mf_U_grad2
        
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        a = np.sum(np.sum((mf2_grad+Ugrad)**2,0))
        feenberg_en = 1/4*(mf2_lap-a)
    return kin_en,feenberg_en, mf_grad, mf_lap

@jit(nopython=True, cache = True)        
def kinetic_energy_var(r, mf_grad, mf_lap,det_mf,b):
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
                    
                Up = Udiff(b, dr,l,i) # calculates U'(r)/r with U = ar/(1+br)
                Upp = Udiff2(b, dr,l,i) #calculates U''(r) with U = ar/(1+br)
                
                #gradient
                Ugrad[0, l] = Ugrad[0, l] + Up * dx
                Ugrad[1, l] = Ugrad[1, l] + Up * dy
                
                #first part of laplacian
                Ulap = Ulap + Upp + Up
                
        # second part of laplacian
        for k in np.arange(2):
            Ugrad2 = Ugrad2 + Ugrad[k,l]*Ugrad[k,l]
    Ulap = Ulap + Ugrad2 
    #kin_en = np.array([1.])
    if num_det ==1:
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf_grad[0,:,0]+Ugrad[1,:]*mf_grad[1,:,0])
    
        #summing the contributions
        kin_en = -1/2*mf_lap[0] -1/2*Ulap -  mf_U_grad2
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        #feenberg_en = np.sum(np.sum(mf_grad[:,:,0]**2,axis=0))
        #feenberg_en = -1/4*(mf_lap-feenberg_en)
#       return kin_en, feenberg_en
    
        a = np.sum(np.sum((mf_grad[:,:,0]+Ugrad)**2,0))
        feenberg_en = 1/4*(mf_lap[0]-a)

        
    if num_det==2 :
        mf2_grad= np.zeros((2,num))
        mf2_grad[0,:] = mf_grad[0,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[0,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_grad[1,:] = mf_grad[1,:,0]/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0]) + mf_grad[1,:,1]/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1])
        mf2_lap = mf_lap[0]*(1/(1+det_mf[0,1]*det_mf[1,1]/det_mf[0,0]/det_mf[1,0])) + mf_lap[1]*(1/(1+det_mf[0,0]*det_mf[1,0]/det_mf[0,1]/det_mf[1,1]))
        #gradient times gradient part
        mf_U_grad2 = np.sum(Ugrad[0,:]*mf2_grad[0,:]+Ugrad[1,:]*mf2_grad[1,:])
    
        #summing the contributions
        kin_en = -1/2*mf2_lap -1/2*Ulap -  mf_U_grad2
        
        #feenberg energy        SOLO SE usiamo la funzione senza jastrow?
        a = np.sum(np.sum((mf2_grad[:,:]+Ugrad)**2,0))
        feenberg_en = 1/4*(mf2_lap-a)
    return kin_en,feenberg_en

""" Jastrow factor part """
@jit(nopython=True, cache = True)
def jastrow_function(r,b):
    out = 0
    for i in np.arange(0,num):
        for j in np.arange(i+1, num):
            r_ij = np.sqrt(np.sum((r[:,i]-r[:,j])**2))
            out = out + a_param[i,j] *r_ij / ( 1 + b[i,j] * r_ij )
    return np.exp(out)

@jit(nopython=True, cache = True) # non parallellizza
def Udiff(b, r, l, i):
    return a_param[l,i]/(1+b[l,i]*r)**2 /r

@jit(nopython=True, cache = True) # non parallellizza
def Udiff2(b, r, l, i):
    return -2*a_param[l,i]*b[l,i]/(1+b[l,i]*r)**3



# EVERYTHING THAT HAS TO DO WITH ENERGY
@jit(nopython=True, cache = True) # parallelliza ma va un pelo più lento
def potential_energy(r):
    return 1/2 * omega**2 * np.sum(r**2)

@jit(nopython=True, cache = True) # parallellizza ma va più lento
def coulomb_energy(r):
    energy = 0 
    for i in prange(num):
        for j in prange(i+1,num):
            energy = energy + 1/np.sqrt(np.sum((r[:,i]-r[:,j])**2))
    return energy


@jit(nopython=True, cache = True) # non va
def density(r,b):
    """ Return the probability density of point r.
        Inputs:
            b = parameters of the Jastrow factors
            r = point of which we want the probability density """
    A_up = np.zeros((N_up,N_up,num_det))
    A_down = np.zeros((N_down,N_down,num_det))   
    
    A_up[:,:,0] = eval_mf_matrix(r[:,:N_up], N_up, level_up[:,0])
    A_down[:,:,0] = eval_mf_matrix(r[:,N_up:], N_down, level_down[:,0])    
    if num_det ==2 :
        A_up[:,:,1] = eval_mf_matrix(r[:,:N_up], N_up, level_up[:,1])
        A_down[:,:,1] = eval_mf_matrix(r[:,N_up:], N_down, level_down[:,1])
    
    #print(np.shape(A_up), np.shape(A_down))
    
    det_mf = np.zeros((2,num_det))
    det_mf[0,0] = np.linalg.det(A_up[:,:,0])
    det_mf[1,0] = np.linalg.det(A_down[:,:,0])
    psi_mf = det_mf[0,0]*det_mf[1,0]
    if num_det ==2 :
        det_mf[0,1] = np.linalg.det(A_up[:,:,1])
        det_mf[1,1] = np.linalg.det(A_down[:,:,1])
        psi_mf = psi_mf + det_mf[0,1]*det_mf[1,1]
    
    psi_jastrow = jastrow_function(r,b)
    
    psi = psi_mf*psi_jastrow
    return psi**2, A_up, A_down, det_mf


@jit(nopython=True, cache = True) #non va
def generate_pos(r, delta, mode):
    return r + (np.random.rand(2, num)-1/2)*delta

@jit(nopython=True, cache = True)
def generate_b(b_):
    b = np.zeros((num,num))             
    b[:N_up,:N_up] = np.ones((N_up,N_up))*b_[0] # up-up
    b[:N_up,N_up:] = np.ones((N_up,N_down))*b_[1] # up-down
    b[N_up:,:N_up] = np.ones((N_down,N_up))*b_[1] # down-up
    b[N_up:,N_up:] = np.ones((N_down,N_down))*b_[0] # down-down
    return b

@jit(nopython=True, cache = True)
def reweight_function(samples,mf_grad,mf_lap,det_mf,b_1,b_2,b_vec,Ns):
    reweight = np.zeros((Ns,2))
    kin_energy_var = np.zeros((Ns,2))
    finberg =np.zeros((Ns,2))
    for i in np.arange(Ns):
            sam = samples[:,:,i]
            reweight[i,0] = jastrow_function(sam,b_1)/jastrow_function(sam,b_vec)
            reweight[i,1] = jastrow_function(sam,b_2)/jastrow_function(sam,b_vec)
            kin_energy_var[i,0],finberg[i,0]= kinetic_energy_var(sam,mf_grad[:,:,:,i],mf_lap[:,i],det_mf[:,:,i],b_1)
            kin_energy_var[i,1],finberg[i,1] = kinetic_energy_var(sam,mf_grad[:,:,:,i],mf_lap[:,i],det_mf[:,:,i],b_2)
    return reweight**2, kin_energy_var
         
@jit(nopython=True, cache = True) 
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
    pos = np.zeros((2,num,N_s))
    pot_energy = np.zeros(N_s)
    coul_energy = np.zeros(N_s)
    kin_energy = np.zeros(N_s)
    mf_grad = np.zeros((2,num,num_det,N_s))
    mf_lap = np.zeros((num_det,N_s))
    det_mf = np.zeros((2,num_det,N_s))
    feenberg_energy = np.zeros(N_s)
    
    prev_density, A_up, A_down,det_mf[:,:,0] = density(r,b)
    pos[:,:,0] = r
    tempa = np.copy(A_up)
    tempb = np.copy(A_down)
    kin_energy[0], feenberg_energy[0],mf_grad[:,:,:,0],mf_lap[:,0] = kinetic_energy(r, tempa, tempb, det_mf[:,:,0],b)
    pot_energy[0] = potential_energy(r)
    coul_energy[0] = coulomb_energy(r)
    pos_old = r
    det_mf_old =det_mf[:,:,0]
    A_up_old = A_up
    A_down_old = A_down
    count = count + 1
    
    n=1
    m=1
    m_corr = 1 #correlation lenght (you take one sample every m_corr)
    while n<N_s:      
        pos_temp = generate_pos(pos_old, delta, mode)#generate_pos(pos[:,:,n-1], delta, mode)
        new_density, A_up, A_down,det_mf_temp = density(pos_temp,b)
        w = new_density/prev_density   # VEDI COMMENTO QUADERNO, PUO ESSERE IMPORTANTE
        temp = np.random.rand(1)
        
        if temp[0] <= w:
            if m==m_corr:
                pos[:,:,n] = pos_temp
                pot_energy[n]= potential_energy(pos_temp)
                coul_energy[n] = coulomb_energy(pos_temp) 
                det_mf[:,:,n]=det_mf_temp
                tempa = np.copy(A_up)
                tempb = np.copy(A_down)
                kin_energy[n], feenberg_energy[n],mf_grad[:,:,:,n],mf_lap[:,n] = kinetic_energy(pos_temp, tempa, tempb,det_mf_temp,b)
                count = count + 1
            prev_density = new_density
            pos_old = pos_temp
            A_up_old = A_up
            A_down_old = A_down
            det_mf_old = det_mf_temp
        else:
            if m==m_corr:
                pos[:,:,n] = pos_old
                pot_energy[n] = potential_energy(pos_old) 
                coul_energy[n] = coulomb_energy(pos_old)
                det_mf[:,:,n]=det_mf_old
                tempa = np.copy(A_up_old)
                tempb = np.copy(A_down_old)
                kin_energy[n], feenberg_energy[n], mf_grad[:,:,:,n], mf_lap[:,n] = kinetic_energy(pos_old, tempa, tempb, det_mf_old, b)
#                test che non cambiamo cose
#                tempa = np.copy(A_up_old)
#                tempb = np.copy(A_down_old)
#                kin_energya, feenberg_energya, mf_grada, mf_lapa = kinetic_energy(pos_old, tempa, tempb, det_mf_old, b)
#                if kin_energy[n] == kin_energya:
#                    count += 1

#                pos[:,:,n] = pos[:,:,n-1]
#                pot_energy[n] = pot_energy[n-1]
#                coul_energy[n] = coul_energy[n-1]
#                kin_energy[n] = kin_energy[n-1]
#                mf_grad[:,:,:,n] = mf_grad[:,:,:,n-1]
#                mf_lap[:,n] = mf_lap[:,n-1]
#                det_mf[:,:,n]=det_mf[:,:,n-1]
#                feenberg_energy[n] = feenberg_energy[n-1]

        
        if m==m_corr:
            n = n+1
            m=1 
        else:
            m=m+1
    print("Accepted steps (%):")
    print(count/N_s*100)
    return pos[:,:,cut:], pot_energy[cut:],coul_energy[cut:], kin_energy[cut:], feenberg_energy[cut:],mf_grad[:,:,:,cut:],mf_lap[:,cut:],det_mf[:,:,cut:]

#%% 
global omega, N_up, N_down, num, L4,a_param,level_up, level_down, num_det
def initialize_variables(temp_omega, temp_N_up, temp_N_down, temp_L4, temp_jastrow):
    global omega, N_up, N_down, num , L4,a_param
    omega = temp_omega
    N_up = temp_N_up # number of particles with spin up
    N_down = temp_N_down #number of particles with spin down
    num = N_up + N_down
    L4 = temp_L4 #angular momentum state when num=4, can be 0 (l=0,S=0) 1 (l=0,S=1) 2 (l=2,S=0)
    a_param = np.zeros((num,num))  # jasstrow factor parameter
    a_param[:N_up,:N_up] = np.ones((N_up,N_up))*1/3. # up-up
    a_param[:N_up,N_up:] = np.ones((N_up,N_down))*1. # up-down
    a_param[N_up:,:N_up] = np.ones((N_down,N_up))*1. # down-up
    a_param[N_up:,N_up:] = np.ones((N_down,N_down))*1/3. # down-down
    if temp_jastrow ==0:
        a_param = np.zeros((num,num))
    
def occ_levels(L4):
    global level_up, level_down, num_det
    if num == 2:
        level_up = np.ones((1,1))*0
        level_down = np.ones((1,1))*0
        num_det = 1
    if num == 3: 
        level_up = np.transpose(np.array([[0,1],[0,2]]))
        level_down = np.ones((1,2))*0
        num_det = 2
    if num == 4:
        if L4 == 0:
            level_up = np.transpose(np.array([[0,1],[0,2]]))
            level_down = np.transpose(np.array([[0,2],[0,1]]))
            num_det = 2
        if L4 == 1:
            level_up =np.transpose( np.array([[0,1,2]]))
            level_down = np.ones((1,1))*0
            num_det = 1
        if L4 == 2:
            level_up = np.transpose(np.array([[0,1],[0,2]]))
            level_down = level_up
            num_det = 2
    if num == 5:
        level_up = np.transpose(np.array([[0,1,2],[0,1,2]]))
        level_down = np.transpose(np.array([[0,1],[0,2]]))
        num_det = 2
    if num == 6: 
        level_up = np.transpose(np.array([[0,1,2]]))
        level_down = level_up
        num_det = 1

    level_up = np.int_(level_up)
    level_down = np.int_(level_down)
    
#%% MAIN PARAMETERS DEFINITION
temp_omega = 0.5
temp_num =2
temp_N_up =1
temp_N_down = temp_num - temp_N_up
temp_L4=0
if temp_num ==4:
    if temp_N_up ==3:
        temp_L4 = 1 #to be set when num=4
temp_jastrow = 1

initialize_variables(temp_omega, temp_N_up, temp_N_down,temp_L4, temp_jastrow)
occ_levels(temp_L4)

# STIMA TEMPI: PER 2 PARTICELLE CON JASTROW. 18 sec per 10^5, 51 sec per 10^6, 385 per 10^7
r_init = np.random.rand(2, num)     # initial position NOTE: FIRST 2 PARTICLES MUST BE IN DIFFERENT POSITIONS OTHERWISE DENSITY IS ZERO (E NOI DIVIDIAMO PER LA DENSITà)


# choose your delta
    # num = 2, omega = 1            delta = 2
    # num = 3, omega = 1            delta = 2
    # num = 4, omega = 1, L4 = 0    delta = 2
    # num = 4, omega = 1, L4 = 1    delta = 2
    # num = 4, omega = 1, L4 = 2    delta = 2
    # num = 5, omega = 1            delta = 2
    # num = 6, omega = 1            delta = 2
    
    # num = 2, omega = 0.5            delta = 2
    # num = 3, omega = 0.5            delta = 2
    # num = 4, omega = 0.5, L4 = 0    delta = 2
    # num = 4, omega = 0.5, L4 = 1    delta = 2
    # num = 4, omega = 0.5, L4 = 2    delta = 2
    # num = 5, omega = 0.5            delta = 2
    # num = 6, omega = 0.5            delta = 2
deltamin = 2.5
deltamax = 3.5
nummero = 10
deltaa = np.arange(deltamin,deltamax,(deltamax-deltamin)/nummero)
plt.figure()
for k in range(nummero):  
    delta = deltaa[k]                 # width of movement
    N_s = 10**5          # number of samples
    cut = 10**3
    mode = 1
    
    # choose your b ([b_upup, b_updown])
        # num = 2, omega = 1            b = np.array([1., 0.384815])
        # num = 3, omega = 1            b = np.array([0.215, 0.456])
        # num = 4, omega = 1, L4 = 0    b = np.array([0.225,0.501])
        # num = 4, omega = 1, L4 = 1    b = np.array([0.207,0.507])
        # num = 4, omega = 1, L4 = 2    b = np.array([0.48986,0.4047])
        # num = 5, omega = 1            b = np.array([0.2294, 0.525])
        # num = 6, omega = 1            b = np.array([0.232,0.5449])
        
        # num = 2, omega = 0.5            b = np.array([1.,0.2977])
        # num = 3, omega = 0.5            b = np.array([0.1662,0.3370])
        # num = 4, omega = 0.5, L4 = 0    b = np.array([0.1605,0.3720])
        # num = 4, omega = 0.5, L4 = 1    b = np.array([0.1587,0.3688])
        # num = 4, omega = 0.5, L4 = 2    b = np.array([0.3453,0.303])
        # num = 5, omega = 0.5            b = np.array([0.167,0.388])
        # num = 6, omega = 0.5            b = np.array([0.17495,0.39695])
    b= np.zeros((num,num))
    b= generate_b(np.array([1.,0.2977])) #b_upup b_updown
    
    single_trial = 1
    t_in = time.time()
    if single_trial==1:
        samples, pot_energy,coul_energy, kin_energy, feenberg_energy,mf_grad,mf_lap,det_mf = sampling_function(r_init, delta, N_s, mode, cut,b)
        
        energy = np.mean(pot_energy+kin_energy)
        energy_err = np.sqrt(1/(N_s-cut))*np.sqrt(np.mean((kin_energy+pot_energy)**2)-energy**2)
        #print('kin + pot =', energy, '+-',energy_err)
        energy_fee = np.mean(pot_energy+feenberg_energy)
        energy_err_fee = np.sqrt(1/(N_s-cut))*np.sqrt(np.mean((pot_energy+feenberg_energy)**2)-energy_fee**2)
        # print('energy feenberg (kin + pot)=', energy_fee, '+-',energy_err_fee)
        # print(" ")
        energy = np.mean(pot_energy+kin_energy+coul_energy)
        energy_err = np.sqrt(1/(N_s-cut))*np.sqrt(np.mean((kin_energy+pot_energy+coul_energy)**2)-energy**2)
        # print('kin + pot + coulomb=', energy, '+-',energy_err)
        energy_fee = np.mean(pot_energy+feenberg_energy+coul_energy)
        energy_err_fee = np.sqrt(1/(N_s-cut))*np.sqrt(np.mean((pot_energy+feenberg_energy+coul_energy)**2)-energy_fee**2)
        # print('energy feenberg (kin + pot + coul)=', energy_fee, '+-',energy_err_fee)
        # print(" ")
    print("Delta", delta)
    t_fin= time.time()
    print("Single trial time = ", t_fin-t_in)
    print(" ")
    
            
    # visualize correlation between steps
    correlation = 1
    if correlation == 1:
        massimo = 17
        corr = np.zeros(massimo)
        if temp_jastrow == 0: #no interagente
            energy = np.mean(pot_energy+kin_energy)
            for k in np.arange(massimo):
                corr[k] = np.sum( (pot_energy[:-k-1] + kin_energy[:-k-1] - energy)*(pot_energy[k:-1] + kin_energy[k:-1] - energy) )
            corr = corr/np.sum( (pot_energy + kin_energy - energy)**2 )
        else:
            energy = np.mean(pot_energy+kin_energy+coul_energy)
            for k in np.arange(massimo):
                corr[k] = np.sum( (pot_energy[:-k-1] + kin_energy[:-k-1] + coul_energy[:-k-1] - energy)*(pot_energy[k:-1] + kin_energy[k:-1] + coul_energy[k:-1] - energy) )
            corr = corr/np.sum( (pot_energy + kin_energy + coul_energy - energy)**2 )
        plt.plot(corr, label = delta)
        plt.grid(True)
plt.legend()