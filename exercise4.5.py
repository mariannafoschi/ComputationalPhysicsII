# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:24:35 2020

@author: zeno1
"""

N = 10**4
rho_b = 3/4/np.pi /(3.93)**3
rho_max = 2*rho_b

rho = (np.array(range(N))+1)*(rho_max)/N
rs = (3/(4*np.pi*rho))**(1/3)
gamma = -0.103756 
beta1 = 0.56371
beta2 = 0.27358

v_correlation_1 = gamma*(1+7/6*beta1*np.sqrt(rs)+4/3*beta2*rs)/(1+beta1*np.sqrt(rs)+beta2*rs)**2

x=rs/11.4
v_correlation_2 = -0.0666*((1+x**3)*np.log(1+1/x)-1/3+x/2-x**2)

v_correlation_3 = -0.88/(rs+7.8)


drs_drho = -(3/(4*np.pi))**(1/3) /3 /rho**(4/3)
v_correlation_2 = v_correlation_2-rho/11.4*drs_drho*0.0666*(3*x**2 *np.log(1+1/x)-2*x-1/x+3/2)
v_correlation_3 =  v_correlation_3 + rho*drs_drho*0.88/(rs+7.8)**2
plt.figure()
plt.plot(rho/rho_b,v_correlation_1,label='Perdew-Zunger (our)')
plt.plot(rho/rho_b,v_correlation_3,label='Wigner')
plt.plot(rho/rho_b,v_correlation_2,label='Gunnarson-Lundqvist')
plt.legend()
plt.grid('true')
plt.xlabel('local density')
plt.ylabel('total')
plt.show()


v_corr_2_der = np.array([(v_correlation_2[i+1]-v_correlation_2[i])/(rho[i+1]-rho[i])*rho[i] for i in range(N-1)])
v_corr_3_der = np.array([(v_correlation_3[i+1]-v_correlation_3[i])/(rho[i+1]-rho[i])*rho[i] for i in range(N-1)])

