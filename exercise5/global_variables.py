# GLOBAL VARIABLES
global omega, N_up, N_down, num
def initialize_variables(temp_omega, temp_N_up, temp_N_down):
    global omega, N_up, N_down, num
    omega = temp_omega
    N_up = temp_N_up # number of particles with spin up
    N_down = temp_N_down #number of particles with spin down
    num = N_up + N_down