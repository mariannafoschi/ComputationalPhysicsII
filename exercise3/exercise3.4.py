
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.colors as colors

#%% general definitions
pi = np.arctan(1)*4

#%% CODE

#Defining range and number of steps

r_max = 8
N = 10**4
h = r_max/N

#mash
r = np.array(range(N))*h




