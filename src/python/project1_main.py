#FYS-STK4155 project 1 main code:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn import linear_model
#from random import random, seed
import time

from params import *
from linreg_functions import *
from problem_functions import *
from terrain import *
from plot_3d import *

np.random.seed(seed)
t1=time.time()

#unit test
if (True): 
    n=10
    deg=5 #solution is a 4 deg polynom
    run_unit_test(n,deg,lamb=0.0,lasso=False) #OLS

#fit true terrain data
if (False): 
    plot_terrain()


n=21
n2=n**2
#plot surfaces (OLS, Ridge and Lasso)
if (False):
    #plot surfaces
    if (True):
        plot_3D(n)

    #variance testing
    if (False):
        n=[21]
        v=[False]
        d=[True]
        for ni in n:
            for vi in v:
                for di in d:
                    st0=np.random.get_state()
                    plot_3D(ni,var_check=True,diff_noise=di,var_mse=vi)
                    np.random.set_state(st0)
                    plot_3D(ni,var_check=True,lamb=[1e-4],diff_noise=di,var_mse=vi)

#Plot number of gridpoints trade off
if (False):
    run_tradeoff_number()
    
#plot kfold tradeoff (group size/number of groups)
if (False):
    run_tradeoff_kfold()
    
#plot lambda tradeoff
if (False):
    run_tradeoff_lambda()

# plot complexity tradeoff
if (False):
    run_tradeoff_complexity()

# plot complexity tradeoff
if (False): #use the same splits and noise for Lasso and Ridge
    run_tradeoff_complexity_eqsplit()

# a test of OLS, Ridge and Lasso on same data
if (False):
    run_model_comp()

# Running the fits using OLS (part a and b)
if (False):
    part_ab()

t2=time.time()

print('')
print('run time = %.1f sec'%(t2-t1))
print('')
