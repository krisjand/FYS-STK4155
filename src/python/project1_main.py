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
from plot_3d import *

n=21
n2=n**2
t1=time.time()
#plot surfaces (OLS, Ridge and Lasso)
if (False):
    plot_3D(n)

#Plot number of gridpoints trade off
if (False):
    run_tradeoff_number()
    
#plot kfold tradeoff (group size/number of groups)
if (True):
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
