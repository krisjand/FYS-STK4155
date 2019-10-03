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
from imageio import imread

def plot_terrain():
    terrain1 = imread('SRTM_data_Norway_1.tif')
    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.show()
    plt.savefig('ter1_full.png')
    plt.clf()
    print(np.shape(terrain1))
    t_max=print(np.amax(np.amax(terrain1)))
    t_min=print(np.amin(np.amin(terrain1)))

    #reduce terrain
    shape_ter=np.shape(terrain1)
    nx=shape_ter[1]
    ny=shape_ter[0]
    res=100
    deg=10
    res2=res**2
    nxr=nx//res
    nyr=ny//res
    x=np.arange(0,nxr)
    y=np.arange(0,nyr)
    ter_res=np.zeros(shape=(nyr,nxr))
    print('rescale')
    for i in range(nyr):
        for j in range(nxr):
            ter_res[i,j]=np.sum(terrain1[i*res:(i+1)*res,j*res:(j+1)*res])/res2


    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(ter_res, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.show()
    plt.savefig('ter1_rescale.png')
    plt.clf()

    print('vectorize')
    x_ter,y_ter,t_ter=init_xy_vectors(1,False,rearr=True,ter=True,x=x,y=y,z=ter_res)
    mean_ter=np.sum(terrain1)/(nx*ny)
#    dter=t_max-t_min
    k=4
    n2=nx*ny
    print('split')
    xk,yk,tk,nk=split_data_kfold(x_ter,y_ter,t_ter,k) #k=number of data groups
    print('fit')
    msek,r2k,betak,var_bk=polfit_kfold(xk,yk,tk,nk,k,n2,deg=deg,lamb=0.0)
    beta=np.mean(betak,axis=1)
    print('evaluate')
    ter_fit=eval_terrain(beta,x,y,deg,nxr,nyr)
    print('plot')
    plt.figure()
    plt.title('Terrain fit over Norway 1')
    plt.imshow(ter_fit, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.show()
    plt.savefig('ter1_fit.png')
    plt.clf()
    return
