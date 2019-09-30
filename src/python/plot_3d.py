
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn import linear_model
#from random import random, seed
import time

from params import *
from p1_functions import *


def plot_3D(n,lamb=[0.0],rand=False):
    # Make data.
    n=21
    n2=n**2

    np.random.seed(seed)


    x = np.arange(0, 1.0001, 1.0/(n-1))
    y = np.arange(0, 1.0001, 1.0/(n-1))
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)

    if (sigma>0.0):
        noise=np.random.normal(0.0,sigma,size=(n,n))
        z=z+noise

    xv,yv=init_xy_vectors(n,rand)
    fv=FrankeFunction(xv,yv)
    
    deg=5
    n_p=(deg+1)*(deg+2)//2
    k=4
    m=100
    beta_mean=np.zeros(n_p)
    betas=np.zeros(shape=(n_p,m*k))
    bv_calc=np.zeros(n_p)
    bv_sum=np.zeros(n_p)
    for i in range(m):
        print(i)
        xk,yk,fk,nk=split_data_kfold(xv,yv,fv,k)
        mse,r2,betak,bv=polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.0)
        betas[:,i*k:(i+1)*k]=betak*1.0
        bv_sum+=np.mean(bv,axis=1)

    beta_mean=np.mean(betas,axis=1)
    bv_sum=bv_sum/m

    for i in range(m*k):
        for j in range(n_p):
            bv_calc[j]+=(betas[j,i]-beta_mean[j])**2
    bv_calc=bv_calc/(m*k)
    b_std=np.sqrt(bv_calc)

    for i in range(n_p):
        print(bv_calc[i],bv_sum[i],b_std[i])
        plt.figure(1)
        plt.plot(betas[i,:],'.')
        plt.plot([0,m*k],[beta_mean[i],beta_mean[i]])
        plt.show()
        plt.clf()
        
    
    b_cl95=b_std*1.96
    b_cl99=b_std*2.58
    print(b_std)
    exit()

    
    zfit=eval_pol3D(beta[:,0],x,y,deg)


    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.scatter(x,y,zfit,marker='.',s=1.,color='r')

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(elev=10., azim=65.)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.yticks(rotation=45)
    plt.title('Franke function')
    plt.show()
    plt.clf()

    return
