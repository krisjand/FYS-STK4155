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
    t_max=np.amax(np.amax(terrain1))
    t_min=np.amin(np.amin(terrain1))
    print(np.shape(terrain1))
    print(t_min,t_max)
    terrain1=terrain1-t_min
    terrain1=terrain1*1.0/(t_max-t_min)

    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain1, cmap='gray',vmin=0.0,vmax=1.0)
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.show()
    plt.savefig('ter1_full.png')
    plt.clf()
    
    #reduce terrain
    shape_ter=np.shape(terrain1)
    nx=shape_ter[1]
    ny=shape_ter[0]
    res=50
    deg=40
    res2=res**2
    nxr=nx//res
    nyr=ny//res
    n2=nxr*nyr
    nmax=max(nxr,nyr)
    x=(np.arange(0,nxr)-nxr//2)/nmax
    y=(np.arange(0,nyr)-nyr//2)/nmax
    x3d, y3d = np.meshgrid(x,y)
    ter_res=np.zeros(shape=(nyr,nxr))
    print('rescale')
    for i in range(nyr):
        for j in range(nxr):
            ter_res[i,j]=np.sum(terrain1[i*res:(i+1)*res,j*res:(j+1)*res])/res2

    tr_max=np.amax(np.amax(ter_res))
    tr_min=np.amin(np.amin(ter_res))
    print(tr_min,tr_max)
    print(np.sum(ter_res)/n2)
    plt.figure(1)
    plt.title('Terrain over Norway 1')
    plt.imshow(ter_res, cmap='gray',vmin=0.0,vmax=1.0)
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
    print('split')
    xk,yk,tk,nk=split_data_kfold(x_ter,y_ter,t_ter,k) #k=number of data groups
    l_vec=[0.0,1e-10,1e-8,1e-4]
    n_l=len(l_vec)
    d_vec=np.arange(10,50,5,dtype='int')
    n_d=len(d_vec)
    MSEs=np.zeros(shape=(n_l,n_d,2))

    for i in range(n_l):
        lamb=l_vec[i]
        for j in range(n_d):
            deg=d_vec[j]
            print('fit lambda %.2e, deg %i'%(lamb,deg))
            msek,r2k,betak,var_bk=polfit_kfold(xk,yk,tk,nk,k,n2,deg=deg,lamb=lamb)
            beta=np.mean(betak,axis=1)
            MSEs[i,j,:]=np.mean(msek,axis=0)
            print('evaluate')
            ter_fit=eval_terrain(beta,x,y,deg,nxr,nyr)
            #ter_fit_3d=eval_pol3D(beta,x3d,y3d,deg)

            print('plot')
            plt.figure(1)
            lstr,powstr=get_pow_str(lamb,1)   
            plt.title(r'Terrain fit, $\lambda$ = %s$\cdot 10^{%s}$, pol. deg. %i over Norway 1'%(lstr,powstr,deg))
            plt.imshow(ter_fit, cmap='gray',vmin=0.0,vmax=1.0)
            plt.xlabel('X')
            plt.ylabel('Y')
            #plt.show()
            outfile='ter_fit_lamb_%.1e_deg_%i'%(lamb,deg)+fig_format
            print(outfile)
            plt.savefig('figs/'+outfile)
            plt.clf()


    plt.figure(1)
    for i in range(n_l):
        lstr,powstr=get_pow_str(l_vec[i],1)   
        plt.plot(d_vec,MSEs[i,:,0],ls='-',marker='.',label='$\lambda$ = %s$\cdot 10^{%s}$'%(lstr,powstr))
        plt.plot(d_vec,MSEs[i,:,1],ls='--',marker='.')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Mean Square Error')
    outfile='ter_mse'+fig_format
    plt.savefig('figs/'+outfile)
    plt.clf()

    if (True):
        return

    #Other plotting (3D)
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x3d, y3d, ter_fit_3d, cmap=cm.gray,linewidth=0, antialiased=False, alpha=1.0)
    # Customize the z axis.
    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(elev=10., azim=65.)
    # Add a color bar which maps values to colors.
    plt.xlabel('y',fontsize=14)
    plt.ylabel('x',fontsize=14,labelpad=10)
    plt.yticks(rotation=45)
    #plt.show()
    plt.savefig('ter1_fit_3d.png')
    plt.clf()

    return
