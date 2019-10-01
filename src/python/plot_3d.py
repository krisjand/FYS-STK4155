
from mpl_toolkits.mplot3d import Axes3D
from setup_matplotlib import *
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
    if (verbosity>0):
        print('')        
        print('#############################################')
        print('')
        print('Plotting Franke function without noise')

    plot_surf(x,y,z,colbar=True)    
    plot_surf(x,y,z)
    if (sigma>0.0):
        noise=np.random.normal(0.0,sigma,size=(n,n))
        z=z+noise

    xv,yv,fv=init_xy_vectors(n,rand,rearr=True,x=x,y=y,z=z)
    
    
    
    if (False): #check the variance terms of beta vs. the equation from lecture notes
                #  Var[beta_j]=sigma^2 * sqrt( ((X.T@X)^-1)_jj )
                # Why, when Var[beta] = sigma^2 * (X.T@X)^-1  ??!
        if (verbosity>0):
            print('Variance check')
        deg=5
        n_p=(deg+1)*(deg+2)//2
        k=4
        m=100
        beta_mean=np.zeros(n_p)
        betas=np.zeros(shape=(n_p,m*k))
        bv_calc=np.zeros(n_p)
        bv_sum=np.zeros(n_p)
        for i in range(m):
            if (verbosity>0):
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
            if (verbosity>0):
                print(bv_calc[i],bv_sum[i],b_std[i])
            plt.figure(1)
            plt.plot(betas[i,:],'.')
            plt.plot([0,m*k],[beta_mean[i],beta_mean[i]])
            plt.show()
            plt.clf()
        
    
        b_cl95=b_std*1.96
        b_cl99=b_std*2.58
        if (verbosity>0):
            print(b_std)
        exit()

    #Plot the surface
    if (verbosity>0):
        print('Plotting Franke function with noise')    
    plot_surf(x,y,z,noise=True)    
    plot_surf(x,y,z,noise=True,colbar=True)    

    # Do a 4-fold CV using OLS and complexities ranging from 0 to 5th polynomial
    k=4
    xk,yk,fk,nk=split_data_kfold(xv,yv,fv,k)
    for deg in range(6):
        if (verbosity>0):
            print('Plotting OLS, polynomial degree %i'%deg)    
        mse,r2,betak,bv=polfit_kfold(xk,yk,fk,nk,k,n2,deg=deg,lamb=0.0)
        beta=np.mean(betak,axis=1)
            
        zfit=eval_pol3D(beta,x,y,deg)

        plot_surf(x,y,z,zfit=zfit,model='ols',deg=deg,lamb=1e-4,noise=True)

    #Do a Ridge regression for chosen lambda values for 5th degree polynomial fit
    lamb=[1.0,1e-2,1e-4,1e-6]
    deg=5
    for i in range(len(lamb)):
        if (verbosity>0):
            print('Plotting Ridge, lambda %.2e'%lamb[i])            
        mse,r2,betak,bv=polfit_kfold(xk,yk,fk,nk,k,n2,deg=deg,lamb=lamb[i])
        beta=np.mean(betak,axis=1)
            
        zfit=eval_pol3D(beta,x,y,deg)

        plot_surf(x,y,z,zfit=zfit,model='ridge',deg=deg,lamb=lamb[i],noise=True)

    #Do a Lasso regression for chosen lambda values for 5th degree polynomial fit
    lamb=[1.0,1e-2,1e-4,1e-6]
    for i in range(len(lamb)):
        if (verbosity>0):
            print('Plotting Lasso, lambda %.2e'%lamb[i])            
        
        mse,r2,betak=kfold_CV_lasso(xk,yk,fk,nk,k,n2,deg=deg,lamb=lamb[i])
        beta=np.mean(betak,axis=1)
            
        zfit=eval_pol3D(beta,x,y,deg)

        plot_surf(x,y,z,zfit=zfit,model='lasso',deg=deg,lamb=lamb[i],noise=True)

    return








def plot_surf(x,y,z,zfit=0.0,model='none',deg=-1,lamb=0.0,noise=False,colbar=False):
    # Plot the surface.
    global fig_format
    
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.6)
    if (not model=='none'):
        ax.scatter(x,y,zfit,marker='.',s=1.,color='r')

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(elev=10., azim=65.)
    # Add a color bar which maps values to colors.
    if (colbar):
        fig.colorbar(surf, shrink=0.5, aspect=5)

    if (lamb==0.0):
        lamb_str='0'
    else:
        lamb_str='%.2e'%(lamb)
        a=lamb_str.split('e')
        power=a[1]
        lamb_str=a[0]
        if (power[0]=='-'):
            sign='-'
        else:
            sign=''
        power=power[1:]
        if (power[0]=='0'):
            power=power[1:]
        

            
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14,labelpad=10)
    plt.yticks(rotation=45)
    if (model=='none'):
        plt.title('Franke function')
        filename='franke_function'
        if (noise):
            filename+='_noise'
        if (colbar):
            filename+='_cbar'
        filename+=fig_format
    elif (model=='ols'):
        plt.title(r'OLS, $p=$ %i'%(deg))
        filename='ols'
        if (noise):
            filename+='_noise'
        if (colbar):
            filename+='_cbar'
        filename+='_p%i'%(deg)+fig_format
    elif (model=='ridge'):
        plt.title(r'Ridge, $\lambda = %s \cdot 10^{%s}$'%(lamb_str,sign+power))
        filename='ridge'
        if (noise):
            filename+='_noise'
        if (colbar):
            filename+='_cbar'
        filename+='_lamb_%.2e'%(lamb)+fig_format
    elif (model=='lasso'):
        plt.title(r'Lasso, $\lambda = %s \cdot 10^{%s}$'%(lamb_str,sign+power))
        filename='lasso'
        if (noise):
            filename+='_noise'
        if (colbar):
            filename+='_cbar'
        filename+='_lamb_%.2e'%(lamb)+fig_format
    else:
        plt.clf()
        return

    if (debug):
        plt.show()

    plt.savefig(filename)
    plt.clf()

    return
