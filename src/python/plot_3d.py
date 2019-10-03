
from mpl_toolkits.mplot3d import Axes3D
from setup_matplotlib import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn import linear_model
#from random import random, seed
import time

from params import *
from linreg_functions import *


def plot_3D(n,lamb=[0.0],rand=False,var_check=False,add_new_noise=False):
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
    
    
    
    if (var_check): #check the variance terms of beta vs. the equation from lecture notes
                #  Var[beta_j]=sigma^2 * ((X.T@X)^-1)_jj 
        if (verbosity>0):
            print('Variance check')
        if (add_new_noise):#add different noise for each split. Requires no noise in base data
            fv0=np.copy(fv)
        deg=5
        n_p=(deg+1)*(deg+2)//2
        k=4
        m=100 #100 different splits
        beta_mean=np.zeros(n_p)
        betas=np.zeros(shape=(n_p,m*k))
        bvs=np.zeros(shape=(n_p,m*k))
        bv_calc=np.zeros(n_p)
        bv_std=np.zeros(n_p)
        bv_sum=np.zeros(n_p)
        for i in range(m):
            if (verbosity>0):
                print(i)
            if (add_new_noise):#add different noise for each split. Requires no noise in base data
                fv = fv0 + np.random.normal(0.0,sigma,size=(n**2,1))
            xk,yk,fk,nk=split_data_kfold(xv,yv,fv,k)
            mse,r2,betak,bv=polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.0)
            #get all beta values and estimated variances from all k-fold fits
            betas[:,i*k:(i+1)*k]=betak*1.0 
            bvs[:,i*k:(i+1)*k]=bv*1.0 
            bv_sum+=np.mean(bv,axis=1) #get the mean of the estimated var

        beta_mean=np.mean(betas,axis=1)
        bv_mean=bv_sum/m

        for i in range(m*k):
            for j in range(n_p):
                bv_calc[j]+=(betas[j,i]-beta_mean[j])**2
                bv_std[j]+=(bvs[j,i]-bv_mean[j])**2
        bv_calc=bv_calc/(m*k)
        bv_std=np.sqrt(bv_std/(m*k))
        b_std=np.sqrt(bv_calc)

        for i in range(n_p):
            if (verbosity>0):
                print(bv_calc[i],bv_mean[i])
            plot_betas(betas[i,:],beta_mean[i],b_std[i],i,type='beta')
        
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

    plt.savefig('figs/'+filename)
    plt.clf()

    return


def plot_betas(beta,b_mean,cal_std,nb=-1,eq_std=[-1.0],btype='none',ci=1.96):

    ci_cal=cal_std*ci
    if (eq_std[0] > 0.0):
        ci_eq=eq_std*ci
        
    nplt=len(beta)
    plt.figure(1)
    cols=plt.rcParams['axes.prop_cycle'].by_key()['color']

    

    if (btype=='all'):
        
        bc_std=np.array(shape=(2,nplt))
        bc_std[0,:]=beta-ci_cal
        bc_std[1,:]=beta+ci_cal
        if (eq_std[0] > 0.0):
            be_std=np.array(shape=(2,nplt))
            be_std[0,:]=beta-ci_eq
            be_std[1,:]=beta+ci_eq
        xplt=np.arange(0,nplt)
        mplt=xplt*1.0
    else:
        m_str,pow_str=get_pow_str(b_mean,3)            
        if (nb > -1):
            lab_m=r'<$\beta_%i$> = %s $\cdot$ $10^{%s}'%(nb,m_str,pow_str)
        else:
            lab_m='<'+r'$\beta$'+'>'
        
        b_m=np.array([b_mean,b_mean])
        bc_std=np.array([[b_mean-cal_std,b_mean-cal_std],[b_mean+cal_std,b_mean+cal_std]])
        if (eq_std[0] > 0.0):
            be_std=np.array([[b_mean-eq_std,b_mean-eq_std],[b_mean+eq_std,b_mean+eq_std]])
        xplt=np.arange(1,nplt+0.5)
        mplt=[1,nplt]

    plt.plot(xplt,beta,'.',color=cols[0])
    if (not btype=='all'):
        plt.plot(mplt,b_m,color=cols[1],label=lab_m) 
    plt.plot(mplt,bc_std[0,:],color=cols[2],label=lab_std)
    plt.plot(mplt,bc_std[1,:],color=cols[2])
    if (eq_std[0] > 0.0):
        plt.plot(mplt,be_std[0,:],color=cols[3],label=lab_std_eq)
        plt.plot(mplt,be_std[1,:],color=cols[3])


    plt.legend(loc='upper_right')
    #plt.show()
    if (btype=='beta'):
        outfile='beta'
    elif (btype=='var'):
        outfile='beta_var'
    elif (btype=='all'):
        outfile='beta_all'
    else:
        outfile='beta_check'

    if (eq_std[0] > 0.0):
        outfile+='_eq_comp'

    outfile+=fig_format
    plt.savefig('figs/'+outfile)
    plt.clf()


def get_pow_str(in_val,l):
        v='%.10e'%(inval)
        a=v.split('e')
        v=a[0]
        v=v[:2+l]
        s=a[1]
    if (s[1]=='0'):
        s=s[0]+s[2]
    if (s[0]=='+'):
        s=s[1:]
    return v,s
