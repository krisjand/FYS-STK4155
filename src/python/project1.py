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
from p1_functions import *
from plot_3d import *

n=21
n2=n**2

plot_3D(n)

f_vec=np.zeros(shape=(n2,1))

x_vec,y_vec=init_xy_vectors(n,False)

#calculate franke function values
f_vec=FrankeFunction(x_vec,y_vec)
#add noise
if (sigma > 0.0):
    noise=np.random.normal(0.0,1.0,n2)*sigma
    f_vec[:,0]=f_vec[:,0]+noise
    

#run the polynomial fitting for 0th-5th grade polynomials
for p in range(0):
    beta,mse,r2,beta_var = polfit(p,x_vec,y_vec,f_vec)

    print('param     beta            var         std     95% (+-1.96 std)  99% (+-2.58std)')
    for i in range(len(beta)):
        print('beta_%i   %2.4e    %2.4e   %2.4e    %2.4e    %2.4e'%(i,beta[i,0],beta_var[i,0],np.sqrt(beta_var[i,0]),1.96*np.sqrt(beta_var[i,0]),2.58*np.sqrt(beta_var[i,0])))
    print('')
    print('MSE')
    print(mse)
    print('')
    print('R^2')
    print(r2)
    print('')
    print('')
    print('-----------------------------------------------------------------------------')

#Split data in training and test data (70% training)
ind_tr,ind_te=split_data(x_vec,70.0)

#resample betas for only the training data, then evaluate the MSE for the test data
for p in range(0):
    beta,mse,r2,beta_var = polfit(p,x_vec[ind_tr],y_vec[ind_tr],f_vec[ind_tr])
    if (verbosity > 1):
        for i in range(len(beta)):
            print('beta_%i   %2.4e    %2.4e   %2.4e    %2.4e    %2.4e'%(i,beta[i,0],beta_var[i,0],np.sqrt(beta_var[i,0]),1.96*np.sqrt(beta_var[i,0]),2.58*np.sqrt(beta_var[i,0])))
        print('')
        print('MSE')
        print(mse)
        print('')
        print('R^2')
        print(r2)
        print('')
        print('')
        print('-----------------------------------------------------------------------------')

    f_test=eval_pol(beta,x_vec[ind_te],y_vec[ind_te],p)
    shape_fte=np.shape(f_test)
    n_te=shape_fte[0]
    f_vec_te=np.zeros(shape=(n_te,1))
    f_vec_te[:,0]=f_vec[ind_te]
    mse=np.sum((f_vec_te-f_test)**2)
    mse=mse/n_te
    if (verbosity > 1):
        print(p,mse)
        print()


if (False):
    k=5
    xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)

    if (verbosity > 2):
        for i in range(k):
            print('')
            print('k %i'%(i))
            print('x             y                f')
            for j in range(nk[i]):
                print(xk[i,j],yk[i,j],fk[i,j])
    

    for p in range(6):
        msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,p)
        print('')
        print('-----------------------------------------------------')
        print("polynomial of degree %i"%(p))
        print('')
        print('group    mse_tr       mse_te       r2_tr       r2_te')
        for i in range(k):
            print('  %i    %.4e   %.4e   %.4e  %.4e'%(i,msek[i,0], msek[i,1],r2k[i,0],r2k[i,1]))

        print(' tot   %.4e   %.4e   '%(np.sum(msek[:,0])/k, np.sum(msek[:,1])/k))
        print('')

if (False):
    k=n2
    xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)
    print('')
    print('-----------------------')
    print('LOOCV')
    print('')
    print('degree   mse_tr       mse_te')
    for p in range(6):
        msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,n2,n2,p)

        print(' %i   %.4e   %.4e   '%(p,np.sum(msek[:,0])/n2, np.sum(msek[:,1])/n2))




if (False):
    rnd=[False,True]
    k=5
    l_vec=[0.0, 1e-6,1e-4, 1e-2, 1.0]
    for i in range(2):
        for j in range(1):
            mse_plot_tradeoff_number(rnd[i],p=5,k=k,n_max=21,lamb=l_vec)
            mse_plot_tradeoff_number(rnd[i],p=5,k=k,n_max=21,lamb=l_vec,lasso=True)

if (False):
    n_vec=np.zeros(4,dtype='int')
    n_vec[0:2]=10
    n_vec[2:]=21
    rnd=[False,True,False,True]
    for i in range (4):
        mse_plot_tradeoff_kfold(n_vec[i],rnd[i],p=5)

if (False):
    k=5
    rnd=[False,True]
    n_vec=[10,21]
    for i in range (2):
        for j in range(2):
            mse_plot_tradeoff_lambda(n_vec[j],rnd[i],lamb_min=1e-4,lamb_max=10.0,k=k,p=5)

if (False):
    n=[10,21]
    rnd=[False,True]
    k=5
    l_vec=[0.0, 1e-6,1e-4, 1e-2, 1.0]
    for i in range(2):
        for j in range(1):
            mse_plot_tradeoff_complexity(k,n[j], rnd[i],lamb=l_vec, p_lim=14)
            mse_plot_tradeoff_complexity(k,n[j], rnd[i],lamb=l_vec, p_lim=14,lasso=True)

            
if (False):
    n=21
    k=5
    xv,yv=init_xy_vectors(n,False)
    n2=n**2
    #calculate franke function values
    fv=FrankeFunction(xv,yv)
    #add noise
    if (sigma > 0.0):
        noise=np.random.normal(0.0,1.0,n2)*sigma
        fv[:,0]=fv[:,0]+noise

    #split data
    xk,yk,fk,nk=split_data_kfold(xv,yv,fv,k)

    print('MSE   training     test      solver')
    msek,r2k,beta_k,beta_var_k=polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.0)
    print('      %.4e     %.4e      OLS'%(np.sum(msek[:,0])/k,np.sum(msek[:,1])/k))
    msek,r2k,beta_k,beta_var_k=polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.01)
    print('      %.4e     %.4e      Ridge l=0.01'%(np.sum(msek[:,0])/k,np.sum(msek[:,1])/k))
    msek,r2k=kfold_CV_lasso(xk,yk,fk,nk,k,n2,deg=5,lamb=1e-4,max_iter=100000,tol=0.0000001)
    print('      %.4e     %.4e      Lasso l=1e-4'%(np.sum(msek[:,0])/k,np.sum(msek[:,1])/k))
