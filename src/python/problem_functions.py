import numpy as np
import matplotlib.pyplot as plt
from params import *
from linreg_functions import *

######################################
# Functions for performing the tasks #
# given in the problem set           #
######################################

#Plot number of gridpoints trade off
def run_tradeoff_number():
    rnd=[False,True]
    k=4
    l_vec=[0.0, 1e-6,1e-4, 1e-2, 1.0]
    for i in range(2):
        #ensure the same splits, noise etc for rideg and lasso
        st0=np.random.get_state()
        mse_plot_tradeoff_number(rnd[i],p=5,k=k,n_max=21,lamb=l_vec)
        np.random.set_state(st0)
        mse_plot_tradeoff_number(rnd[i],p=5,k=k,n_max=21,lamb=l_vec,lasso=True)
    return
            
#plot kfold tradeoff (group size/number of groups)
def run_tradeoff_kfold():
    n_vec=np.zeros(3,dtype='int')
    n_vec[0]=10
    n_vec[1:]=21
    rnd=[False,False,True]
    for i in range (3):
        mse_plot_tradeoff_kfold(n_vec[i],rnd[i],p=5)
        mse_plot_tradeoff_kfold(n_vec[i],rnd[i],p=5,lamb=1e-3)
    return

#plot lambda tradeoff
def run_tradeoff_lambda():
    k=5
    rnd=[False,True]
    n_vec=[10,21]
    for i in range (1):
        for j in range(2):
            n2=n_vec[j]**2
            x_vec,y_vec=init_xy_vectors(n_vec[j],rnd[i])
            #calculate franke function values
            f_vec=FrankeFunction(x_vec,y_vec)
            #add noise
            if (sigma > 0.0):
                noise=np.random.normal(0.0,1.0,n2)*sigma
                f_vec[:,0]=f_vec[:,0]+noise

            #split data
            xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)

            mse_plot_tradeoff_lambda(n_vec[j],rnd[i],lamb_min=1e-6,lamb_max=10.0,k=k,p=7,plt_lines=True,data_input=True,xv=xk,yv=yk,fv=fk,nv=nk)
            mse_plot_tradeoff_lambda(n_vec[j],rnd[i],lamb_min=1e-6,lamb_max=10.0,k=k,p=5,data_input=True,xv=xk,yv=yk,fv=fk,nv=nk)
    return

# plot complexity tradeoff
def run_tradeoff_complexity():
    n=[10,21]
    rnd=[False,True]
    k=5
    l_vec=[0.0, 1e-6,1e-4, 1e-2, 1.0]
    for i in range(2):
        for j in range(1):
            mse_plot_tradeoff_complexity(k,n[j], rnd[i],lamb=l_vec, p_lim=14)
            mse_plot_tradeoff_complexity(k,n[j], rnd[i],lamb=l_vec, p_lim=14,lasso=True)
    return

# plot complexity tradeoff
def run_tradeoff_complexity_eqsplit():
    #use the same splits and noise for Lasso and Ridge
    n=[10,21]
    ylim=np.array([[1e-3,1e1],[4e-3,0.15]])
    rnd=[False,True]
    k=5
    leg_pos=['upper left','lower left']
    for i in range(2):
        for j in range(2):
            n2=n[j]**2
            x_vec,y_vec=init_xy_vectors(n[j],rnd[i])
            #calculate franke function values
            f_vec=FrankeFunction(x_vec,y_vec)
            #add noise
            if (sigma > 0.0):
                noise=np.random.normal(0.0,1.0,n2)*sigma
                f_vec[:,0]=f_vec[:,0]+noise

            #split data
            xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)
            l_vec=np.array([0.0, 1e-6,1e-4, 1e-2, 1.0])
            mse_plot_tradeoff_complexity(k,n[j],rnd[i],lamb=l_vec,p_lim=14,xv=xk,yv=yk,fv=fk,nv=nk,data_input=True,ylim=ylim[j,:],leg_pos=leg_pos[j])
            l_vec=np.array([0.0, 1e-8, 1e-6,1e-4, 1e-2, 1.0])
            mse_plot_tradeoff_complexity(k,n[j],rnd[i],lamb=l_vec,p_lim=14,lasso=True,xv=xk,yv=yk,fv=fk,nv=nk,data_input=True,ylim=ylim[j,:],leg_pos=leg_pos[j])
    return

# test of OLS, Ridge and Lasso on same data
def run_model_comp():
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

    return


    
# Running the fits using OLS (part a and b)
def part_ab():
    n=21
    n2=n**2
    f_vec=np.zeros(shape=(n2,1))

    x_vec,y_vec=init_xy_vectors(n,False)

    #calculate franke function values
    f_vec=FrankeFunction(x_vec,y_vec)
    #add noise
    if (sigma > 0.0):
        noise=np.random.normal(0.0,1.0,n2)*sigma
        f_vec[:,0]=f_vec[:,0]+noise
    

    #run the polynomial fitting for 0th-5th grade polynomials
    for p in range(6):
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
    for p in range(6):
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


    #k-fold CV on OLS
    if (True):
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

    #LOOCV on OLS
    if (True):
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

    return




#####################################
# Functions doing the fits and      #
# plotting the results              #
#####################################

def mse_plot_tradeoff_complexity(k,n, rnd,xmin=0.0,xmax=1.0, p_lim=-1,lamb=[0.0],lasso=False, single=False,data_input=False,xv=0.0,yv=0.0,fv=0.0,nv=-1,ylim=[0.0],leg_pos='lower left'):
    global verbosity
    global sigma
    # k = number of groups (i.e. k-fold CV)
    # n_vec = number of gridpoints in x and y direction
    # rnd = True/False; if True, then points are drawn from a uniform distribution
    #                   Draws n**2 points (x,y)
    # xmin,xmax = minimum and maximum values for the grid (in both x and y direction)
    # p_lim = upper limit of the polynomial degree (i.e. complexity)

    n_l=len(lamb)
    if (n_l > 1):
        l_vec=np.zeros(n_l)
        for i in range(n_l):
            l_vec[i]=lamb[i]
            
    if(verbosity > 0):
        print('')
        print('######################################')
        print('')
        print(' xy-grid NxN, N = %i'%(n))
        print('')
        print('----------------------------------')

    n2=n**2
    if (data_input): # In order to use the same data (and splits) 
        xk=xv
        yk=yv
        fk=fv
        nk=nv
    else:
        # init x and y points, both in equidistant grids or as random points
        x_vec,y_vec=init_xy_vectors(n,rnd,xmin=xmin,xmax=xmax)
        #calculate franke function values
        f_vec=FrankeFunction(x_vec,y_vec)
        #add noise
        if (sigma > 0.0):
            noise=np.random.normal(0.0,1.0,n2)*sigma
            f_vec[:,0]=f_vec[:,0]+noise

        #split data
        xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)

    # run over polynomial degrees from 0 to the maximal number given that
    # the matrices are invertable
    # We need at least N = n_p = (p+1)*(p+2)/2 different points
    # 2*N = p**2 + 3p + 2
    # p**2 + 3p - 2(N-1) = 0
    # I.e. for any N, p must be smaller than the positive solution to the equation above
    # p = (-3 + sqrt(3**2 +8(N-1) )/2
    # for N = 7, p = 2.275, i.e. p_max = 2 --> n_p = 6
        
    N_data=n2-nk[0] #first fold (data set) always contains the most data points
    p_max = int((-3.0 + np.sqrt(9.0+8.0*(N_data-1)))/2.0)
    if (p_lim>0):
        p_max=min(p_lim,p_max)
    n_deg=p_max+1

    mse_mean=np.zeros(shape=(n_deg,2,n_l))
    mse_error=np.zeros(shape=(3,n_deg,2,n_l)) #(error type, complexity, train/test)
    # error type: (0) lower MSE, (1) higher MSE, (2) std of mean MSE
    # train/test: (0) train, (1) test
    p_plot=np.zeros(shape=(n_deg))

    for i_l in range (n_l):

        lamb=l_vec[i_l]
        if(verbosity > 0):
            print('')
            print(' lambda %.3e'%(lamb))

        for p in range(n_deg):
            if (lasso):
                if (lamb==0.0):
                    msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
                else:
                    msek,r2k,betak=kfold_CV_lasso(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
            else:
                msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
            mse_mean[p,0,i_l]=np.sum(msek[:,0])/k
            mse_mean[p,1,i_l]=np.sum(msek[:,1])/k
            mse_error[0,p,0,i_l]=mse_mean[p,0,i_l]-np.amin(msek[:,0])
            mse_error[1,p,0,i_l]=np.amax(msek[:,0])-mse_mean[p,0,i_l]
            mse_error[2,p,0,i_l]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,0]-mse_mean[p,0,i_l])**2))
            mse_error[0,p,1,i_l]=mse_mean[p,1,i_l]-np.amin(msek[:,1])
            mse_error[1,p,1,i_l]=np.amax(msek[:,1])-mse_mean[p,1,i_l]
            mse_error[2,p,1,i_l]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,1]-mse_mean[p,1,i_l])**2))
            
            p_plot[p]=p*1.0
            
            # print the different MSE values for each k-fold modelling
            if (verbosity > 2):
                print('')
                print('-----------------------------------------------------')
                if (lasso):
                    print('Lasso fit')
                print("polynomial of degree %i, lambda %.3e"%(p,lamb))
                print('')
                print('group    mse_tr       mse_te       r2_tr')
                for j in range(k):
                    print('  %i    %.4e   %.4e   %.4e '%(j,msek[j,0], msek[j,1],r2k[j,0]))
                print(' tot   %.4e   %.4e   '%(np.sum(msek[:,0])/k, np.sum(msek[:,1])/k))
                print('')
                if (p==2):
                    print(' train_err        %.4e   %.4e   '%(mse_error[0,p,0,i_l], mse_error[1,p,0,i_l]))
                    print(' test_err         %.4e   %.4e   '%(mse_error[0,p,1,i_l], mse_error[1,p,1,i_l]))
                    print(' std (train,test) %.4e   %.4e   '%(mse_error[2,p,0,i_l], mse_error[2,p,1,i_l]))
            elif(verbosity > 1):
                print('  pol. deg. %i, lamb = %.3e'%(p,lamb))
    
#    ylim=[np.exp(np.log(np.amin(np.amin(np.amin(mse_mean))))-0.2), 1.5]
    
    plot_mse(p_plot,mse_mean,mse_error,'Complexity (polynomial degree)','complex',rnd,n=n,l_vec=l_vec,lasso=lasso,logx=False,logy=True,ylim=ylim,single=False,leg_pos=leg_pos)
    plot_mse(p_plot,mse_mean,mse_error,'Complexity (polynomial degree)','complex',rnd,n=n,l_vec=l_vec,lasso=lasso,logx=False,logy=True,ylim=ylim,single=False,leg_pos=leg_pos,plt_lines=True)
    return 
    

def mse_plot_tradeoff_number(rnd,n_min=6,n_max=30, p=5,k=5,xmin=0.0,xmax=1.0,lamb=0.0,lasso=False):
    global verbosity
    global sigma
    # k = number of groups (i.e. k-fold CV)
    # rnd = True/False; if True, then points are drawn from a uniform distribution
    #                   Draws n**2 points (x,y)
    # nmin,nmax = minimum and maximum values for the number of grid points
    # p = the polynomial degree (i.e. complexity) of the fit
    # xmin,xmax = minimum and maximum values for the grid (in both x and y direction)
    
    # n_vec = number of gridpoints in x and y direction
    n_vec=np.arange(n_min,n_max+1,dtype='int')
    n_n=len(n_vec)

    n_l=len(lamb)
    if (n_l > 1):
        l_vec=np.zeros(n_l)
        for i in range(n_l):
            l_vec[i]=lamb[i]

    # MSE means and errors
    mse_mean=np.zeros(shape=(n_n,2,n_l))
    mse_error=np.zeros(shape=(3,n_n,2,n_l)) #(error type, grid points (n*n), train/test)
    # error type: (0) lower MSE, (1) higher MSE, (2) std of mean MSE
    # train/test: (0) train, (1) test
    n_plot=np.zeros(shape=(n_n))

    #Find the minimum number of data points in training group
    # each group has maximum n**2//k + 1 points
    # each training group has then at least n**2 - (n**2//k + 1) data points
    # n_min has the least total, so
    n_d = n_min**2 - (n_min**2//k + 1)
    
    p_max = int((-3.0 + np.sqrt(9.0+8.0*(n_d-1)))/2.0)
    if (p>p_max):
        p=p_max
        
    for i in range(len(n_vec)):
        n=np.copy(n_vec[i])
        n_plot[i]=n
        if(verbosity > 0):
            print('n = %i'%(n))

        # init x and y points, both in equidistant grids or as random points
        x_vec,y_vec=init_xy_vectors(n,rnd,xmin=xmin,xmax=xmax)
        n2=n**2
        #calculate franke function values
        f_vec=FrankeFunction(x_vec,y_vec)
        #add noise
        if (sigma > 0.0):
            noise=np.random.normal(0.0,1.0,n2)*sigma
            f_vec[:,0]=f_vec[:,0]+noise

        #split data
        xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)

        for i_l in range(n_l):
            lamb=l_vec[i_l]
            if (lasso):
                if (lamb==0.0):
                    msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
                else:
                    msek,r2k,betak=kfold_CV_lasso(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
            else:
                msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
            mse_mean[i,0,i_l]=np.sum(msek[:,0])/k
            mse_mean[i,1,i_l]=np.sum(msek[:,1])/k
            mse_error[0,i,0,i_l]=mse_mean[i,0,i_l]-np.amin(msek[:,0])
            mse_error[1,i,0,i_l]=np.amax(msek[:,0])-mse_mean[i,0,i_l]
            mse_error[2,i,0,i_l]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,0]-mse_mean[i,0,i_l])**2))
            mse_error[0,i,1,i_l]=mse_mean[i,1,i_l]-np.amin(msek[:,1])
            mse_error[1,i,1,i_l]=np.amax(msek[:,1])-mse_mean[i,1,i_l]
            mse_error[2,i,1,i_l]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,1]-mse_mean[i,1,i_l])**2))
            
            
    plot_mse(n_plot,mse_mean,mse_error,r'$n$','number',rnd,deg=p,l_vec=l_vec,lasso=lasso,logx=False,logy=True,ylim=[0.95e-3,2e-1],leg_pos='lower right')        
    plot_mse(n_plot,mse_mean,mse_error,r'$n$','number',rnd,deg=p,l_vec=l_vec,lasso=lasso,logx=False,logy=True,ylim=[0.95e-3,2e-1],leg_pos='lower right',plt_lines=True)        
    return


def mse_plot_tradeoff_kfold(n,rnd, k_min=2, k_max=10, p=5,xmin=0.0,xmax=1.0,lamb=0.0,n_s=10):
    global fig_format
    global verbosity
    global sigma
    # k = number of groups (i.e. k-fold CV)
    # rnd = True/False; if True, then points are drawn from a uniform distribution
    #                   Draws n**2 points (x,y)
    # nmin,nmax = minimum and maximum values for the number of grid points
    # p = the polynomial degree (i.e. complexity) of the fit
    # xmin,xmax = minimum and maximum values for the grid (in both x and y direction)
    
    # n = number of gridpoints in x and y direction

    k_vec = np.arange(k_min,k_max+1,dtype='int')
    n_k = len(k_vec)
    
    # MSE means and errors
    mse_mean=np.zeros(shape=(n_k,2))
    mse_error=np.zeros(shape=(3,n_k,2)) #(error type, grid points (n*n), train/test)
    # error type: (0) lower MSE, (1) higher MSE, (2) std of mean MSE
    # train/test: (0) train, (1) test
    k_plot=np.zeros(shape=(n_k))

    #Find the minimum number of data points in training group
    # each group has maximum n**2//k + 1 points
    # each training group has then at least n**2 - (n**2//k + 1) data points
    n2=n**2
    n_d = n2 - (n2//k_min + 1)
    
    p_max = int((-3.0 + np.sqrt(9.0+8.0*(n_d-1)))/2.0)
    if (p>p_max):
        p=p_max
        
    # init x and y points, both in equidistant grids or as random points
    x_vec,y_vec=init_xy_vectors(n,rnd,xmin=xmin,xmax=xmax)
    #calculate franke function values
    f_vec=FrankeFunction(x_vec,y_vec)
    #add noise
    if (sigma > 0.0):
        noise=np.random.normal(0.0,1.0,n2)*sigma
        f_vec[:,0]=f_vec[:,0]+noise

    for i in range(n_k):
        k=k_vec[i]
        k_plot[i]=k
        if(verbosity > 0):
            print(' k = %i'%(k))

        #we want to test for different splits as well, so that we get a broader feel of
        #the spread in mse
        msek_s=np.zeros(shape=(k,2,n_s))
        for j in range(n_s):
            if(verbosity > 0):
                print('   split %i of %i, k %i of %i'%(j+1,n_s,i+1,n_k))
            #split data
            xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)
        
            msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
            msek_s[:,:,j]=msek
            mse_mean[i,0]+=np.sum(msek[:,0])/k
            mse_mean[i,1]+=np.sum(msek[:,1])/k
            if (j==0):
                mse_error[0,i,0]=np.amin(msek[:,0])
                mse_error[0,i,1]=np.amin(msek[:,1])
            else:
                mse_error[0,i,0]=min(np.amin(msek[:,0]),mse_error[0,i,0])
                mse_error[0,i,1]=min(np.amin(msek[:,1]),mse_error[0,i,1])
            mse_error[1,i,0]=max(np.amax(msek[:,0]),mse_error[1,i,0])
            mse_error[1,i,1]=max(np.amax(msek[:,1]),mse_error[1,i,1])
            
            
        mse_mean[i,:]=mse_mean[i,:]/n_s
        sqerr=np.zeros(2)
        for j in range(n_s):
            sqerr[0]+=np.sum((msek_s[:,0,j]-mse_mean[i,0])**2)
            sqerr[1]+=np.sum((msek_s[:,1,j]-mse_mean[i,1])**2)
        mse_error[2,i,:]=np.sqrt(1.0/(k*n_s-1.0)*sqerr)
        mse_error[0,i,:]=mse_mean[i,:]-mse_error[0,i,:]    
        mse_error[1,i,:]=mse_error[1,i,:]-mse_mean[i,:]

    #plot the MSE to number of gridpoints plot plot
    for m in range(2):
        plt.figure(1)
        if (m==0):
            plt.errorbar(k_plot,mse_mean[:,0],yerr=mse_error[:2,:,0],label='Training', fmt='.')
            plt.errorbar(k_plot+0.1,mse_mean[:,1],yerr=mse_error[:2,:,1],label='Test',fmt='x')
        else:
            plt.errorbar(k_plot,mse_mean[:,0],yerr=mse_error[2,:,0],label='Training',fmt='.')
            plt.errorbar(k_plot+0.1,mse_mean[:,1],yerr=mse_error[2,:,1],label='Test',fmt='x')
                
        plt.legend()
        plt.xlabel('Number of groups (k)')
        plt.ylabel('Mean Square Error')
        #if (n<15):
        #    plt.yscale('log')

        outfile='tradeoff_kfold'
        if (lamb>0.0):
            outfile=outfile+'_ridge_lamb_%.1e'%(lamb)
        if (rnd):
            plt.title('Random grid points')
            outfile=outfile+'_rnd_n%i'%(n)
        else:
            plt.title('Equidistant grid points')
            outfile=outfile+'_n%i'%(n)
        if (m==0):
            outfile=outfile+'_minmax'+fig_format
        else:
            outfile=outfile+'_std'+fig_format

        plt.savefig('figs/'+outfile)

                
        #plt.show()
        plt.clf()
    return

def mse_plot_tradeoff_lambda(n,rnd,lamb_min=0.1,lamb_max=1.0,n_lamb=10, log_lamb=True, k=5, p=5,xmin=0.0,xmax=1.0,plt_title=False,plt_lines=False,data_input=False,xv=0.0,yv=0.0,fv=0.0,nv=-1):
    global verbosity
    global sigma
    global fig_format
    # n = number of gridpoints in x and y direction
    # k = number of groups (i.e. k-fold CV)
    # rnd = True/False; if True, then points are drawn from a uniform distribution
    #                   Draws n**2 points (x,y)
    # lamb_min,lamb_max = minimum and maximum values for lambda in Ridge regression
    # n_lamb is the number of lambda values to test for
    # log_lamb = distribute lambda on logarithmic scale
    # p = the polynomial degree (i.e. complexity) of the fit
    # xmin,xmax = minimum and maximum values for the grid (in both x and y direction)
    

    lamb_vec = np.zeros(n_lamb)
    if (log_lamb):
        lamb_min=np.log(lamb_min)
        lamb_max=np.log(lamb_max)
        
    dl=(lamb_max-lamb_min)/(n_lamb-1.0)
    for i in range(n_lamb):
        lamb_vec[i]=lamb_min+i*dl
    if (log_lamb):
        lamb_vec=np.exp(lamb_vec)
        

    #Find the minimum number of data points in training group
    # each group has maximum n**2//k + 1 points
    # each training group has then at least n**2 - (n**2//k + 1) data points
    n2=n**2
    n_d = n2 - (n2//k + 1)
    
    p_max = int((-3.0 + np.sqrt(9.0+8.0*(n_d-1)))/2.0)
    if (p>p_max):
        p=p_max
        
    # MSE means and errors
    mse_mean=np.zeros(shape=(n_lamb,2,p+1))
    mse_error=np.zeros(shape=(3,n_lamb,2,p+1)) #(error type, grid points (n*n), train/test)
    # error type: (0) lower MSE, (1) higher MSE, (2) std of mean MSE
    # train/test: (0) train, (1) test
    l_plot=np.zeros(shape=(n_lamb))

    if (data_input): # In order to use the same data (and splits) 
        xk=xv
        yk=yv
        fk=fv
        nk=nv
    else:
        # init x and y points, both in equidistant grids or as random points
        x_vec,y_vec=init_xy_vectors(n,rnd,xmin=xmin,xmax=xmax)
        #calculate franke function values
        f_vec=FrankeFunction(x_vec,y_vec)
        #add noise
        if (sigma > 0.0):
            noise=np.random.normal(0.0,1.0,n2)*sigma
            f_vec[:,0]=f_vec[:,0]+noise

        #split data
        xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)

    for p_i in range(1,p+1):
        for i in range(n_lamb):
            lamb=lamb_vec[i]
            l_plot[i]=lamb
    
            msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p_i,lamb=lamb)
            mse_mean[i,0,p_i]=np.sum(msek[:,0])/k
            mse_mean[i,1,p_i]=np.sum(msek[:,1])/k
            mse_error[0,i,0,p_i]=mse_mean[i,0,p_i]-np.amin(msek[:,0])
            mse_error[1,i,0,p_i]=np.amax(msek[:,0])-mse_mean[i,0,p_i]
            mse_error[2,i,0,p_i]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,0]-mse_mean[i,0,p_i])**2))
            mse_error[0,i,1,p_i]=mse_mean[i,1,p_i]-np.amin(msek[:,1])
            mse_error[1,i,1,p_i]=np.amax(msek[:,1])-mse_mean[i,1,p_i]
            mse_error[2,i,1,p_i]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,1]-mse_mean[i,1,p_i])**2))
                
            if (verbosity > 0):
                print(' p = %i, lambda = %.3e'%(p_i,lamb))

    #plot the MSE to lambda for pol. degree p
    for m in range(2):
        plt.figure(1)
        if (log_lamb): #adding a shift to test data
            l_plot2=np.exp(np.log(l_plot)+0.1*dl)
        else:
            l_plot2=l_plot+0.1*dl
            
        if (m==0):
            plt.errorbar(l_plot,mse_mean[:,0,p],yerr=mse_error[:2,:,0,p],label='Training', fmt='.')
            plt.errorbar(l_plot2,mse_mean[:,1,p],yerr=mse_error[:2,:,1,p],label='Test',fmt='x')
            
        else:
            plt.errorbar(l_plot,mse_mean[:,0,p],yerr=mse_error[2,:,0,p],label='Training',fmt='.')
            plt.errorbar(l_plot2,mse_mean[:,1,p],yerr=mse_error[2,:,1,p],label='Test',fmt='x')
                
        plt.legend()
        plt.xlabel('Lambda value')
        plt.ylabel('Mean Square Error')
        #plt.yscale('log')
        outfile='tradeoff_lamb'
        if (log_lamb):
            plt.xscale('log')
            outfile=outfile+'_log'
        outfile=outfile+'_p%i'%(p)
        if (rnd):
            if (plt_title):
                plt.title('Random grid points')
            outfile=outfile+'_rnd_n%i'%(n)
        else:
            if (plt_title):
                plt.title('Equidistant grid points')
            outfile=outfile+'_n%i'%(n)
        if (m==0):
            outfile=outfile+'_minmax'+fig_format
        else:
            outfile=outfile+'_std'+fig_format

        plt.savefig('figs/'+outfile)

                
        #plt.show()
        plt.clf()
    #plot the MSE to lambda for pol. degree p
    for m in range(2):
        n_s=2
        if (plt_lines):
            n_s=3
            cols=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for s in range(n_s):
            if (s == 2):
                si=0
            else:
                si=s
            plt.figure(1)
            if (plt_lines):
                if (m==1):
                    plt.clf()
                    return
                if (s==2):
                    for p_i in range(1,p+1):
                        plt.plot(l_plot,mse_mean[:,1,p_i],ls='--',marker='.',color=cols[p_i-1])
            for p_i in range(1,p+1):
                if (log_lamb): #adding a shift to test data
                    l_plot2=np.exp(np.log(l_plot)+(p_i-p//2)*0.05*dl)
                else:
                    l_plot2=l_plot+(p_i-p//2)*0.05*dl

                if (plt_lines):
                    plt.plot(l_plot,mse_mean[:,si,p_i],ls='-',marker='.',label='pol %i'%(p_i),color=cols[p_i-1])
                else:
                    if (m==0):
                        plt.errorbar(l_plot2,mse_mean[:,si,p_i],yerr=mse_error[:2,:,si,p_i],label='pol %i'%(p_i), fmt='.')
                    else:
                        plt.errorbar(l_plot2,mse_mean[:,si,p_i],yerr=mse_error[2,:,si,p_i],label='pol %i'%(p_i), fmt='.')
                
            
            plt.legend()
            plt.xlabel(r'$\lambda$', fontsize=14)
            plt.ylabel('Mean Square Error', fontsize=14)
            #plt.yscale('log')
            outfile='tradeoff_lamb_pol'
            if (log_lamb):
                plt.xscale('log')
                outfile=outfile+'_log'
            if (rnd):
                if (plt_title):
                    plt.title('Random grid points')
                outfile=outfile+'_rnd_n%i'%(n)
            else:
                if (plt_title):
                    plt.title('Equidistant grid points')
                outfile=outfile+'_n%i'%(n)

            if (s==0):
                outfile=outfile+'_train'
            elif (s==1):
                outfile=outfile+'_test'
            else:
                outfile=outfile+'_all'
            if (plt_lines):
                outfile=outfile+'_lines'+fig_format
            else:
                if (m==0):
                    outfile=outfile+'_minmax'+fig_format
                else:
                    outfile=outfile+'_std'+fig_format

            plt.savefig('figs/'+outfile)

                
            #plt.show()
            plt.clf()


    return



def plot_mse(x_plot,mse_mean,mse_error,xlab,lab,rnd,n=0,deg=-1,ylim=[0.0],l_vec=[0.0],lasso=False,logx=False,logy=False,single=False,plt_lines=False,plt_title=False,leg_pos=''):

    global fig_format
    
    shape_mse=np.shape(mse_mean)
    n_l=shape_mse[2]
    for i_l in range(n_l):
        for m in range(2):
            if ((not n_l==1) and (not single)):
                break
            plt.figure(1)
            lamb=l_vec[i_l]
            if (m==0):
                plt.errorbar(x_plot,mse_mean[:,0,i_l],yerr=mse_error[:2,:,0,i_l],label='Training', fmt='.')
                if (logx):
                    plt.errorbar(np.exp(np.log(x_plot)+0.1),mse_mean[:,1,i_l],yerr=mse_error[:2,:,1,i_l],label='Test',fmt='x')
                else:
                    plt.errorbar(x_plot+0.1,mse_mean[:,1,i_l],yerr=mse_error[:2,:,1,i_l],label='Test',fmt='x')

            else:
                plt.errorbar(x_plot,mse_mean[:,0,i_l],yerr=mse_error[2,:,0,i_l],label='Training',fmt='.')
                if (logx):
                    plt.errorbar(np.exp(np.log(x_plot)+0.1),mse_mean[:,1,i_l],yerr=mse_error[2,:,1,i_l],label='Test',fmt='x')
                else:
                    plt.errorbar(x_plot+0.1,mse_mean[:,1,i_l],yerr=mse_error[2,:,1,i_l],label='test',fmt='x')
                
                
            plt.legend()
            plt.xlabel(xlab)
            plt.ylabel('Mean Square Error')
            if (not ylim[0]==0.0):
                plt.ylim(ylim)
            if (logy):
                plt.yscale('log')
            outfile='tradeoff_'+lab
            if (lasso):
                outfile=outfile+'_lasso'
            else:
                outfile=outfile+'_ridge'
            if (lamb>0.0):
                outfile=outfile+'_lamb_%.3e'%(lamb)
            if (plt_title):
                if (rnd):
                    plt.title('Random grid points')
                else:
                    plt.title('Equidistant grid points')
            if (not n==0):
                outfile=outfile+'_n%i'%(n)
            if (not deg==-1):
                outfile=outfile+'_pol%i'%(deg)
            if (m==0):
                outfile=outfile+'_minmax'+fig_format
            else:
                outfile=outfile+'_std'+fig_format

            plt.savefig('figs/'+outfile)
            #plt.show()
            plt.clf()

    if (n_l ==1):
        return # only plot multi lambda if we have more than 1 value
    for m in range(2):
        if (plt_lines):
            n_s=3
            cols=plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            n_s=2
            
        for s in range(n_s):
            if (s==2):
                si=0
            else:
                si=s
                
            plt.figure(1)
            if (plt_lines):
                if (m==1):
                    plt.clf()
                    return
                if (s==2):
                    for i_l in range(n_l):
                        plt.plot(x_plot,mse_mean[:,1,i_l],ls='--',marker='.',color=cols[i_l])
            for i_l in range(n_l):
                lstr='%.1e'%(l_vec[i_l])
                a=lstr.split('e')
                lstr=a[0]
                powstr=a[1]
                if(powstr[1]=='0'):
                    powstr=powstr[0]+powstr[2]
                if(powstr[0]=='+'):
                    powstr=powstr[1:]
                if (logx):
                    xi_plot=np.exp(np.log(x_plot)+(-n_l//2 + i_l)*0.1)
                else:
                    xi_plot=x_plot+(-n_l//2 + i_l)*0.1
                if (plt_lines):
                    if (lstr=='0.0'):
                        plt.plot(x_plot,mse_mean[:,si,i_l],ls='-',marker='.',label='OLS',color=cols[i_l])
                    else:
                        plt.plot(x_plot,mse_mean[:,si,i_l],ls='-',marker='.',label=r'$\lambda = %s \cdot 10^{%s}$'%(lstr,powstr),color=cols[i_l])
                else:
                    if (m==0):
                        if (lstr=='0.0'):
                            plt.errorbar(xi_plot,mse_mean[:,si,i_l],yerr=mse_error[:2,:,si,i_l],label='OLS', fmt='.')
                        else:
                            plt.errorbar(xi_plot,mse_mean[:,si,i_l],yerr=mse_error[:2,:,si,i_l],label=r'$\lambda = %s \cdot 10^{%s}$'%(lstr,powstr), fmt='.')
                    else:
                        if (lstr=='0.0'):
                            plt.errorbar(xi_plot,mse_mean[:,si,i_l],yerr=mse_error[2,:,si,i_l],label='OLS', fmt='.')
                        else:
                            plt.errorbar(xi_plot,mse_mean[:,si,i_l],yerr=mse_error[2,:,si,i_l],label=r'$\lambda = %s \cdot 10^{%s}$'%(lstr,powstr), fmt='.')

            if (not leg_pos == ''):
                plt.legend(loc=leg_pos)
            else:
                plt.legend()
            plt.xlabel(xlab,fontsize=14)
            plt.ylabel('Mean Square Error',fontsize=14)
            if (not ylim[0]==0.0):
                plt.ylim(ylim)
            if (logy):
                plt.yscale('log')
            if (lasso):
                outfile='tradeoff_'+lab+'_lasso_lamb_multi'
            else:
                outfile='tradeoff_'+lab+'_ridge_lamb_multi'
            if (rnd):
                if (plt_title):
                    plt.title('Random grid points')
                outfile=outfile+'_rnd'
            else:
                if (plt_title):
                    plt.title('Equidistant grid points')
                
            if (not n==0):
                outfile=outfile+'_n%i'%(n)
            if (not deg==-1):
                outfile=outfile+'_pol%i'%(deg)
            if (s==0):
                outfile=outfile+'_train'
            elif (s==1):
                outfile=outfile+'_test'
            else:
                if (s<2):
                    outfile=outfile+'_lines'
                else:
                    outfile=outfile+'all_lines'

            if (not plt_lines):
                if (m==0):
                    outfile=outfile+'_minmax'
                else:
                    outfile=outfile+'_std'
            outfile+=fig_format

            plt.savefig('figs/'+outfile)
            #plt.show()
            plt.clf()

    return
