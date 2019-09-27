import numpy as np
import matplotlib.pyplot as plt
from params import *

def init_xy_vectors(n,rand,xmin=0.0,xmax=1.0):
    
    if (rand):
        x_vec=np.random.uniform(xmin,xmax,size=(n**2,1))
        y_vec=np.random.uniform(xmin,xmax,size=(n**2,1))
    else:
        dx=np.zeros(1)
        dx=(xmax-xmin)/(n-1)
        n2=n**2
        x_vec=np.zeros(shape=(n**2,1))
        y_vec=np.zeros(shape=(n**2,1))
        for i in range(n):
            x_vec[n*i:n*(i+1),0]=xmin + i*dx
            y_vec[i:n2:n,0]=xmin + i*dx
    return x_vec,y_vec

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def polfit(deg,xv,yv,fv,lamb=0.0):
    n_p = (deg+1)*(deg+2)//2 #triangular rule but we have defined deg=0 for n=1
    shape_x=np.shape(xv)
    n2=shape_x[0]
    beta=np.zeros(shape=(n_p,1))


    if (len(shape_x)<2):
        temp=np.copy(xv)
        xv=np.zeros(shape=(n2,1))
        xv[:,0]=temp*1.0
    shape_y=np.shape(yv)
    n3=shape_y[0]
    if (len(shape_y)<2):
        temp=np.copy(yv)
        yv=np.zeros(shape=(n3,1))
        yv[:,0]=temp*1.0
    shape_f=np.shape(fv)
    n4=shape_f[0]
    if (len(shape_f)<2):
        temp=np.copy(fv)
        fv=np.zeros(shape=(n4,1))
        fv[:,0]=temp*1.0
    #build matrix

    p_min=0
    if (lamb>0.0): #Ridge reg. start at polynomials of 1st degree
        p_min=1
        beta[0,0]=np.sum(fv[:,0])/n2 #beta_0 is the average
    X=np.ones(shape=(n2,n_p)) #we change these values, but in the case of Ridge we don't
                              # have to change X[:,0] values for later evaluation
    if (deg > 0 or lamb==0.0):
        for i in range(n2):
            l=p_min-1
            for j in range(p_min,deg+1):
                for k in range(j+1):
                    l+=1
                    X[i,l]=xv[i,0]**(j-k) * yv[i,0]**k

        #If we use Ridge reg. we only use X[:,1:], i.e. X from polynom degree 1
        Xt = X[:,p_min:].T
        XtX = np.matmul(Xt,X[:,p_min:])
        if (lamb>0.0): #i.e. Ridge regression
            Xlamb=np.zeros(shape=(n_p-1,n_p-1)) #(n_p-1 x n_p-1 matrix)
            for i in range(n_p-1):
                Xlamb[i,i]=lamb
            XtX=XtX+Xlamb
            f_temp=fv-beta[0,0] #need to subtract away the average from the data
            Xtf = np.matmul(Xt,f_temp)
        else:
            Xtf = np.matmul(Xt,fv)
        XtXi = np.linalg.inv(XtX)
        beta[p_min:,:]=np.matmul(XtXi,Xtf)

    fm=np.mean(fv[:,0])
    fxy=np.matmul(X,beta)

    mse=np.sum((fv-fxy)**2)
    t1=np.copy(mse)
    mse=mse/(n2-1)
    
    t2=np.sum((fv-fm)**2)
    r2=1.0-t1/t2

    beta_var=np.zeros(shape=(n_p,1))
    for i in range(n_p):
        if (i==0 and lamb>0.0): #Ridge
            beta_var[i,0]=t2/(n2-1)**2 #variance of the calculated mean is Var(f_i)/(n-1)
                                       # Var(f_i)=sum((fv-fm)**2)/(n2-1)
        else:
            beta_var[i,0]=mse*np.sqrt(XtXi[i,i])

    
    return beta,mse,r2,beta_var


def eval_pol(betas,x,y,deg):
    n_p = (deg+1)*(deg+2)//2

    shape_x=np.shape(x)
    n2=shape_x[0]
    if (len(shape_x)<2):
        temp=np.copy(x)
        x=np.zeros(shape=(n2,1))
        x[:,0]=temp*1.0
    shape_y=np.shape(y)
    n3=shape_y[0]
    if (len(shape_y)<2):
        temp=np.copy(y)
        y=np.zeros(shape=(n3,1))
        y[:,0]=temp*1.0

    #build matrix
    shape_x=np.shape(x)
    n=shape_x[0]
    Xm=np.zeros(shape=(n,n_p))
    l=-1
    for j in range(deg+1):
        for k in range(j+1):
                l+=1
                Xm[:,l]=x[:,0]**(j-k) * y[:,0]**k

    return np.matmul(Xm,betas)

def split_data(x_vec,pct): #pct=percentage of data that is training data

    shape=np.shape(x_vec)
    n=shape[0]
    n2=int(n*(100.0-pct)/100.0)
    if (n2<1 or n2 > n):
        print('Use different value for percentage. Range = (0,100)')
        exit()
        
    bool_test=np.zeros(shape=(n,1),dtype='bool')
    bool_train=np.ones(shape=(n,1),dtype='bool')
    n_test=0
    
    while (n_test < n2): #works fast enough as long percentage is not very small
        ind=np.random.randint(n)
        if (bool_train[ind,0]==True):
            bool_test[ind,0]=True
            bool_train[ind,0]=False
            n_test+=1

    
    ind_test=np.where(bool_test==True)
    ind_train=np.where(bool_train==True)
    
    return ind_train,ind_test


    
def split_data_kfold(x_vec,y_vec,f_vec,k): #k=number of data groups

    shape=np.shape(x_vec)
    n=shape[0]
    if (k>n):
        print('k > n in split_data_kfold')
        exit()
    elif (k==n):
        outx=np.copy(x_vec)
        outy=np.copy(y_vec)
        outf=np.copy(f_vec)
        outn=np.ones(n,dtype='int')
        return outx,outy,outf,outn

    n_k=n//k + 1 #max number of points per group
    outx=np.zeros(shape=(k,n_k))
    outy=np.zeros(shape=(k,n_k))
    outf=np.zeros(shape=(k,n_k))
    outn=np.zeros(shape=(k),dtype='int')
    arr=np.zeros(shape=(n,3))
    arr[:,0]=np.copy(x_vec[:,0])
    arr[:,1]=np.copy(y_vec[:,0])
    arr[:,2]=np.copy(f_vec[:,0])

    n_left=n
    for i in range(n):
        k_ind = np.mod(i,k) #group number
        nk_ind = i//k       #index in group
        outn[k_ind]=nk_ind+1 #update number of values in group
        ind=np.random.randint(n_left) #draw random sample in data
        outx[k_ind,nk_ind]=np.copy(arr[ind,0])
        outy[k_ind,nk_ind]=np.copy(arr[ind,1])
        outf[k_ind,nk_ind]=np.copy(arr[ind,2])
        arr[ind:n_left-1]=np.copy(arr[ind+1:n_left])
        n_left-=1

    return outx,outy,outf,outn



def polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.0):
    msek=np.zeros(shape=(k,2))
    r2k=np.zeros(shape=(k,2))
    n_p=(deg+1)*(deg+2)//2
    betas=np.zeros(shape=(n_p,k))
    beta=np.zeros(shape=(n_p,1))
    beta_vark=np.zeros(shape=(n_p,k))
    for i in range(k):
#        print('group %i'%(i))
        #create X matrix and training data from groups j /= i
        nind=n2-nk[i]
        Xtr = np.ones(shape=(nind,n_p))
        ftr = np.zeros(shape=(nind,1))
        fte = np.zeros(shape=(nk[i],1))
        Xte = np.ones(shape=(nk[i],n_p))
        
        ind=-1
        p_min=0
        if (lamb>0.0):
            p_min=1
            beta[0,0]=np.sum(ftr[:,0])/nind
        for j in range(k):
            if (j==i):
                fte[:,0]=np.copy(fk[i,:nk[i]])
                for m in range(nk[j]):
                    l=p_min-1
                    for r in range(p_min,deg+1):
                        for s in range(r+1):
                            l+=1
                            Xte[m,l]=xk[j,m]**(r-s) * yk[j,m]**s
                continue
            
            for m in range(nk[j]):
                ind+=1
                ftr[ind,0]=np.copy(fk[j,m])
                
                l=p_min-1
                for r in range(p_min,deg+1):
                    for s in range(r+1):
                        l+=1
                        Xtr[ind,l]=xk[j,m]**(r-s) * yk[j,m]**s

        # perform polfit of beta
        Xt = Xtr[:,p_min:].T
        XtX = np.matmul(Xt,Xtr[:,p_min:])
        if (lamb>0.0): #i.e. Ridge regression
            Xlamb=np.zeros(shape=(n_p-1,n_p-1)) #(n_p x n_p matrix)
            for s in range(n_p-1):
                Xlamb[s,s]=lamb
            XtX=XtX+Xlamb
            f_temp=ftr-beta[0,0] #need to subtract away the average from the data
            Xtf = np.matmul(Xt,f_temp)
        else:
            Xtf = np.matmul(Xt,ftr)
        XtXi = np.linalg.inv(XtX)
        beta[p_min:,:]=np.matmul(XtXi,Xtf)
        betas[:,i]=beta[:,0]

        #mse and r2 for training data
        fm=np.mean(ftr[:,0])
        fxy=np.matmul(Xtr,beta)
        mse=np.sum((ftr-fxy)**2)
        t1=np.copy(mse)
        mse=mse/(nind-1)
        msek[i,0]=mse
        t2=np.sum((ftr-fm)**2)
        r2=1.0-t1/t2
        r2k[i,0]=r2

        for j in range(n_p):
            if (j==0 and lamb>0.0): #Ridge
                beta_vark[j,i]=t2/(nind-1)**2 
            elif (lamb>0.0):
                beta_vark[j,i]=mse*np.sqrt(XtXi[j-1,j-1]) #XtXi is a (n_p-1 x n_p-1) matrix
            else:
                beta_vark[j,i]=mse*np.sqrt(XtXi[j,j])

        
        #mse and r2 for test data (group i)
        fm=np.mean(fte[:,0])
        fxy=np.matmul(Xte,beta)
        mse=np.sum((fte-fxy)**2)
        t1=np.copy(mse)
        mse=mse/nk[i]
        msek[i,1]=mse
        t2=np.sum((fte-fm)**2)
        if (t2==0.0):
            r2=0.0
        else:
            r2=1.0-t1/t2
        r2k[i,1]=r2

        
    return msek,r2k,betas,beta_vark

def mse_plot_tradeoff_complexity(k,n, rnd,xmin=0.0,xmax=1.0, p_lim=-1,lamb=[0.0]):
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
        for p in range(n_deg):
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
            elif(verbosity > 0):
                print('  pol. deg. %i, lamb = %.3e'%(p,lamb))

        #plot the MSE to complexity plot train vs test for lambda 
        for m in range(2):
            plt.figure(1)
            if (m==0):
                plt.errorbar(p_plot,mse_mean[:,0,i_l],yerr=mse_error[:2,:,0,i_l],label='train', fmt='.')
                plt.errorbar(p_plot+0.1,mse_mean[:,1,i_l],yerr=mse_error[:2,:,1,i_l],label='test',fmt='x')
            else:
                plt.errorbar(p_plot,mse_mean[:,0,i_l],yerr=mse_error[2,:,0,i_l],label='train',fmt='.')
                plt.errorbar(p_plot+0.1,mse_mean[:,1,i_l],yerr=mse_error[2,:,1,i_l],label='test',fmt='x')
                
            plt.legend()
            plt.xlabel('Complexity (polynomial degree)')
            plt.ylabel('Mean Square Error')
            plt.yscale('log')
            outfile='tradeoff_complex'
            if (lamb>0.0):
                outfile=outfile+'_lamb_%.3e'%(lamb)
            if (rnd):
                plt.title('Random grid points')
                outfile=outfile+'_rnd_n%i'%(n)
                if (m==0):
                    outfile=outfile+'_minmax.png'
                else:
                    outfile=outfile+'_std.png'
            else:
                plt.title('Equidistant grid points')
                outfile=outfile+'_n%i'%(n)
                if (m==0):
                    outfile=outfile+'_minmax.png'
                else:
                    outfile=outfile+'_std.png'

            plt.savefig(outfile)
            #plt.show()
            plt.clf()

    for m in range(2):
        for s in range(2):
            plt.figure(1)
            for i_l in range(n_l):
                plt.errorbar(p_plot+(-n_l//2 + i_l)*0.1,mse_mean[:,s,i_l],yerr=mse_error[:2,:,s,i_l],label='lambda = %.1e'%(l_vec[i_l]), fmt='.')
                
            plt.legend()
            plt.xlabel('Complexity (polynomial degree)')
            plt.ylabel('Mean Square Error')
            plt.yscale('log')
            outfile='tradeoff_complex_lamb_multi'
            if (rnd):
                plt.title('Random grid points')
                outfile=outfile+'_rnd_n%i'%(n)
            else:
                plt.title('Equidistant grid points')
                outfile=outfile+'_n%i'%(n)

            if (s==0):
                outfile=outfile+'_train'
            else:
                outfile=outfile+'_test'

            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'

            plt.savefig(outfile)
            #plt.show()
            plt.clf()

    print('')
    print('')
    print('')
    print('')
    return

def mse_plot_tradeoff_number(k,rnd,n_min=6,n_max=30, p=5,xmin=0.0,xmax=1.0,lamb=0.0):
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
    # MSE means and errors
    mse_mean=np.zeros(shape=(n_n,2))
    mse_error=np.zeros(shape=(3,n_n,2)) #(error type, grid points (n*n), train/test)
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
            print(' xy-grid NxN, N = %i'%(n))

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
        
        msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
        mse_mean[i,0]=np.sum(msek[:,0])/k
        mse_mean[i,1]=np.sum(msek[:,1])/k
        mse_error[0,i,0]=mse_mean[i,0]-np.amin(msek[:,0])
        mse_error[1,i,0]=np.amax(msek[:,0])-mse_mean[i,0]
        mse_error[2,i,0]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,0]-mse_mean[i,0])**2))
        mse_error[0,i,1]=mse_mean[i,1]-np.amin(msek[:,1])
        mse_error[1,i,1]=np.amax(msek[:,1])-mse_mean[i,1]
        mse_error[2,i,1]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,1]-mse_mean[i,1])**2))
            
            
        # print the different MSE values for each k-fold modelling
        if (verbosity > 2):
            print('')
            print('-----------------------------------------------------')
            print('grid n = %i,  pol. deg. = %i'%(n,p))
            print('')
            print('group    mse_tr       mse_te       r2_tr')
            for j in range(k):
                print('  %i    %.4e   %.4e   %.4e '%(j,msek[j,0], msek[j,1],r2k[j,0]))
            print(' tot   %.4e   %.4e   '%(np.sum(msek[:,0])/k, np.sum(msek[:,1])/k))
            print('')
            if (p==2):
                print(' train_err        %.4e   %.4e   '%(mse_error[0,i,0], mse_error[1,i,0]))
                print(' test_err         %.4e   %.4e   '%(mse_error[0,i,1], mse_error[1,i,1]))
                print(' std (train,test) %.4e   %.4e   '%(mse_error[2,i,0], mse_error[2,i,1]))
        elif(verbosity > 0):
            print('  grid n = %i,  pol. deg. = %i'%(n,p))

    #plot the MSE to number of gridpoints plot plot
    for m in range(2):
        plt.figure(1)
        if (m==0):
            plt.errorbar(n_plot,mse_mean[:,0],yerr=mse_error[:2,:,0],label='train', fmt='.')
            plt.errorbar(n_plot+0.1,mse_mean[:,1],yerr=mse_error[:2,:,1],label='test',fmt='x')
        else:
            plt.errorbar(n_plot,mse_mean[:,0],yerr=mse_error[2,:,0],label='train',fmt='.')
            plt.errorbar(n_plot+0.1,mse_mean[:,1],yerr=mse_error[2,:,1],label='test',fmt='x')
                
        plt.legend()
        plt.xlabel('Number of grid points in each dimension (n)')
        plt.ylabel('Mean Square Error')
        plt.yscale('log')
        outfile='tradeoff_number'
        if (lamb>0.0):
            outfile=outfile+'_lamb_%.3e'%(lamb)
        if (rnd[i]):
            plt.title('Random grid points')
            outfile=outfile+'_rnd_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'
        else:
            plt.title('Equidistant grid points')
            outfile=outfile+'_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'

        plt.savefig(outfile)
        #plt.show()
        plt.clf()
    return


def mse_plot_tradeoff_kfold(n,rnd, k_min=2, k_max=10, p=5,xmin=0.0,xmax=1.0,lamb=0.0):
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

        #split data
        xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)
        
        msek,r2k,betak,var_bk=polfit_kfold(xk,yk,fk,nk,k,n2,deg=p,lamb=lamb)
        mse_mean[i,0]=np.sum(msek[:,0])/k
        mse_mean[i,1]=np.sum(msek[:,1])/k
        mse_error[0,i,0]=mse_mean[i,0]-np.amin(msek[:,0])
        mse_error[1,i,0]=np.amax(msek[:,0])-mse_mean[i,0]
        mse_error[2,i,0]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,0]-mse_mean[i,0])**2))
        mse_error[0,i,1]=mse_mean[i,1]-np.amin(msek[:,1])
        mse_error[1,i,1]=np.amax(msek[:,1])-mse_mean[i,1]
        mse_error[2,i,1]=np.sqrt(1.0/(k-1.0)*np.sum((msek[:,1]-mse_mean[i,1])**2))
            
            
        # print the different MSE values for each k-fold modelling
        if (verbosity > 2):
            print('')
            print('-----------------------------------------------------')
            print('grid n = %i,  pol. deg. = %i,  k-fold k = %i'%(n,p,k))
            print('')
            print('group    mse_tr       mse_te       r2_tr')
            for j in range(k):
                print('  %i    %.4e   %.4e   %.4e '%(j,msek[j,0], msek[j,1],r2k[j,0]))
            print(' tot   %.4e   %.4e   '%(np.sum(msek[:,0])/k, np.sum(msek[:,1])/k))
            print('')
            if (p==2):
                print(' train_err        %.4e   %.4e   '%(mse_error[0,i,0], mse_error[1,i,0]))
                print(' test_err         %.4e   %.4e   '%(mse_error[0,i,1], mse_error[1,i,1]))
                print(' std (train,test) %.4e   %.4e   '%(mse_error[2,i,0], mse_error[2,i,1]))
        elif(verbosity > 0):
            print('  grid n = %i,  pol. deg. = %i,  k-fold k = %i'%(n,p,k))

    #plot the MSE to number of gridpoints plot plot
    for m in range(2):
        plt.figure(1)
        if (m==0):
            plt.errorbar(k_plot,mse_mean[:,0],yerr=mse_error[:2,:,0],label='train', fmt='.')
            plt.errorbar(k_plot+0.1,mse_mean[:,1],yerr=mse_error[:2,:,1],label='test',fmt='x')
        else:
            plt.errorbar(k_plot,mse_mean[:,0],yerr=mse_error[2,:,0],label='train',fmt='.')
            plt.errorbar(k_plot+0.1,mse_mean[:,1],yerr=mse_error[2,:,1],label='test',fmt='x')
                
        plt.legend()
        plt.xlabel('Number of groups (k)')
        plt.ylabel('Mean Square Error')
        if (n<15):
            plt.yscale('log')

        outfile='tradeoff_kfold'
        if (lamb>0.0):
            outfile=outfile+'_lamb_%.3e'%(lamb)
        if (rnd[i]):
            plt.title('Random grid points')
            outfile=outfile+'_rnd_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'
        else:
            plt.title('Equidistant grid points')
            outfile=outfile+'_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'

        plt.savefig(outfile)

                
        #plt.show()
        plt.clf()
    return

def mse_plot_tradeoff_lambda(n,rnd,lamb_min=0.1,lamb_max=1.0,n_lamb=10, log_lamb=True, k=5, p=5,xmin=0.0,xmax=1.0):
    global verbosity
    global sigma
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
                
            if (verbocity > 0):
                print(' p = %i, lambda = %.3e'%(p_i,lamb))

    #plot the MSE to lambda for pol. degree p
    for m in range(2):
        plt.figure(1)
        if (log_lamb): #adding a shift to test data
            l_plot2=np.exp(np.log(l_plot)+0.1*dl)
        else:
            l_plot2=l_plot+0.1*dl
            
        if (m==0):
            plt.errorbar(l_plot,mse_mean[:,0,p],yerr=mse_error[:2,:,0,p],label='train', fmt='.')
            plt.errorbar(l_plot2,mse_mean[:,1,p],yerr=mse_error[:2,:,1,p],label='test',fmt='x')
            
        else:
            plt.errorbar(l_plot,mse_mean[:,0,p],yerr=mse_error[2,:,0,p],label='train',fmt='.')
            plt.errorbar(l_plot2,mse_mean[:,1,p],yerr=mse_error[2,:,1,p],label='test',fmt='x')
                
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
            plt.title('Random grid points')
            outfile=outfile+'_rnd_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'
        else:
            plt.title('Equidistant grid points')
            outfile=outfile+'_n%i'%(n)
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'

        plt.savefig(outfile)

                
        #plt.show()
        plt.clf()
    #plot the MSE to lambda for pol. degree p
    for m in range(2):
        for s in range(2):
            for p_i in range(1,p+1):
                plt.figure(1)
                if (log_lamb): #adding a shift to test data
                    l_plot2=np.exp(np.log(l_plot)+0.1*dl)
                else:
                    l_plot2=l_plot+0.1*dl

                if (m==0):
                    plt.errorbar(l_plot,mse_mean[:,s,p_i],yerr=mse_error[:2,:,s,p_i],label='pol %i'%(p_i), fmt='.')
                else:
                    plt.errorbar(l_plot,mse_mean[:,s,p_i],yerr=mse_error[2,:,s,p_i],label='pol %i'%(p_i), fmt='.')

                
            plt.legend()
            plt.xlabel('Lambda value')
            plt.ylabel('Mean Square Error')
            #plt.yscale('log')
            outfile='tradeoff_lamb_pol'
            if (log_lamb):
                plt.xscale('log')
                outfile=outfile+'_log'
            if (rnd):
                plt.title('Random grid points')
                outfile=outfile+'_rnd_n%i'%(n)
            else:
                plt.title('Equidistant grid points')
                outfile=outfile+'_n%i'%(n)

            if (s==0):
                outfile=outfile+'_train'
            else:
                outfile=outfile+'_test'
            if (m==0):
                outfile=outfile+'_minmax.png'
            else:
                outfile=outfile+'_std.png'

            plt.savefig(outfile)

                
            #plt.show()
            plt.clf()


    return
