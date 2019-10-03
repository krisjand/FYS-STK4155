import numpy as np
import matplotlib.pyplot as plt
from params import *
from sklearn import linear_model

##################################
# Functions for creation of data #
# grid and performing regression #
##################################

def inv_mat_SVD(X):

    U, s, VT = np.linalg.svd(X)
    D = np.zeros((len(U), len(VT)))
    for i in range(0,len(VT)):
        D[i,i] = s[i]
    UT = U.T;
    V = VT.T;
    invD = np.linalg.inv(D)
    invX = np.matmul(V, np.matmul(invD, UT))

    return invX

def init_xy_vectors(n,rand,xmin=0.0,xmax=1.0,rearr=False,x=np.zeros(shape=(1,1)),y=np.zeros(shape=(1,1)),z=np.zeros(shape=(1,1)),ter=False):

    if (rearr and (not ter)):
        shape_x=np.shape(x)
        n=shape_x[0]
        m=shape_x[1]
        nm=n*m
        x_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        y_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        z_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        for i in range(n):
            x_vec[m*i:m*(i+1),0]=x[i,:]
            y_vec[m*i:m*(i+1),0]=y[i,:]
            z_vec[m*i:m*(i+1),0]=z[i,:]
        return x_vec,y_vec,z_vec
    if (rearr and ter):
        shape_z=np.shape(z)
        n=shape_z[0] #along y
        m=shape_z[1] #along x
        nm=n*m
        x_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        y_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        z_vec=np.random.uniform(xmin,xmax,size=(nm,1))
        for i in range(n):
            x_vec[m*i:m*(i+1),0]=x[:]   # along row i, x increases
            y_vec[m*i:m*(i+1),0]=y[i]   # along row i, y is constant
            z_vec[m*i:m*(i+1),0]=z[i,:] # along row i for the terrain
        return x_vec,y_vec,z_vec
    else:
        n2=n**2
        if (rand):
            x_vec=np.random.uniform(xmin,xmax,size=(n2,1))
            y_vec=np.random.uniform(xmin,xmax,size=(n2,1))
        else:
            dx=np.zeros(1)
            dx=(xmax-xmin)/(n-1)
            x_vec=np.zeros(shape=(n2,1))
            y_vec=np.zeros(shape=(n2,1))
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

    X=np.ones(shape=(n2,n_p)) #we change these values, but in the case of Ridge we don't
                              # have to change X[:,0] values for later evaluation
    for i in range(n2):
        l=-1
        for j in range(deg+1):
            for k in range(j+1):
                l+=1
                X[i,l]=xv[i,0]**(j-k) * yv[i,0]**k

    Xt = X.T
    XtX = np.matmul(Xt,X)
    if (lamb>0.0): #i.e. Ridge regression
        Xlamb=np.zeros(shape=(n_p,n_p)) #(n_p x n_p matrix)
        for i in range(n_p):
            Xlamb[i,i]=lamb
        XtX=XtX+Xlamb
    Xtf = np.matmul(Xt,fv)
    XtXi = np.linalg.inv(XtX)
    beta=np.matmul(XtXi,Xtf)

    fm=np.mean(fv[:,0])
    fxy=np.matmul(X,beta)

    mse=np.sum((fv-fxy)**2)
    t1=np.copy(mse)
    mse=mse/(n2-1)
    
    t2=np.sum((fv-fm)**2)
    r2=1.0-t1/t2

    beta_var=np.zeros(shape=(n_p,1))
    for i in range(n_p):
        beta_var[i,0]=mse*XtXi[i,i]
    
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

def eval_pol3D(betas,x,y,deg):
    n_p = (deg+1)*(deg+2)//2

    shape_x=np.shape(x)
    z=np.zeros(shape=(shape_x[0],shape_x[1]))
    xy=np.zeros(n_p)
    for r in range(shape_x[0]):
        for s in range(shape_x[1]):
            l=-1
            for j in range(deg+1):
                for k in range(j+1):
                    l+=1
                    xy[l]=x[r,s]**(j-k) * y[r,s]**k
            z[r,s]=np.sum(xy*betas)
    return z

def eval_terrain(betas,x,y,deg,nx,ny):
    n_p = (deg+1)*(deg+2)//2

    z=np.zeros(shape=(ny,nx))
    xy=np.zeros(n_p)
    for r in range(ny):
        for s in range(nx):
            l=-1
            for j in range(deg+1):
                for k in range(j+1):
                    l+=1
                    xy[l]=x[s]**(j-k) * y[r]**k
            z[r,s]=np.sum(xy*betas)
    return z

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



def polfit_kfold(xk,yk,fk,nk,k,n2,deg=5,lamb=0.0,var_mse=False):
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
        for j in range(k):
            if (j==i):
                fte[:,0]=np.copy(fk[i,:nk[i]])
                for m in range(nk[j]):
                    l=-1
                    for r in range(deg+1):
                        for s in range(r+1):
                            l+=1
                            Xte[m,l]=xk[j,m]**(r-s) * yk[j,m]**s
                continue
            
            for m in range(nk[j]):
                ind+=1
                ftr[ind,0]=np.copy(fk[j,m])
                
                l=-1
                for r in range(deg+1):
                    for s in range(r+1):
                        l+=1
                        Xtr[ind,l]=xk[j,m]**(r-s) * yk[j,m]**s

        # perform polfit of beta
        Xt = Xtr.T
        XtX = np.matmul(Xt,Xtr)
        if (lamb>0.0): #i.e. Ridge regression
            Xlamb=np.zeros(shape=(n_p,n_p)) #(n_p x n_p matrix)
            XtX0=np.copy(XtX) #for variance determination
            for s in range(n_p):
                Xlamb[s,s]=lamb
            XtX=XtX+Xlamb
        Xtf = np.matmul(Xt,ftr)
        XtXi = inv_mat_SVD(XtX)
        beta=np.matmul(XtXi,Xtf)
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

        if (var_mse):
            var_scale=mse
        else:
            var_scale=sigma**2
            
        for j in range(n_p):
            if (lamb > 0.0):
                Xv=np.matmul(XtXi,np.matmul(XtX0,np.transpose(XtXi)))
                beta_vark[j,i]=var_scale*Xv[j,j]
            else:
                beta_vark[j,i]=var_scale*XtXi[j,j]
            #beta_vark[j,i]=mse*XtXi[j,j] #not sure if one can say that it is
            #the white noise level, or the mse of either the training or test data.
            #If we only have the data, and do not know the noise, out best estimate of the
            #white noise level is the test mse

        
    return msek,r2k,betas,beta_vark

def fit_lasso(deg,xv,yv,fv,lamb=1.0,max_iter=10000,return_fit=False,tol=0.000001):
    shape_x=np.shape(xv)
    n2=shape_x[0]
    
    n_p=(deg+1)*(deg+2)//2
    print(n_p)
    X=np.zeros(shape=(n2,n_p))
    for i in range(n2):
        l=-1
        for j in range(deg+1):
            for k in range(j+1):
                l+=1
                X[i,l]=xv[i,0]**(j-k) * yv[i,0]**k

    reg = linear_model.Lasso(alpha=lamb,max_iter=max_iter)
    Fxy = reg.fit(X,fv[:,0]).predict(X)
    mse=np.sum((Fxy-fv[:,0])**2)
    t1=np.copy(mse)
    mse=mse/(n2-1)
    fm=sum(fv[:,0])/n2
    t2=np.sum((fv[:,0]-fm)**2)
    r2=1.0-t1/t2
    beta=np.zeros(shape=(n_p,1))
    beta[:,0]=reg.coef_
    beta[0,0]+=reg.intercept_

    if (return_fit):
        return mse,r2,beta,reg
    else:
        return mse,r2,beta

def kfold_CV_lasso(xk,yk,fk,nk,k,n2,deg=5,lamb=1.0,tol=0.000001,max_iter=100000):
    msek=np.zeros(shape=(k,2))
    r2k=np.zeros(shape=(k,2))
    n_p=(deg+1)*(deg+2)//2
    betas=np.zeros(shape=(n_p,k))
    beta=np.zeros(shape=(n_p,1))

    for i in range(k):
        #print('group %i'%(i))
        #create X matrix and training data from groups j /= i
        nind=n2-nk[i]
        Xtr = np.ones(shape=(nind,n_p))
        ftr = np.zeros(shape=(nind,1))
        fte = np.zeros(shape=(nk[i],1))
        Xte = np.ones(shape=(nk[i],n_p))
        
        ind=-1
        p_min=0
        for j in range(k):
            if (j==i):
                fte[:,0]=np.copy(fk[i,:nk[i]])
                for m in range(nk[j]):
                    l=-1
                    for r in range(deg+1):
                        for s in range(r+1):
                            l+=1
                            Xte[m,l]=xk[j,m]**(r-s) * yk[j,m]**s
                continue
            
            for m in range(nk[j]):
                ind+=1
                ftr[ind,0]=np.copy(fk[j,m])
                
                l=-1
                for r in range(deg+1):
                    for s in range(r+1):
                        l+=1
                        Xtr[ind,l]=xk[j,m]**(r-s) * yk[j,m]**s

        # perform Lasso polfit
        reg = linear_model.Lasso(alpha=lamb,max_iter=max_iter,tol=tol)
        reg.fit(Xtr,ftr[:,0])
        beta[:,0]=reg.coef_
        beta[0,0]+=reg.intercept_
        betas[:,i]=beta[:,0]
        #mse and r2 for training data
        fm=np.mean(ftr[:,0])
        fxy=reg.predict(Xtr)
        mse=np.sum((ftr[:,0]-fxy)**2)
        t1=np.copy(mse)
        mse=mse/(nind-1)
        msek[i,0]=mse
        t2=np.sum((ftr[:,0]-fm)**2)
        r2=1.0-t1/t2
        r2k[i,0]=r2


        #mse and r2 for test data (group i)
        fm=np.mean(fte[:,0])
        fxy=reg.predict(Xte)
        mse=np.sum((fte[:,0]-fxy)**2)
        t1=np.copy(mse)
        mse=mse/nk[i]
        msek[i,1]=mse
        t2=np.sum((fte-fm)**2)
        if (t2==0.0):
            r2=0.0
        else:
            r2=1.0-t1/t2
        r2k[i,1]=r2

        
    return msek,r2k,betas
