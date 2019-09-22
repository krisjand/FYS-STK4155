import numpy as np

def polfit(deg,xv,yv,fv):
    n_p = (deg+1)*(deg+2)//2 #triangular rule but we have defined deg=0 for n=1

    shape_x=np.shape(xv)
    n2=shape_x[0]
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

    X=np.zeros(shape=(n2,n_p))
    for i in range(n2):
        l=-1
        for j in range(deg+1):
            for k in range(j+1):
                l+=1
                X[i,l]=xv[i,0]**(j-k) * yv[i,0]**k

    Xt = X.T
    XtX = np.matmul(Xt,X)
    XtXi = np.linalg.inv(XtX)
    Xtf = np.matmul(Xt,fv)
    beta=np.matmul(XtXi,Xtf)

    fm=np.mean(fv[:,0])
    fxy=np.matmul(X,beta)

    mse=np.sum((fv-fxy)**2)
    t1=np.copy(mse)
    mse=mse/n2
    
    t2=np.sum((fv-fm)**2)
    r2=1.0-t1/t2

    beta_var=np.zeros(shape=(n_p,1))
    for i in range(n_p):
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



    
def polfit_kfold(xk,yk,fk,nk,k,n2,deg):

    msek=np.zeros(shape=(k,2))
    r2k=np.zeros(shape=(k,2))
    n_p=(deg+1)*(deg+2)//2
    betas=np.zeros(shape=(n_p,k))
    for i in range(k):
#        print('group %i'%(i))
        #create X matrix and training data from groups j /= i
        nind=n2-nk[i]
        Xtr = np.zeros(shape=(nind,n_p))
        ftr = np.zeros(shape=(nind,1))
        fte = np.zeros(shape=(nk[i],1))
        Xte = np.zeros(shape=(nk[i],n_p))
        
        ind=-1
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
        XtXi = np.linalg.inv(XtX)
        Xtf = np.matmul(Xt,ftr)
        beta=np.matmul(XtXi,Xtf)
        betas[:,i]=beta[:,0]

        #mse and r2 for training data
        fm=np.mean(ftr[:,0])
        fxy=np.matmul(Xtr,beta)
        mse=np.sum((ftr-fxy)**2)
        t1=np.copy(mse)
        mse=mse/nind
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

        
    return msek,r2k,betas
