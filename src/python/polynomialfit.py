import numpy as np

def polfit(deg,n2,xv,yv,fv):
    n_p=0
    for i in range(deg+1):
        n_p+=i+1

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
    return beta,mse,r2


    
