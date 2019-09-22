#FYS-STK4155 project 1 main code:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import time
from p1_functions import *


# Make data.
n=21
n2=n**2
sigma=0.1
seed= 2236242
np.random.seed(seed)
verbosity=1

x = np.arange(0, 1.0001, 1.0/(n-1))
y = np.arange(0, 1.0001, 1.0/(n-1))
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Franke function')
#plt.show()
plt.clf()

t1=time.time()

x_vec=np.zeros(shape=(n2,1))
y_vec=np.zeros(shape=(n2,1))
f_vec=np.zeros(shape=(n2,1))

dx=np.zeros(1)
dx=1.0/(n-1)
for i in range(n):
    x_vec[n*i:n*(i+1),0]=i*dx
    y_vec[i:n2:n,0]=i*dx

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
    if (verbocity > 1):
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
    msek,r2k,betak=polfit_kfold(xk,yk,fk,nk,k,n2,p)
    print('')
    print('-----------------------------------------------------')
    print("polynomial of degree %i"%(p))
    print('')
    print('group    mse_tr       mse_te       r2_tr       r2_te')
    for i in range(k):
        print('  %i    %.4e   %.4e   %.4e  %.4e'%(i,msek[i,0], msek[i,1],r2k[i,0],r2k[i,1]))

    print(' tot   %.4e   %.4e   '%(np.sum(msek[:,0])/k, np.sum(msek[:,1])/k))
    print('')

k=n2
xk,yk,fk,nk=split_data_kfold(x_vec,y_vec,f_vec,k)
print('')
print('-----------------------')
print('LOOCV')
print('')
print('degree   mse_tr       mse_te')
for p in range(6):
    msek,r2k,betak=polfit_kfold(xk,yk,fk,nk,n2,n2,p)

    print(' %i   %.4e   %.4e   '%(p,np.sum(msek[:,0])/n2, np.sum(msek[:,1])/n2))
