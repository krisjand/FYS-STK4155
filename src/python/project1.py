#FYS-STK4155 project 1 main code:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import time

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
n=21
n2=n**2
sigma=0.0
seed= 2236242
np.random.seed(seed)

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
plt.show()

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

from polynomialfit import *

#run the polynomial fitting for 0th-5th grade polynomials
for p in range(5,6):
    beta,mse,r2 = polfit(p,n2,x_vec,y_vec,f_vec)
    t2=time.time()
    print(t2-t1)
    print('beta')
    print(beta)
    print('')
    print('MSE')
    print(mse)
    print('')
    print('R^2')
    print(r2)


