import numpy as np
import matplotlib.pyplot as plt


plt.figure(1)

b = np.loadtxt('beta.dat', unpack=True)
x_plt=np.zeros(101)
for i in range(101):
    x_plt[i] = i*0.01
    
y_plt=b[0]+b[1]*x_plt+b[2]*x_plt**2
plt.plot(x_plt,y_plt)

x,y = np.loadtxt('xy.dat', unpack=True)
plt.plot(x,y,'.')

plt.show()
plt.clf()
exit()







plt.figure(1)
for n in ['1000','10000','100000', '1000000']:
    x,p = np.loadtxt('prb_ran0_n'+n+'.dat', unpack=True)

    plt.plot(x,p,label=n)

plt.legend()
plt.show()
plt.clf()


plt.figure(1)
for n in ['1000','10000','100000', '1000000']:
    x,p = np.loadtxt('prb_ran1_n'+n+'.dat', unpack=True)

    plt.plot(x,p,label=n)

plt.legend()
plt.show()
plt.clf()


plt.figure(1)
n= ['1000','10000','100000', '1000000']
d= [0.01,0.004,0.003,0.001]
for i in range(4):
    for j in ['0','1']:
        x,p = np.loadtxt('prb_ran'+j+'_n'+n[i]+'.dat', unpack=True)
        plt.plot(x,p,label=j)

    plt.ylim(0.01-d[i],0.01+d[i])
    plt.legend()
    plt.show()
    plt.clf()


plt.figure(1)
for n in ['1000','10000','100000', '1000000']:
    x,p = np.loadtxt('prb_normal_0_1_n'+n+'.dat', unpack=True)

    plt.plot(x,p,label=n)

plt.legend()
plt.show()
plt.clf()
