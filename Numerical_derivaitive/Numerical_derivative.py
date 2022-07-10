import numpy as np 
import matplotlib.pyplot as plt 

def func(x,A,B):
    return A*x**2 + B*x -3

def D(xlist,ylist):
    yprime = np.diff(ylist)/np.diff(xlist) # ?????
    xprime = []
    for i in range(len(yprime)):
        xtemp  = (xlist[i+1]+xlist[i])/2
        xprime = np.append(xprime,xtemp)
    return xprime, yprime


xlist = np.linspace(-10,10,20)
ylist = func(xlist,1,-2)

xprime, yprime = D(xlist,ylist)


plt.figure(1,dpi=120)
plt.plot(xlist,ylist,label="function")
plt.plot(xprime,yprime,label = "Derivavtive")
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()

