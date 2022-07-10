# this code is from toward data sciene artical: https://towardsdatascience.com/complete-step-by-step-gradient-descent-algorithm-from-scratch-acba013e8420
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product

def func(x): #function
    return x**2 

def fprime(x): # derivavitive
    return 2*x 

def plot_slope():
    plt.plot(x,fprime(x))

def plotFunc(x):  # used to plot the fuction
   # plt.plot(x0, func(x0), 'ro') #mark the starting point
   # plt.plot(x0, fprime(x0), 'ro') #mark the starting point
    plt.plot(x, func(x))
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Objective Function')
       

def plotPath(xs, ys, x0): # plot the Path of the Gradient
    plotFunc(x)
    plt.plot(xs, ys, linestyle='-', marker='o', color='orange')
    plt.plot(xs[-1], ys[-1], 'ro')
    plt.grid()
    
    

def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=2): # Gradient Descent algorithm
    # initialize x, f(x), and -f'(x)
    xk = x0 #  
    fk = func(xk) # choose a starting point on the function
    pk = -fprime(xk) # Calculate the rate of change of the starting point
    print(pk)
    # initialize number of steps, save x and f(x\)
    num_iter = 0
    curve_x = [xk] # point.x that will be on the graph
    curve_y = [fk] # point.y that will be on the graph
    #x_value = [xk,fk]
    #y_value = [pk,fk]
    #plt.plot(x0, func(x0), 'ro') #mark the starting point
    #plt.plot(x0, func(x0), 'ro') #mark the starting point
    #plt.plot(x_value,y_value)
    # take steps
    while abs(pk) > tol and num_iter < max_iter:
        # calculate new x, f(x), and -f'(x)
        xk = xk + alpha * pk # equation of the gradient descent
        fk = func(xk)
        pk = -fprime(xk)
        print(pk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        print('the new x',xk)
        print('the new f(X)',fk)
        curve_x.append(xk)
        curve_y.append(fk)
    # print results
    if num_iter == max_iter:
        print('Gradient descent does not converge.')
    else:
        print('Solution found:\n  y = {:.4f}\n  x = {:.4f}'.format(fk, xk))
    
    return curve_x, curve_y

x = np.linspace(-10, 10, 200) # make the function continouis
x0 = 3
plotFunc(x)
#plot_slope()

xs, ys = GradientDescentSimple(func, fprime, x0, alpha=0.9)
plotPath(xs, ys, x0)
plt.show()