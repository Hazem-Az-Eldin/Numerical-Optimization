###################################################################
###################### EXAMPLE - 2 ################################
###################################################################
import numpy as np
import numpy.linalg as npla
#Randomly Initializing A and b. A is 100*3 matrix , b is a 100*1 matrix
A = np.asmatrix(np.random.rand(100,3))
b = np.asmatrix(np.random.rand(100,1))
#Defining the Function
def f(x): #x is 3d point e.g [1,1,1], Returns a scalar function value
    point = np.asmatrix(x)
    return np.asscalar(0.5*np.dot((np.dot(A,point.T) - b).T,(np.dot(A,point.T) - b)))
# first order derivatives of the function at point x
def fdx(x):
    point = np.asmatrix(x)
    return np.dot(np.dot(A.T,A),point.T) - np.dot(A.T,b) #Returns a matrix 3*1
# second order derivatives of the function at point x
def hess():
    
    return np.dot(A.T,A) #This is always Positive Definite, Returns a 3*3 matrix
###################################################################
###################### ANALYTICAL METHOD ##########################
###################################################################

#Analytical Solution - As it turns out linear algebra gives us an analytical solution. We will see how well numerical methods approximate the analytical solution
J = npla.pinv(np.dot(A.T,A))
print("Optimal x = ",np.dot(np.dot(J,A.T),b))
###################################################################
###################### LINE SEARCH STEP SIZE ######################
###################################################################
# This is an implementation of the backtracking algorithm that automatically generates a Step Size for every iteration at a point x0. This means you do not have to specify an arbitrary step size.
def backtrack4(x0, f, fdx, t = 1, alpha = 0.2, beta = 0.8):
    
    point = np.asmatrix(x0) #Necessary to ensure matrix form
    while f(point - np.dot(t,fdx(point).T)) > f(point) + alpha * t * np.asscalar(np.dot(fdx(point).T, -1*fdx(point))):
         t *= beta
    return t
###################################################################
###################### GRAD. DESCENT #############################
###################################################################
def grad(x0, max_iter):
iter = 1
    
    while (np.linalg.norm(np.array(fdx(x0).flatten())[0]) > 0.000001):
    #Find stepsize by backtracking
    t = backtrack4(x0, f, fdx) #Step Size
    x0 = x0 - np.dot(t, fdx(x0).T)
    #Calculate New Value of Function
    print(x0, f(x0), fdx(x0), iter)
    iter += 1
    if iter > max_iter:
       break
return x0, f(x0), iter

#Test
grad([0.5,0.5,0.5], 100)
###################################################################
###################### NEWTON'S METHOD ############################
###################################################################
def lambda_sq(fdx, hess, x0):
    lambda_sq = np.dot(np.dot(fdx(x0).T , npla.pinv(hess())) ,fdx(x0))
    return(np.asscalar(lambda_sq))

def delta_x(fdx, hess, x0): #Returns a 3*1 matrix
    delta_x = np.dot(-npla.pinv(hess()) , fdx(x0))
    return(delta_x)
   
def newtons_method(x0, eps=0.00001, max_iters=300):
    # Compute 
    iters = 1
    lmb_sq = lambda_sq(fdx, hess, x0)
    while((lmb_sq/2.0) > eps):
        # Compute delta_x and lambda_sq
        dlt_x = delta_x(fdx, hess, x0)
        #Line search for t
        t = backtrack4(x0, f, fdx)
        x0 = x0 + np.dot(t , dlt_x.T)
        
        #Show Iterations
        print(x0, f(x0), iters)
        
        # Update lmb_sq, see if we still stay in the loop
        lmb_sq = lambda_sq(fdx, hess, x0)
        iters += 1
        
        if(iters > max_iters):
            break
return(x0, f(x0), iters)
#Test
newtons_method([0.5,0.5,0.5])

#Your final optimal point will be very close to what you find through analytical solution