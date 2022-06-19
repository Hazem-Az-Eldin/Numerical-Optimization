# this code is from :https://machinelearningmastery.com/line-search-optimization-with-python/
# perform a line search on a convex objective function
from numpy import arange
from scipy.optimize import line_search
from matplotlib import pyplot
 
# objective function
def objective(x):
	return (-5.0 + x)**2.0
 
# gradient for the objective function
def gradient(x):
	return 2.0 * (-5.0 + x)
 
# define the starting point
point = -5.0
# define the direction to move
direction = 100.0
# print the initial conditions
print('start=%.1f, direction=%.1f' % (point, direction))
# perform the line search
result = line_search(objective, gradient, point, direction)
# summarize the result
alpha = result[0]
print('Alpha: %.3f' % alpha)
print('Function evaluations: %d' % result[1])
# define objective function minima
end = point + alpha * direction
# evaluate objective function minima
print('f(end) = f(%.3f) = %.3f' % (end, objective(end)))
# define range
r_min, r_max = -10.0, 20.0
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [objective(x) for x in inputs]
# plot inputs vs objective
pyplot.plot(inputs, targets, '--', label='objective')
# plot start and end of the search
pyplot.plot([point], [objective(point)], 's', color='g')
pyplot.plot([end], [objective(end)], 's', color='r')
pyplot.legend()
pyplot.show()