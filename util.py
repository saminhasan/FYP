import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *
from matplotlib import cm


def calc_length(curve):

	curve = curve.reshape(int(curve.shape[0]//3), 3)
	length = np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2,axis=1))) #  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	return length
	
def calc_time(path):

	# constant velocity model, very uselss , place holder for now 
	vel = 10 # m/s
	time = calc_length(path) / vel
	return time
	
def rosenbrock(X):
	X = X.reshape((int(X.shape[0]//3), 3))

	"""
	This R^2 -> R^1 function should be compatible with algopy.
	http://en.wikipedia.org/wiki/Rosenbrock_function
	A generalized implementation is available
	as the scipy.optimize.rosen function
	"""
	x = X[:,0]
	y = X[:,1]
	a = 1. - x
	b = y - x*x
	c = X[:,2]
	return np.sum(a*a + b*b*100.)


	
def calc_energy_consumption(path):
	# for now place holder for environment
	
	
	return 0 #rosenbrock(path)
	

	
def cost_func(path):
	

	length = calc_length(path)
	length_weight = 1.0
	length_cost = length * length_weight

	
	time = calc_time(path)
	time_weight = 1.0
	time_cost = time * time_weight
	
	
	
	energy_consumption = calc_energy_consumption(path)
	energy_weight = 2.0
	energy_cost = energy_consumption * energy_weight
	
	#print(length_cost, time_cost, energy_cost)
	cost = length_cost + time_cost + energy_cost


	return cost
	

	
def optimize_path(path):

	bnds = calc_bounds(path)
	nm = "Nelder-Mead"
	pw = "Powell"
	lb = "L-BFGS-B"
	tnc = "TNC"
	tr="trust-constr" 
	sl = "SLSQP"
	result = minimize(cost_func, path, method=nm, bounds=bnds)
	#result = minimize(cost_func, path, method=sl)
	#result = brute(cost_func, path)
	print(result.success)
	return result.x



def calc_bounds(path):
	bounds = []
	lb= []
	ub = []
	z = 1.0 # wiggle room
	for i in range(0, len(path)):
		if i == 0 or i == len(path) - 1:
			lim_r = 0
		else:
			lim_r = z
		xlb = path[i][0] - lim_r
		xub = path[i][0] + lim_r
		ylb = path[i][1] - lim_r
		yub = path[i][1] + lim_r
		zlb = path[i][2] - lim_r
		zub = path[i][2] + lim_r
		bx = [xlb, xub]
		by = [ylb, yub]
		bz = [zlb, zub]
		bounds.append([bx,by,bz])
	bounds = np.asarray(bounds)
	bounds = bounds.reshape(len(path)*3, 2)
	return bounds
	
def plot(initial_path, path):
		fig = plt.figure('3D plots')
		ax = plt.axes(projection='3d')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		ax.set_title('3D plot Title')
		ax.scatter(initial_path[:,0],initial_path[:,1],initial_path[:,2], c = initial_path[:,2])
		ax.plot(initial_path[:,0],initial_path[:,1],initial_path[:,2],label='Initial path')
		ax.scatter(path[:,0],path[:,1],path[:,2], c = np.arange(0,len(path), 1))
		ax.plot(path[:,0],path[:,1],path[:,2],label='Optimized path')
		plt.legend()
		plt.show()
	
def main():
	'''
	genarate a straight line path as initial solution.
	feed it to optimizer
	plot output
	
	'''

	step_size = 1.0 # in meters
	# starting co-ordinate 
	start = np.array([-2,3,0]) 
	# Finishing co-ordinate 
	finish = np.array([3,3,10])
	# shorttest path distance,  straight line path
	linear_distance = np.linalg.norm(finish-start)
	#calculate number of steps to divide path into
	num_steps = linear_distance //  step_size + 1

	# The guess path for the optimizer to start workinig with, a straight line connecting start and finish line
	initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)

	initial_path += initial_path + np.random.randn(*initial_path.shape) * 0.1
	
	path = optimize_path(initial_path)
	#reshapinig  output array  for plotttinig
	#print(path, path.shape)

	path = path.reshape((int(path.shape[0]//3), 3))

	# plotting 
	
	plot(initial_path, path)
	
	
	# cost of stragith line path
	#c = cost_func(initial_path)
	#print(c)
	#print(linear_distance, c)
	#print(path,"\n","\n",len(path),"\n",np.diff(path,axis=0))
	
if __name__ == "__main__":
	main()

