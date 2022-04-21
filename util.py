import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *
from matplotlib import cm
from sim import energy_from_waypoints
import pyximport; pyximport.install()

np.random.seed(0)
obstacles = [[2,2,3,0.5], [4,5,6,1]] #x, y, z, r
fig = plt.figure('3D plots')
ax = plt.axes(projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D plot Title')

		
def calc_length(curve):

	curve = curve.reshape(int(curve.shape[0]//3), 3)
	length = np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2,axis=1))) #  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	return length
	
def calc_time(path):

	# constant velocity model, very uselss , place holder for now 
	vel = 10 # m/s
	time = calc_length(path) / vel
	return time
	


	
def calc_energy_consumption(path):
	
	path = path.reshape((int(path.shape[0]//3), 3))
	#energy = energy_from_waypoints(path)
	
	
	return 0#energy
	
def calc_collision(path, obstacles):
	path = path.reshape((int(path.shape[0]//3), 3))

	collision_cost = 100
	total_cost = 0
	for i in range(len(path) -1):
		for j in range(len(obstacles)):

			p1 = path[i]
			p2 = path[i + 1]

			p3 = obstacles[j][0:3]
			r = obstacles[j][3]

			a = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2
			b = 2*((p2[0] - p1[0])*(p1[0] - p3[0]) + (p2[1] - p1[1])*(p1[1] - p3[1]) + (p2[2] - p1[2])*(p1[2] - p3[2]))
			c = p3[0]**2 + p3[1]**2 + p3[2]**2 + p1[0]**2 + p1[1]**2 + p1[2]**2 - 2*(p3[0]*p1[0] + p3[1]*p1[1] + p3[2]*p1[2]) - r**2

			D = b * b - 4 * a * c

			#if D < 0:
				#print('no intersection')
			#if D == 0:
				#print('tangent')
			if D > 0:
				#print('section')
				total_cost += collision_cost
	return total_cost
	
def cost_func(path):
	
	length = calc_length(path)
	length_weight = 1.0
	length_cost = length * length_weight

	vel = 10  # m/s
	time = length / vel 
	time_weight = 1.0
	time_cost = time * time_weight
	
	
	
	energy_consumption = calc_energy_consumption(path)
	energy_weight = 1.0
	energy_cost = energy_consumption * energy_weight
	
	
	collision = calc_collision(path,obstacles)
	collision_weight = 1.0
	collision_cost = collision * collision_weight
	#print(length_cost, time_cost, energy_cost, collision_cost)
	cost = length_cost + time_cost + energy_cost + collision_cost


	return cost
	

	
def optimize_path(path):

	bnds = calc_bounds(path)
	nm = "Nelder-Mead" # 100
	pw = "Powell" # 120
	lb = "L-BFGS-B" #19
	tnc = "TNC" # 
	tr="trust-constr" 
	sl = "SLSQP"
	result = minimize(cost_func, path, method=nm, bounds=bnds ,options={'maxiter':len(path) * 3333 * 2 , 'adaptive': True, 'disp': True })
	#result = minimize(fun, x0, args=(), method='Nelder-Mead', bounds=None, tol=None, callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
	
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
		ax.scatter(initial_path[:,0],initial_path[:,1],initial_path[:,2], c = initial_path[:,2])
		ax.plot(initial_path[:,0],initial_path[:,1],initial_path[:,2],label='Initial path')
		ax.scatter(path[:,0],path[:,1],path[:,2], c = np.arange(0,len(path), 1))
		ax.plot(path[:,0],path[:,1],path[:,2],label='Optimized path')
		plot_obstacles(obstacles)
		plt.legend()
		plt.show()


def plot_obstacles(obstacles):
	for j in range(len(obstacles)):
		u = np.linspace(0, 2 * np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		r = obstacles[j][3]
		x = r * np.outer(np.cos(u), np.sin(v)) + obstacles[j][0]
		y = r * np.outer(np.sin(u), np.sin(v)) + obstacles[j][1]
		z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacles[j][2]

		# Plot the surface
		ax.plot_surface(x, y, z, color='b', alpha=0.1)
def main():
	'''
	genarate a straight line path as initial solution.
	feed it to optimizer
	plot output
	
	'''

	step_size = 1.0 # in meters
	# starting co-ordinate 
	start = np.array([0,0,0]) 
	# Finishing co-ordinate 
	finish = np.array([5,5,5])
	# shorttest path distance,  straight line path
	linear_distance = np.linalg.norm(finish-start)
	#calculate number of steps to divide path into
	num_steps = linear_distance //  step_size + 1

	# The guess path for the optimizer to start workinig with, a straight line connecting start and finish line
	initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)
	noise = np.random.randn(*initial_path.shape) * 0.5
	initial_path +=  noise
	#print(initial_path, noise)
	path = optimize_path(initial_path)
	#reshapinig  output array  for plotttinig
	#print(path, path.shape)
	c = cost_func(path)
	#print(c)
	path = path.reshape((int(path.shape[0]//3), 3))
	#print(path)

	# plotting 
	
	plot(initial_path, path)
	
	
	# cost of stragith line path

	#print(linear_distance, c)
	#print(path,"\n","\n",len(path),"\n",np.diff(path,axis=0))
	
if __name__ == "__main__":
	for _ in range(1):
		main()

