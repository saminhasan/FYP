from cf import*
import numpy as np
from scipy.optimize import *

np.random.seed(0)

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
	
	
def optimize_path(path, obstacles, max_iter=9000):
	bounds = calc_bounds(path)
	nm = "Nelder-Mead" # 100
	result = minimize(calc_cost, path, args=(obstacles), method=nm, bounds=bounds ,options={'maxiter':len(path) * max_iter , 'adaptive': True, 'disp': True })
	#result = minimize(fun, x0, args=(), method='Nelder-Mead', bounds=None, tol=None, callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
	if result.success:
		print("Optimization Successful.")
	else:
		print("Optimization Failed.")
	return result.x


def genrate_initial_path(start, finish, step_size):
	linear_distance = np.linalg.norm(finish-start) # shorttest path distance,  straight line path

	num_steps = linear_distance //  step_size + 1 	#calculate number of steps to divide path into

	initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)# The guess path for the optimizer to start workinig with, a straight line connecting start and finish line

	noise = np.random.randn(*initial_path.shape) * 1.0

	base_path =  initial_path + noise * 0.5

	initial_path +=  noise
	
	return base_path, initial_path

if __name__ == "__main__":
	print(__file__)