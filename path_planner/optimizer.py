import numpy as np
from scipy.optimize import *

from cf import *

np.random.seed(0)
i = 0


def calc_bounds(path):
	bounds = []
	lb = []
	ub = []
	z = 3.0	 # wiggle room
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
		bounds.append([bx, by, bz])
	bounds = np.asarray(bounds)
	bounds = bounds.reshape(len(path) * 3, 2)
	return bounds


def optimize_path(path, obstacles, max_iter=9000):
	bounds = calc_bounds(path)
	nm = "Nelder-Mead"	# 100
	iter_len = len(path) * max_iter
	print("max iter : ", iter_len)
	global i

	i = 0

	def iter_counter(a):

		global i
		i += 1
		print("Progress : {:.2f} %".format(i * 100 / iter_len), end='\r')

	result = minimize(calc_cost, path, args=(obstacles), method=nm, bounds=bounds, callback=iter_counter,
					  options={'maxiter': len(path) * max_iter, 'adaptive': True, 'disp': True})
	if result.success:
		print("Optimization Successful.")
	else:
		print("Optimization Failed.")
	return result.x


def genrate_initial_path(start, finish, step_size):
	linear_distance = np.linalg.norm(finish - start)  # shorttest path distance,  straight line path

	num_steps = linear_distance // step_size + 1  # calculate number of steps to divide path into

	initial_path = np.linspace(start, finish, num=int(num_steps),
							   endpoint=True)  # The guess path for the optimizer to start workinig with, a straight line connecting start and finish line

	noise = np.random.randn(*initial_path.shape) * 3.0

	base_path = initial_path + noise * 0.5

	initial_path += noise
	initial_path = smooth(initial_path)
	base_path = smooth(base_path)
	initial_path[0], initial_path[-1] = start, finish
	base_path[0], base_path[-1] = start, finish
	return base_path, initial_path


if __name__ == "__main__":
	print(__file__)
