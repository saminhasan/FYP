import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt


def calc_length(curve,start,stop):
	#curve = curve.reshape(int(curve.shape[0]//2), 2)
	#print(curve.shape)
	length = np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2,axis=1))) #  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	l1 = np.sqrt(     (start[0] - curve[0][0])**2 + (start[1] - curve[0][1])**2      )
	l2 = np.sqrt((stop[0] - curve[len(curve) - 1][0])**2 + (stop[1] - curve[len(curve) - 1][1])**2)
	return length + l1 + l2


def cost_func(path):
	path = path.reshape(int(path.shape[0]//2), 2)
	start = path[0]
	stop = path[-1]
	length = calc_length(path,start,stop)
	length_weight = 1.0
	length_cost = length * length_weight
	cost = length_cost
	return cost

def optimize_path(path):
	bnds = []
	lb= []
	ub = []
	z = 1.0
	for i in range(0, len(path)):
		if i == 0 or i == len(path) - 1:
			lim_r = 0
		else:
			lim_r = z
		xlb = path[i][0] - lim_r
		xub = path[i][0] + lim_r
		ylb = path[i][1] - lim_r
		yub = path[i][1] + lim_r
		bx = [xlb, xub]
		by = [ylb, yub]
		bnds.append([bx,by])
	bnds = np.asarray(bnds)
	bnds = bnds.reshape(len(path)*2, 2)
	print(bnds.shape)
	nm = "Nelder-Mead"
	result = minimize(cost_func, path, bounds=bnds ,method="BFGS")
	#result = shgo(cost_func, bounds=bnds)
	path= result.x
	path = path.reshape((int(path.shape[0]//2), 2))
	return path

def main():
	print(__file__)
	start = 0
	finish = 5
	num_steps = 10
	x = np.linspace(start, finish, num=int(num_steps), endpoint= True).T
	radius = 5
	y = np.sqrt(radius**2 - x**2).T

	points = np.empty(shape=(num_steps,2))
	points [:,0] = x
	points [:,1] = y
	#print(points)
	plt.plot(points[:,0],points[:,1])
	path = points

	path = optimize_path(points)
	plt.plot(path[:,0],path[:,1])

	plt.show()
if __name__ == "__main__":
	main()

