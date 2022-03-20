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
	
	
	return rosenbrock(path)* 10
	

	
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
		zlb = path[i][1] - lim_r
		zub = path[i][1] + lim_r
		bx = [xlb, xub]
		by = [ylb, yub]
		bz = [zlb, zub]
		bnds.append([bx,by,bz])
	bnds = np.asarray(bnds)
	bnds = bnds.reshape(len(path)*3, 2)
	#print(bnds.shape)
	nm = "Nelder-Mead"
	pw = "Powell"
	lb = "L-BFGS-B"
	tnc = "TNC"
	tr="trust-constr" 
	sl = "SLSQP"
	result = minimize(cost_func, path, method=lb, bounds=bnds)
	#result = brute(cost_func, path)
	print(result)
	return result.x

step_size = 2.0 # in meters
start = np.array([-2,3,0])
finish = np.array([3,3,10])
linear_distance = np.linalg.norm(finish-start)
num_steps = linear_distance //  step_size + 1
initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)

#c = cost_func(initial_path)
#print(c)
'''
initial_path = np.array(
[[ 1.12921675e+00,  1.25984616e+00,  1.99792697e-13],
 [ 1.04546265e+00,  1.09112034e+00,  1.16279070e-01],
 [ 7.92199331e-01, 6.37526966e-01, 2.32558140e-01],
 [ 9.44819014e-01,  8.65317375e-01,  3.48837209e-01],
 [ 1.13484253e+00,  1.29985092e+00,  4.65116279e-01],
 [ 1.00645561e+00,  1.01578063e+00,  5.81395349e-01],
 [ 1.08837232e+00,  1.17470155e+00,  6.97674419e-01],
 [ 9.61956361e-01,  9.29854445e-01,  8.13953488e-01],
 [ 1.13552294e+00,  1.27371933e+00,  9.30232558e-01],
 [ 8.48121972e-01,  7.20251236e-01, 1.04651163e+00],
 [ 9.62891206e-01,  9.29028452e-01,  1.16279070e+00],
 [ 9.62088327e-01,  9.22735788e-01,  1.27906977e+00],
 [ 9.83587665e-01,  9.60680621e-01,  1.39534884e+00],
 [ 9.41606336e-01,  8.80035374e-01,  1.51162791e+00],
 [ 9.44890218e-01,  8.94457506e-01,  1.62790698e+00],
 [ 9.75811754e-01,  9.54284634e-01,  1.74418605e+00],
 [ 1.02519910e+00,  1.05877809e+00,  1.86046512e+00],
 [ 1.12581260e+00,  1.27316239e+00,  1.97674419e+00],
 [ 1.29213341e+00,  1.66739497e+00,  2.09302326e+00],
 [ 1.12691297e+00,  1.26490984e+00,  2.20930233e+00],
 [ 1.17760342e+00,  1.37992646e+00,  2.32558140e+00],
 [ 1.11626082e+00,  1.23018491e+00,  2.44186047e+00],
 [ 1.14287865e+00,  1.31095134e+00,  2.55813953e+00],
 [ 1.13716339e+00,  1.29074784e+00,  2.67441860e+00],
 [ 1.11612340e+00,  1.24542372e+00,  2.79069767e+00],
 [ 1.09069168e+00,  1.17081168e+00,  2.90697674e+00],
 [ 1.10926900e+00,  1.22589102e+00,  3.02325581e+00],
 [ 1.02991533e+00,  1.05741095e+00,  3.13953488e+00],
 [ 1.16824857e+00,  1.34673597e+00,  3.25581395e+00],
 [ 1.12780876e+00,  1.26302328e+00,  3.37209302e+00],
 [ 1.27666816e+00,  1.62827709e+00,  3.48837209e+00],
 [ 1.27706336e+00,  1.62881734e+00,  3.60465116e+00],
 [ 1.71705065e+00,  2.93980087e+00,  3.72093023e+00],
 [ 1.51219462e+00,  2.28575892e+00,  3.83720930e+00],
 [ 1.19528576e+00,  1.42047665e+00,  3.95348837e+00],
 [-1.47360720e+00,  2.18021649e+00,  4.06976744e+00],
 [-2.39867404e+00,  5.77214960e+00,  4.18604651e+00],
 [-2.21629105e+00,  4.93751674e+00,  4.30232558e+00],
 [-2.33426594e+00,  5.45824077e+00,  4.41860465e+00],
 [-1.33653426e+00,  1.79582778e+00,  4.53488372e+00],
 [-1.81414365e+00,  3.29203956e+00,  4.65116279e+00],
 [-2.11902443e+00,  4.47303231e+00,  4.76744186e+00],
 [-2.32035834e+00,  5.36113477e+00,  4.88372093e+00],
 [-2.13236309e+00,  4.54880160e+00,  5.00000000e+00]]
)
'''
path = initial_path
path = optimize_path(initial_path)
path = path.reshape((int(path.shape[0]//3), 3))
#print(path)
fig = plt.figure('3D plots')
ax = plt.axes(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D plot Title')
ax.scatter(initial_path[:,0],initial_path[:,1],initial_path[:,2], c = initial_path[:,2])
ax.plot(initial_path[:,0],initial_path[:,1],initial_path[:,2])
ax.scatter(path[:,0],path[:,1],path[:,2], c = np.arange(0,len(path), 1))
ax.plot(path[:,0],path[:,1],path[:,2])





plt.show()

#print(linear_distance, c)
#print(path,"\n","\n",len(path),"\n",np.diff(path,axis=0))

