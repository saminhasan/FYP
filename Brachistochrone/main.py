import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def cost_func(path):
	
	return cost
	
	
def	optimize_path(path):
	result = minimize(cost_func, path, method="SLSQP")
	path= result.x
	path = path.reshape((int(path.shape[0]//3), 3))
	return path
		
		
def main():
	print(__file__)
	
	step_size = 0.01 # in meters
	start = np.array([0,0,5])
	finish = np.array([5,0,0])
	linear_distance = np.linalg.norm(finish-start)
	num_steps = linear_distance //  step_size + 1
	initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)
	path = initial_path
	#path = optimize_path(initial_path)
	#path = path.reshape((int(path.shape[0]//3), 3))
	print(path)
	fig = plt.figure('3D plots')
	ax = plt.axes(projection='3d')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_title('3D plot Title')
	#ax.scatter(initial_path[:,0],initial_path[:,1],initial_path[:,2], c = initial_path[:,2])
	#ax.plot(initial_path[:,0],initial_path[:,1],initial_path[:,2])
	ax.scatter(path[:,0],path[:,1],path[:,2], c = np.arange(0,len(path), 1))
	ax.plot(path[:,0],path[:,1],path[:,2])
	plt.show()

if __name__ == "__main__":
	main()

