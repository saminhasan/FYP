import numpy as np
from quad_sim import *

i=0 
def calc_length(path):

	#  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	return np.sum(np.sqrt(np.sum((path[:-1] - path[1:])**2,axis=1)))


def calc_time(path):

	vel = 10 # m/s# constant velocity model, very uselss , place holder for now 
	time = calc_length(path) / vel
	return time


def calc_energy_consumption(path):
	energy = energy_from_waypoints(path)
	return energy


def calc_collision_cylinder(path, obstacles):

	total_cost = 0

	for i in range(len(path) -1):
		for j in range(len(obstacles)):
			p1 = path[i][0:2]
			p2 = path[i + 1][0:2]
			p3 = obstacles[j][0:2]
			r = obstacles[j][2]
			d1 = np.sqrt((p1[0] -p3[0])**2 + (p1[1] -p3[1])**2)
			d2 = np.sqrt((p2[0] -p3[0])**2 + (p2[1] -p3[1])**2)
			d3 = np.abs(np.linalg.norm(np.cross(p2-p1, p1-p3)))/np.linalg.norm(p2-p1)
			if d1 < r:
				total_cost += 999.0 / d1**2 
			if d2 < r:
				total_cost += 999.0 / d2**2 
			if d3 < r:
				total_cost += 999.0 / d3**2  
			#print(d1, d2, d3)

	return total_cost

def calc_cost(path, obstacles, verbose=False):

	path = path.reshape((int(path.shape[0]//3), 3))

	length = calc_length(path)
	length_weight = 1.0
	length_cost = length * length_weight
	if verbose:
		print("Path Length : ", length, ' meters.')

	vel = 10  # m/s
	time = length / vel 
	time_weight = 1.0
	time_cost = time * time_weight
	if verbose:
		print("Minimum Time  : ", time , ' seconds.' )

	energy_consumption = 0#calc_energy_consumption(path)
	energy_weight = 1.0
	energy_cost = energy_consumption * energy_weight
	if verbose:
		print("Energy Consumption : ", calc_energy_consumption(path) * 25, "joules.")

	collision = calc_collision_cylinder(path,obstacles)
	collision_weight = 1.0
	collision_cost = collision * collision_weight
	if verbose:
		print("Collision Cost : ", collision_cost)

	cost = length_cost + time_cost + energy_cost + collision_cost
	if verbose:
		print("Path Cost : ", cost)

	return cost

if __name__ == "__main__":
	print(__file__)