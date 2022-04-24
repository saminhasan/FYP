import numpy as np
from util import *
from optimizer import *
from mission_planner import *

def plan_path():

	obstacles = np.array([[0,25,1.0]]) # x, y, r; h = inf


	step_size = 5.0 # in meters
	
	start = np.array([0.0, 0.0, 5.0]) 	# starting co-ordinate 

	finish = np.array([0.0,50.0,50.0])	# Finishing co-ordinate 

	base_path, initial_path = genrate_initial_path(start, finish, step_size)
	
	path = optimize_path(base_path, obstacles, max_iter=1000)

	
	print("-----------------------------------For Initial Path-----------------------------------")
	calc_cost(initial_path.flatten(), obstacles, True)
	print("collision  state in initial guess : ", bool(calc_collision_cylinder(initial_path, obstacles)))

	print("-----------------------------------For Optimized Path---------------------------------")
	calc_cost(path,  obstacles, True)

	path = path.reshape((int(path.shape[0]//3), 3))

	print("collision  state in final result : ", bool(calc_collision_cylinder(path, obstacles)))



	
	plot_path(initial_path, 'Initial path')
	plot_path(path, 'Optimized path')
	plot_obstacles_cylinder(obstacles)
	show_plots()
	return path
if __name__ == "__main__":
	path = plan_path()
	genrate_mission(path)
