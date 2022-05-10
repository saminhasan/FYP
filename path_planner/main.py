from mission_planner import *
from optimizer import *
from util import *
from scipy import interpolate

def plan_path():
	obstacles = np.array([[0, 10, 4.0], [0, 25, 2.0], [0, 40, 3.0]])  # x, y, r; h = inf

	step_size = 16.0	 # in meters

	start = np.array([0.0, 0.0, 0.0])  # starting co-ordinate

	finish = np.array([0.0, 72.0, 72.0])  # Finishing co-ordinate
	base_path, initial_path = genrate_initial_path(start, finish, step_size)
	#print(initial_path.ndim, initial_path.shape)
	path = base_path
	iteration = 0
	while True:
		path = np.array(optimize_path(path, obstacles, max_iter=len(base_path)*200))
		path = path.reshape((int(path.shape[0] // 3), 3))

		#print(path.ndim, path.shape)
		if not bool(calc_collision_cylinder(path, obstacles)):
			break
		iteration +=1

		print("iteration : ", iteration)
		if iteration >= 2:
			break
	print('obstacles : ', obstacles.tolist())
	print('Start : ', start.tolist())
	print('Stop : ', finish.tolist())
	print('Step Size : ', step_size)

	print("-----------------------------------For Initial Path-----------------------------------")
	calc_cost(initial_path.flatten(), obstacles, True)
	print("collision  state in initial guess : ", bool(calc_collision_cylinder(initial_path, obstacles)))

	print("-----------------------------------For Optimized Path---------------------------------")
	calc_cost(path, obstacles, True)

	#path = path.reshape((int(path.shape[0] // 3), 3))

	print("collision  state in final result : ", bool(calc_collision_cylinder(path, obstacles)))

	#plot_path_3d([initial_path, path], ['Initial path', 'Optimized path'], obstacles)

	plot_path_2d([initial_path, path], ['Initial path', 'Optimized path'], obstacles)
	print('initial_path : ', initial_path.tolist())
	print('path : ', path.tolist())
	return path


if __name__ == "__main__":
	path = plan_path()
	#genrate_mission(path)
