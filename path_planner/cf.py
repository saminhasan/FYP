import numpy as np
from quad_sim import *
from scipy import interpolate

def smooth(path,ss=None):
	num_true_pts = len(path)
	if ss is None:
		ss = num_true_pts
	tck, u = interpolate.splprep([path[:,0],path[:,1],path[:,2]], s=ss)
	x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
	u_fine = np.linspace(0,1,num_true_pts)
	path = np.array(interpolate.splev(u_fine, tck)).T
	return path

def calc_length(path):
	return np.sum(np.sqrt(np.sum((path[:-1] - path[1:]) ** 2, axis=1)))


def calc_time(path):
	vel = 10
	time = calc_length(path) / vel
	return time


def calc_energy_consumption(path):
	energy = energy_from_waypoints(path)
	return energy


def calc_collision_cylinder(path, obstacles):
	total_cost = 0

	for i in range(len(path) - 1):
		for j in range(len(obstacles)):
			p1 = path[i][0:2]
			p2 = path[i + 1][0:2]
			p3 = obstacles[j][0:2]
			r = obstacles[j][2]
			d1 = np.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
			d2 = np.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
			d3 = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3))) / np.linalg.norm(p2 - p1)
			if d1 < r:
				total_cost += 999.0 / d1 ** 2
			if d2 < r:
				total_cost += 999.0 / d2 ** 2
			if d3 < r:
				total_cost += 999.0 / d3 ** 2

	return total_cost


def calc_smoothness(path):
	dx_dt = np.gradient(path[:, 0])
	dy_dt = np.gradient(path[:, 1])
	dz_dt = np.gradient(path[:, 2])
	velocity = np.array([[dx_dt[i], dy_dt[i], dz_dt[i]] for i in range(dx_dt.size)])
	# print(velocity.shape) # vectorize
	ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt)
	tangent = np.array([1 / ds_dt] * 3).transpose() * velocity
	tangent_x = tangent[:, 0]
	tangent_y = tangent[:, 1]
	tangent_z = tangent[:, 2]
	deriv_tangent_x = np.gradient(tangent_x)
	deriv_tangent_y = np.gradient(tangent_y)
	deriv_tangent_z = np.gradient(tangent_z)
	dT_dt = np.array(
		[[deriv_tangent_x[i], deriv_tangent_y[i], deriv_tangent_z[i]] for i in range(deriv_tangent_x.size)])
	# print(dT_dt.shape) # vectorize
	length_dT_dt = np.sqrt(
		deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y + deriv_tangent_z * deriv_tangent_z)
	normal = np.array([1 / length_dT_dt] * 3).transpose() * dT_dt
	d2s_dt2 = np.gradient(ds_dt)
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	d2z_dt2 = np.gradient(dz_dt)
	# https://en.wikipedia.org/wiki/Curvature
	numerator = (d2z_dt2 * dy_dt - d2y_dt2 * dz_dt) ** 2 + (d2x_dt2 * dz_dt - d2z_dt2 * dx_dt) ** 2 - (
				d2y_dt2 * dx_dt - d2x_dt2 * dy_dt) ** 2
	denominator = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
	curvature = np.sqrt(abs(numerator)) / denominator
	# t_component = np.array([d2s_dt2] * 2).transpose()
	# n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()
	# acceleration = t_component * tangent + n_component * normal
	return np.sum(curvature)


def calc_cost(path, obstacles, verbose=False):
	if path.ndim == 1:
		path = path.reshape((int(path.shape[0] // 3), 3))
	path = smooth(path, len(path)*20)
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

	collision = calc_collision_cylinder(path, obstacles)
	collision_weight = 1.0
	collision_cost = collision * collision_weight

	smoothness = calc_smoothness(path)
	smoothness_weight = 1.0
	smoothness_cost = smoothness * smoothness_weight

	cost = length_cost + time_cost + energy_cost + collision_cost + smoothness_cost

	if verbose:
		print("Path Length : ", length, ' meters.')
		print("Minimum Time	 : ", time, ' seconds.')
		print("Energy Consumption : ", calc_energy_consumption(path) * 25, "joules.")
		print("Collision Cost : ", collision_cost)
		print("SmoothnessCost : ", smoothness_cost)
		print("Path Cost : ", cost)

	return cost


if __name__ == "__main__":
	print(__file__)
