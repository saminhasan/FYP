
#new
from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import time
import math
from pymavlink import mavutil
import argparse	 
'''new'''
parser = argparse.ArgumentParser(description='Demonstrates basic mission operations.')
parser.add_argument('--connect', 
				   help="vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None


#Start SITL if no connection string specified
if not connection_string:
	import dronekit_sitl
	sitl = dronekit_sitl.start_default()
	connection_string = sitl.connection_string()


# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)
#new

import numpy as np
import navpy as nv
from matplotlib import cm
from scipy.optimize import *
from sim import energy_from_waypoints
import matplotlib.pyplot as plt

#import pyximport; pyximport.install()

lat_ref,  lon_ref, alt_ref = 23.8620809, 90.3611941, 0.0
np.random.seed(0)
obstacles = [[0.6,25,25,0.5]] #x, y, z, r
fig = plt.figure('3D plots')
ax = plt.axes(projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Flight Path')
ax.set_xlim(-30, 30)
ax.set_ylim(0, 60)
ax.set_zlim(0, 60)
		
def calc_length(curve):

	length = np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2,axis=1))) #  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	return length
	
def calc_time(path):

	# constant velocity model, very uselss , place holder for now 
	vel = 10 # m/s
	time = calc_length(path) / vel
	return time
	


	
def calc_energy_consumption(path):
	
	#path = path.reshape((int(path.shape[0]//3), 3))
	#energy = energy_from_waypoints(path)
	
	
	return 0#energy
	
def calc_collision(path, obstacles):

	collision_cost = 500
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
	path = path.reshape((int(path.shape[0]//3), 3))

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
	result = minimize(cost_func, path, method=nm, bounds=bnds ,options={'maxiter':len(path) * 900 , 'adaptive': True, 'disp': True })
	#result = minimize(fun, x0, args=(), method='Nelder-Mead', bounds=None, tol=None, callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
	
	#result = minimize(cost_func, path, method=sl)
	#result = brute(cost_func, path)
	print("optimization : ", result.success)
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
		ax.plot_surface(x, y, z, color='b', alpha=0.9)
		
		
'''new'''

def download_mission():
	"""
	Downloads the current mission and returns it in a list.
	It is used in save_mission() to get the file information to save.
	"""
	print(" Download mission from vehicle")
	missionlist=[]
	cmds = vehicle.commands
	cmds.download()
	cmds.wait_ready()
	for cmd in cmds:
		missionlist.append(cmd)
		print(cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
	return missionlist

def save_mission(aFileName):
	"""
	Save a mission in the Waypoint file format 
	(http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).
	"""
	print("\nSave mission from Vehicle to file: %s" % aFileName)	
	#Download mission from vehicle
	missionlist = download_mission()
	#Add file-format information
	output='QGC WPL 110\n'
	#Add home location as 0th waypoint
	home = vehicle.home_location
	output+="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0,1,0,16,0,0,0,0,lat_ref,lon_ref,alt_ref,1)
	#Add commands
	for cmd in missionlist:
		commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
		output+=commandline
	with open(aFileName, 'w') as file_:
		print(" Write mission to file")
		file_.write(output)
		
		
def printfile(aFileName):
	"""
	Print a mission file to demonstrate "round trip"
	"""
	print("\nMission file: %s" % aFileName)
	with open(aFileName) as f:
		for line in f:
			print(' %s' % line.strip())	
'''new'''

def main():
	'''
	genarate a straight line path as initial solution.
	feed it to optimizer
	plot output
	
	'''

	step_size = 4.0 # in meters
	# starting co-ordinate 
	start = np.array([0,0,0]) 
	# Finishing co-ordinate 
	finish = np.array([0,50,50])
	# shorttest path distance,  straight line path
	linear_distance = np.linalg.norm(finish-start)
	#calculate number of steps to divide path into
	num_steps = linear_distance //  step_size + 1

	# The guess path for the optimizer to start workinig with, a straight line connecting start and finish line
	initial_path = np.linspace(start, finish, num=int(num_steps), endpoint= True)
	noise = np.random.randn(*initial_path.shape) * 1.0
	base_path =  initial_path + noise * 0.05
	initial_path +=  noise

	#print(initial_path, noise)
	print("collision : ", bool(calc_collision(initial_path, obstacles)))
	path = optimize_path(base_path)
	#reshapinig  output array  for plotttinig
	#print(path, path.shape)
	c = cost_func(path)
	#print(c)
	path = path.reshape((int(path.shape[0]//3), 3))
	#print(path)

	# plotting 
	print("collision : ", bool(calc_collision(path, obstacles)))

	plot(initial_path, path)
	lla = np.asarray(nv.ned2lla(path, lat_ref, lon_ref, alt_ref, latlon_unit='deg', alt_unit='m', model='wgs84'))
	lla = lla.T#reshape((lla.shape[1],lla.shape[0]))
	#print(lla, lla.shape)
	# cost of stragith line path

	#print(linear_distance, c)
	#print(path,"\n","\n",len(path),"\n",np.diff(path,axis=0))
			
	cmds = vehicle.commands
	print(" Clear any existing commands")
	cmds.clear() 
	print(" Cleared any existing commands")
	time.sleep(5)
	cmds.clear()
	print(" Adding new commands.")
	cmd = Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, 5)
	print((cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue))
	cmds.add(cmd)
	cmds.add(cmd)
	for i in range(len(lla)):
		cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 1, 0, 0, 0, 0, 0, float(lla[i][0]), float(lla[i][1]), -float(lla[i][2])))
	cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0, 0))
	cmds.upload()
	time.sleep(5)
	save_mission('src.txt')
	print(" saved new waypoints.")

	#Close vehicle object before exiting script
	print("Close vehicle object")
	vehicle.close()

	# Shut down simulator if it was started.
	if sitl is not None:
		sitl.stop()
	'''new'''
if __name__ == "__main__":
	for _ in range(1):
		main()

