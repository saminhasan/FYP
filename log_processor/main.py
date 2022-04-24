import numpy as np
import scipy.interpolate as interp
from scipy import integrate

from FileProcessing import *
from LogFileStructure import *

#from util import *
from optimizer import *
from mission_planner import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from collections import defaultdict

obstacles = np.array([[100,100,1.0]]) # x, y, r; h = inf


def split_d(d, key):
	include = defaultdict(list)
	exclude = defaultdict(list)
	output = None
	for i, instance in enumerate(d[key]):
		instance = float(instance)
		'''

		if not(instance == 1.0	or	instance == 0.0):
			print(i, instance)
			#print(d[i])
			for key in d:
				print(key, ' : ', d[key][i])
		'''
		if instance == 1.0:
			output = include
		elif instance == 0.0:
			output = exclude
		else:
			pass
		if output is not None :
			for k, v in d.items():
				if k == key:
					continue
				output[k].append(v[i])

	return include, exclude


def calc_length(path):

	#  https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
	return np.sum(np.sqrt(np.sum((path[:-1] - path[1:])**2,axis=1)))


def resizer(ref_arr, data_arr):
	New_Power = interp.interp1d(np.arange(len(data_arr)),data_arr)
	pp = New_Power(np.linspace(0,len(ref_arr)-1,len(ref_arr)))
	return pp
	
def plot_path(x, y, z, power):
	fig = plt.figure('3D plots')
	ax = plt.axes(projection='3d')
	ax.set_xlabel('X Axis (Meter)')
	ax.set_ylabel('Y Axis (Meter)')
	ax.set_zlabel('Z Axis (Meter)')
	ax.set_title('Power consumption and Flight path')
	ax.xaxis.pane.fill = False
	ax.xaxis.pane.set_edgecolor('white')
	ax.yaxis.pane.fill = False
	ax.yaxis.pane.set_edgecolor('white')
	ax.zaxis.pane.fill = False
	ax.zaxis.pane.set_edgecolor('white')
	ax.grid(False)
	pp = np.asarray(resizer(x,Power))
	New_Power = []
	m = 0
	for i in range(0,int(len(Power)/2)-1):
		New_Power.append(Power[m])
		m = m + 2
	#npp = interp.interp1d(np.arange(len(New_Power)),New_Power)
	#pp = npp(np.linspace(0,len(u)-1,len(u)))
	pp = np.asarray(resizer(x,Power))
	norm = np.linalg.norm(pp)
	normal_array = pp / norm
	size = normal_array * 100
	ax.scatter(x[0], y[0], z[0], s = 100)

	s = ax.scatter(x, y, z ,s = size, marker = 'o' , c = pp,cmap = cm.jet, linewidths = 0.25,edgecolors = 'k') 
	c_bar = fig.colorbar(s, ax = ax)
	c_bar.ax.set_title('Power Consumption (Watt)')
	plt.show()

if __name__ == '__main__':
	file_name = 'test_4.log'
	lf = LogFile(file_name)
	data = lf.extract_data()
	rctime = []
	rcin = data['RCIN']
	rcinTime = np.asarray(rcin['TimeUS'][1:])* 1e-6 # microseconds to seconds
	rc8 = np.asarray(rcin['C8'][1:])


	plt.plot(rcinTime, rc8)
	#plt.xlabel("Time (Seconds)")
	#plt.ylabel("PWM (RC Input) ")

	#plt.show()
	

	for i, instance in enumerate(rcin['C8']):
		if float(instance) > 1833.3:

			rctime.append(float(rcin['TimeUS'][i]))
	auto_start, auto_end = min(rctime), max(rctime)
	Voltage = np.asarray(data['BAT']['Volt'][1:]).T
	Current = np.asarray(data['BAT']['Curr'][1:]).T 
	Time = np.asarray(data['BAT']['TimeUS'][1:]).T * 1e-6 # microseconds to seconds
	Power = np.array([Voltage[i] * Current[i]  for i in range(len(Time))])
	Energy =  np.asarray(data['BAT']['EnrgTot'][1:]).T * 3600
	consumption = integrate.cumtrapz(Power, Time)[-1] 
	xkf1 = data['XKF1']
	xkf10, xkf11 = split_d(xkf1, 'C')
	x = np.asarray(xkf10['PN'][1:]).T
	y = np.asarray(xkf10['PE'][1:]).T
	z = -np.asarray(xkf10['PD'][1:]).T
	t = np.asarray(xkf10['TimeUS'][1:])
	path = np.hstack((x,y,z))
	plot_path(x, y, z, Power)


	T_max = np.where(t==t[t<auto_end].max())[0] - 2
	T_min = np.where(t==t[t>auto_start].min())[0] + 1
	#path = path[int(T_min): int(T_max)]
	#print(X)
	path = path.reshape((int(path.shape[0]//3), 3))
	print("Path Length (Experimental) : ", calc_length(path), ' Meters')
	print("Time Required (Experimental) : ", (t[T_max]- t[T_min])[0] * 1e-6 , ' Seconds')

	#plt.scatter(path[:,0],path[:,1])
	#plt.show()
	#Volt	Battery voltage in volts * 100
	#Curr	Current drawn from the battery in amps * 100
	#TimeUS	Time stamp for messages in microseconds 
	#predicted_consumption = energy_from_waypoints(path)
	print("Energy Consumption (Pixhawk, Experimental) : ", Energy[-1], "  Joules" )
	print("Energy Consumption (LogFile, Experimental) : ", consumption, "  Joules" )

	plt.plot(Time, Voltage, label ='Voltage (Volt)')
	plt.plot(Time, Current, label ='Current (Ampere)')
	plt.plot(Time, Power, label ='Power (Watt)')
	plt.xlabel("Time (Seconds)")
	plt.legend()
	plt.show()
	
