import numpy as np
import scipy.interpolate as interp
from scipy import integrate

from FileProcessing import *
from LogFileStructure import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from collections import defaultdict


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


if __name__ == '__main__':
	file_name = 'flight_1.log'
	lf = LogFile(file_name)
	data = lf.extract_data()
	time = []
	rcin = data['RCIN']
	rcinTime = np.asarray(rcin['TimeUS'][1:]).T * 1e-6 # microseconds to seconds
	rc8 = np.asarray(rcin['C8'][1:]).T
	#plt.scatter(rcinTime, rc8)
	#plt.show()
	
	for i, instance in enumerate(rcin['C8']):
		if float(instance) > 1833.3:

			time.append(float(rcin['TimeUS'][i]))
	auto_start, auto_end = min(time), max(time)

	xkf1 = data['XKF1']
	xkf10, xkf11 = split_d(xkf1, 'C')
	x = np.asarray(xkf10['PN'][1:]).T
	y = np.asarray(xkf10['PE'][1:]).T
	z = np.asarray(xkf10['PD'][1:]).T
	t = np.asarray(xkf10['TimeUS'][1:])
	path = np.hstack((x,y,z))
	#X = np.vsplit(x, np.where(path[:, 3] >= auto_start and path[:, 3] <= auto_end))
	#X = x[np.where(x==x[x>auto_start].min())]
	#T_max = np.where(t==t[t<auto_end].max())[0] - 2
	#T_min = np.where(t==t[t>auto_start].min())[0] + 1
	#path = path[int(T_min): int(T_max)]
	#print(t[T_max]- t[T_min])
	#print(X)
	path = path.reshape((int(path.shape[0]//3), 3))
	print("Path Length : ", calc_length(path), ' Meters')
	plt.scatter(path[:,0],path[:,1])
	plt.show()
	'''
	Volt	Battery voltage in volts * 100
	Curr	Current drawn from the battery in amps * 100
	TimeUS	Time stamp for messages in microseconds 
	'''
	Voltage = np.asarray(data['BAT']['Volt'][1:]).T
	Current = np.asarray(data['BAT']['Curr'][1:]).T 
	Time = np.asarray(data['BAT']['TimeUS'][1:]).T * 1e-6 # microseconds to seconds
	Power = np.array([Voltage[i] * Current[i]  for i in range(len(Time))])
	Energy =  np.asarray(data['BAT']['EnrgTot'][1:]).T * 3.6
	consumption = integrate.cumtrapz(Power, Time)[-1] * 1e-3
	print("Energy Consumption : ", Energy[-1], "  Joules" )
	print("Energy Consumption : ", consumption, "  Joules" )
	plt.plot(Time, Voltage, label ='Voltage (Volt)')
	plt.plot(Time, Current, label ='Current (Ampere)')
	plt.plot(Time, Power, label ='Power (Watt)')
	plt.legend()
	plt.show()