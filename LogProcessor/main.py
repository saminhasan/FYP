from LogFileStructure import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import numpy as np
from collections import defaultdict
import scipy.interpolate as interp
from FileProcessing import *

def split_dict(d, key):
	tkf0,kf0,pn0,pe0,pd0 = [],[],[],[], []
	tkf1,kf1,pn1,pe1,pd1 = [],[],[],[], []

	for i in range(len(d['TimeUS']) // 2):
		if d['C'][i] == 0:
			tkf0.append(d['TimeUS'][i])
			kf0.append(d['GZ'][i])
			pn0.append(d['PN'][i])
			pe0.append(d['PE'][i])
			pd0.append(-d['PD'][i])
		elif d['C'][i] == 1:
			tkf1.append(d['TimeUS'][i])
			kf1.append(d['GZ'][i])
			pn1.append(d['PN'][i])
			pe1.append(d['PE'][i])
			pd1.append(d['PD'][i])

	return tkf0,kf0,tkf1,kf1,pn0,pe0,pd0
def resizer(ref_arr, data_arr):
	New_Power = interp.interp1d(np.arange(len(data_arr)),data_arr)
	pp = New_Power(np.linspace(0,len(ref_arr)-1,len(ref_arr)))
	return pp
	
if __name__ == '__main__':

	file_name = 'flight_1.log'
	lf = LogFile(file_name)
	data = lf.extract_data()
	Voltage = data['BAT']['Volt'][1:]
	Current = data['BAT']['Curr'][1:]
	Time = data['BAT']['TimeUS'][1:]
	Power = np.array([Voltage[i] * Current[i] * 0.1 for i in range(len(Voltage))])
	plt.plot(data['BAT']['TimeUS'][1:], Voltage, label ='Voltage (V)')
	plt.plot(data['BAT']['TimeUS'][1:], Current, label ='Current (A)')
	plt.plot(data['BAT']['TimeUS'][1:], Power, label ='Power (W)')
	#plt.plot(data['BAT']['TimeUS'][1:], Current, label ='Current (A)')

	plt.legend()
	plt.show()
	xkf1 = data['XKF1']
	a,b,c,d,u,v,w = split_dict(xkf1, 'C')
	#plt.plot(u,v, 'b')
	#plt.show()
	#plt.plot(a,b, 'g',label='jeje')
	#plt.legend()
	#plt.show()
	 # 3D Plot of flight data
	latitude = u#analysis.Data['GPS']['Lat'][1:]
	longitude = v#analysis.Data['GPS']['Lng'][1:]
	altitude = w#analysis.Data['GPS']['Alt'][1:]
	fig = plt.figure(figsize=(8,6))
	fig.tight_layout()

	ax = plt.subplot(111, projection='3d')
	fig = plt.figure('3D path')
	ax = plt.axes(projection='3d')
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')
	ax.set_title('Power consumption and Flight path')
	ax.xaxis.pane.fill = False
	ax.xaxis.pane.set_edgecolor('white')
	ax.yaxis.pane.fill = False
	ax.yaxis.pane.set_edgecolor('white')
	ax.zaxis.pane.fill = False
	ax.zaxis.pane.set_edgecolor('white')
	ax.grid(False)
	#ax.plot(u, v, w)
	

	New_Power = []
	m = 0
	for i in range(0,int(len(Power)/2)-1):
		New_Power.append(Power[m])
		m = m + 2
	#npp = interp.interp1d(np.arange(len(New_Power)),New_Power)
	#pp = npp(np.linspace(0,len(u)-1,len(u)))
	pp = np.asarray(resizer(u,Power))
	print(len(u),len(Power),len(pp))
	norm = np.linalg.norm(pp)
	normal_array = pp / norm
	size = normal_array * 100
	s = ax.scatter(latitude, longitude, altitude ,s = size, marker = 'o' , c = pp,cmap = cm.jet, linewidths = 0.25,edgecolors = 'k') 
	c_bar = fig.colorbar(s, ax = ax)
	c_bar.ax.set_title('Power Consumption (Watt)')
	ax.scatter(latitude[0], longitude[0], altitude[0], s = 100)
	plt.show()
	#t1 = resizer(pp,Time)
	#t2 = resizer(Power,Time)
	#plt.plot(t1, pp,label = 'reference')
	#plt.plot(t2,Power,label = 'interplorated')
	#plt.legend()
	#plt.show()