import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

lim=50
fig = plt.figure('3D plots')
ax = plt.axes(projection='3d')
ax.set_xlabel('X Axis (m)')
ax.set_ylabel('Y Axis(m)')
ax.set_zlabel('Z Axis(m)')
ax.set_title('Flight Path')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(0, lim)

def plot_path(path, label):

	ax.scatter(path[:,0],path[:,1],path[:,2], c = np.arange(0,len(path), 1))
	ax.plot(path[:,0],path[:,1],path[:,2],label=label)
	plt.legend()


def data_for_cylinder_along_z(center_x,center_y,radius,height_z):

	z = np.linspace(0, height_z, 50)
	theta = np.linspace(0, 2*np.pi, 50)
	theta_grid, z_grid=np.meshgrid(theta, z)
	x_grid = radius*np.cos(theta_grid) + center_x
	y_grid = radius*np.sin(theta_grid) + center_y
	return x_grid,y_grid,z_grid


def plot_obstacles_cylinder(obstacles):

	for j in range(len(obstacles)):
		Xc,Yc,Zc = data_for_cylinder_along_z(obstacles[j][0], obstacles[j][1],obstacles[j][2],100)
		ax.plot_surface(Xc, Yc, Zc, cmap=cm.coolwarm, alpha=0.9)
		
def show_plots():
	plt.show()

if __name__ == "__main__":
	print(__file__)