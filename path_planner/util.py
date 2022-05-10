import matplotlib.pyplot as plt
import numpy as np

lim = 60


def plot_path_3d(paths, labels, obstacles):
    fig = plt.figure('3D plots')
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X Axis (Meters)')
    ax.set_ylabel('Y Axis(Meters)')
    ax.set_zlabel('Z Axis(Meters)')
    ax.set_title('Flight path with obstacles')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    for path, label in zip(paths, labels):
        ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=np.arange(0, len(path), 1), s=0.9)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label=label)
    for obstacle in obstacles:
        Xc, Yc, Zc = data_for_cylinder_along_z(obstacle[0], obstacle[1], obstacle[2], lim)
        # ax.plot_surface(Xc, Yc, Zc, cmap=cm.jet, alpha=0.9, linewidth=0.9, label='Obstacles')
        surf = ax.plot_surface(Xc, Yc, Zc, alpha=0.9, linewidth=0.9)

    ax.legend()
    plt.show()


def plot_path_2d(paths, labels, obstacles):
    plt.axis("equal")
    plt.xlabel('X Axis (Meters)')
    plt.ylabel('Y Axis(Meters)')
    plt.title('Flight path with obstacles')
    for path, label in zip(paths, labels):
        plt.scatter(path[:, 0], path[:, 1], c=np.arange(0, len(path), 1), s=0.9)
        plt.plot(path[:, 0], path[:, 1], label=label, linewidth=0.9)

    for obstacle in obstacles:
        plt.gca().add_artist(plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='k', label='Obstacles'))
    plt.legend()
    plt.show()


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


if __name__ == "__main__":
    print(__file__)
