from math import cos, sin

from numpy import array, sqrt, matmul, linalg
from scipy import integrate


class TrajectoryGenerator():
    def __init__(self, start_pos, des_pos, T, start_vel=[0, 0, 0], des_vel=[0, 0, 0], start_acc=[0, 0, 0],
                 des_acc=[0, 0, 0]):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]

        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]

        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]

        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]

        self.T = T

    def solve(self):
        A = array(
            [[0, 0, 0, 0, 0, 1],
             [self.T ** 5, self.T ** 4, self.T ** 3, self.T ** 2, self.T, 1],
             [0, 0, 0, 0, 1, 0],
             [5 * self.T ** 4, 4 * self.T ** 3, 3 * self.T ** 2, 2 * self.T, 1, 0],
             [0, 0, 0, 2, 0, 0],
             [20 * self.T ** 3, 12 * self.T ** 2, 6 * self.T, 2, 0, 0]
             ])

        b_x = array(
            [[self.start_x],
             [self.des_x],
             [self.start_x_vel],
             [self.des_x_vel],
             [self.start_x_acc],
             [self.des_x_acc]
             ])

        b_y = array(
            [[self.start_y],
             [self.des_y],
             [self.start_y_vel],
             [self.des_y_vel],
             [self.start_y_acc],
             [self.des_y_acc]
             ])

        b_z = array(
            [[self.start_z],
             [self.des_z],
             [self.start_z_vel],
             [self.des_z_vel],
             [self.start_z_acc],
             [self.des_z_acc]
             ])

        self.x_c = linalg.solve(A, b_x)
        self.y_c = linalg.solve(A, b_y)
        self.z_c = linalg.solve(A, b_z)


# Simulation parameters
g = 9.81
m = 2.6
Ixx = 1
Iyy = 1
Izz = 1
T = 5

# Proportional coefficients
Kp_x = 1
Kp_y = 1
Kp_z = 1
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25

# Derivative coefficients
Kd_x = 10
Kd_y = 10
Kd_z = 1


def quad_sim(x_c, y_c, z_c,
             wp):  # this looks like the outpuyt of the trajectory planner, trajectory planner takes wayopints as input.
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    run = len(wp)
    x_pos = wp[0][1]
    y_pos = wp[0][1]
    z_pos = wp[0][2]
    x_vel = 0
    y_vel = 0
    z_vel = 0
    x_acc = 0
    y_acc = 0
    z_acc = 0
    roll = 0
    pitch = 0
    yaw = 0
    roll_vel = 0
    pitch_vel = 0
    yaw_vel = 0

    des_yaw = 0

    dt = 0.11
    t = 0

    i = 0
    n_run = run - 1  # waypoints
    irun = 0
    ppp, ttt = [], []
    while True:
        while t <= T:
            des_z_pos = calculate_position(z_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)

            thrust = m * (g + des_z_acc + Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel))
            power = thrust * sqrt(x_vel ** 2 + y_vel ** 2 + z_vel ** 2)
            ppp.append(power[0])
            ttt.append(t)
            roll_torque = Kp_roll * (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)

            pitch_torque = Kp_pitch * (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)

            yaw_torque = Kp_yaw * (des_yaw - yaw)

            roll_vel += roll_torque * dt / Ixx
            pitch_vel += pitch_torque * dt / Iyy
            yaw_vel += yaw_torque * dt / Izz

            roll += roll_vel * dt
            pitch += pitch_vel * dt
            yaw += yaw_vel * dt

            R = rotation_matrix(roll, pitch, yaw)
            acc = (matmul(R, array([0, 0, thrust.item()]).T) - array([0, 0, m * g]).T) / m
            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]
            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt
            x_pos += x_vel * dt
            y_pos += y_vel * dt
            z_pos += z_vel * dt

            t += dt

        t = 0
        i = (i + 1) % run
        irun += 1
        if irun >= n_run:
            break
    energy_consumption = integrate.cumtrapz(ppp, ttt, initial=0)[-1]

    return energy_consumption


def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the position

    Returns
        Position
    """
    return c[0] * t ** 5 + c[1] * t ** 4 + c[2] * t ** 3 + c[3] * t ** 2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the velocity

    Returns
        Velocity
    """
    return 5 * c[0] * t ** 4 + 4 * c[1] * t ** 3 + 3 * c[2] * t ** 2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the acceleration

    Returns
        Acceleration
    """
    return 20 * c[0] * t ** 3 + 12 * c[1] * t ** 2 + 6 * c[2] * t + 2 * c[3]


def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the ZYX rotation matrix.

    Args
        Roll: Angular position about the x-axis in radians.
        Pitch: Angular position about the y-axis in radians.
        Yaw: Angular position about the z-axis in radians.

    Returns
        3x3 rotation matrix as NumPy array
    """
    return array(
        [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
          sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
         [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),
          -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
         ])


def energy_from_waypoints(wp):
    """
    Calculates the x, y, z coefficients for the four segments
    of the trajectory
    """
    x_coeffs = [[] for _ in range(len(wp))]
    y_coeffs = [[] for _ in range(len(wp))]
    z_coeffs = [[] for _ in range(len(wp))]
    start_vel = [0, 0, 0]
    des_vel = [0, 0, 0]
    for i in range(len(wp) - 1):
        traj = TrajectoryGenerator(wp[i], wp[(i + 1) % len(wp)], T)
        traj.solve()
        start_vel = []
        des_vel = []
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c

    return quad_sim(x_coeffs, y_coeffs, z_coeffs, wp)


def main():
    waypoints = [[0, 0, 0], [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5], [0, 0, 5]]
    energy = energy_from_waypoints(waypoints)
    print("Energy Consumption : ", energy, "Joules")


if __name__ == "__main__":
    print(__file__)
    main()
