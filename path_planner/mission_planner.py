import os
from datetime import datetime

import dronekit_sitl
import navpy as nv
import numpy as np
from dronekit import connect, Command
from pymavlink import mavutil

# lat_ref,  lon_ref, alt_ref = 23.8620809, 90.3611941, 0.0
lat_ref, lon_ref, alt_ref = 23.8619973139143, 90.3610950708389, 0.0


def genrate_mission(path):
    sitl = dronekit_sitl.start_default()

    connection_string = sitl.connection_string()
    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=False)

    def download_mission():
        """
        Downloads the current mission and returns it in a list.
        It is used in save_mission() to get the file information to save.
        """
        print(" Download mission from vehicle")
        missionlist = []
        cmds = vehicle.commands
        cmds.download()
        cmds.wait_ready()
        for cmd in cmds:
            missionlist.append(cmd)
            print(cmd.seq, cmd.current, cmd.frame, cmd.command, cmd.param1, cmd.param2, cmd.param3, cmd.param4, cmd.x,
                  cmd.y, cmd.z, cmd.autocontinue)
        return missionlist

    def save_mission(aFileName):
        """
        Save a mission in the Waypoint file format
        (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).
        """
        print("\nSave mission from Vehicle to file: %s" % aFileName)
        # Download mission from vehicle
        missionlist = download_mission()
        # Add file-format information
        output = 'QGC WPL 110\n'
        # Add home location as 0th waypoint
        home = vehicle.home_location
        output += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            0, 1, 0, 16, 0, 0, 0, 0, lat_ref, lon_ref, alt_ref, 1)
        # Add commands
        for cmd in missionlist:
            commandline = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                cmd.seq, cmd.current, cmd.frame, cmd.command, cmd.param1, cmd.param2, cmd.param3, cmd.param4, cmd.x,
                cmd.y,
                cmd.z, cmd.autocontinue)
            output += commandline
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        path = os.getcwd()
        output_path = path + "\\waypoint_mission\\"
        listdir = os.listdir(path)
        if 'waypoint_mission' not in listdir:
            os.mkdir(output_path)
        os.mkdir(output_path + dt_string)
        os.chdir(output_path + dt_string)
        with open(aFileName, 'w') as file_:
            print(" Writing Mission to file")
            file_.write(output)

    lla = np.asarray(nv.ned2lla(path, lat_ref, lon_ref, alt_ref, latlon_unit='deg', alt_unit='m', model='wgs84')).T
    cmds = vehicle.commands
    cmds.clear()
    print("Genarating new mission file")

    cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0,
                  0, 0, 0, 0, 0, 5)
    cmds.add(cmd)
    cmds.add(cmd)
    for i in range(len(lla)):
        cmds.add(
            Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0,
                    0, 0, 0, 0, float(lla[i][0]), float(lla[i][1]), -float(lla[i][2])))

    cmds.add(
        Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0,
                0, 0, 0, 0))
    cmds.upload()

    save_mission('waypoints.txt')
    print("Waypoint file genarated successfully.")
    vehicle.close()
    sitl.stop()


if __name__ == "__main__":
    print(__file__)
    waypoints = [[0, 0, 5], [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5], [0, 0, 5]]
    genrate_mission(waypoints)
