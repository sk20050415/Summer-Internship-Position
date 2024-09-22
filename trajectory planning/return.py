#!/usr/bin/env python

# edit the width of gripper

import time
import argparse
import numpy as np

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on



def main():
    '''argparser = argparse.ArgumentParser()
    argparser.add_argument("robot_ip", help="IP address of the robot server")
    argparser.add_argument("local_ip", help="IP address of this PC")
    args = argparser.parse_args()'''
    
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.104"

    log = flexivrdk.Log()
    mode = flexivrdk.Mode


    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot
        log.info("Enabling robot ...")
        robot.enable()

        while not robot.isOperational():
            time.sleep(1)

        log.info("Robot is now operational")

        # primitive execution mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)


        log.info("MoveL:placehigh")
        robot.executePrimitive(
            "MoveL(target=0.471448 -0.369169 0.536714 177.678 0.163 157.855 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)


        robot.getRobotStates(robot_states)
        print("return",robot_states.q)
      
        log.info("Opening gripper")
        gripper.move(0.015, 0.1, 20)
        time.sleep(1)

        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()

