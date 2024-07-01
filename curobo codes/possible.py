#!/usr/bin/env python

# edit the width of gripper

import time
import argparse

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on



def main():
    
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

        log.info("MoveL:flat")
        robot.executePrimitive(
            "MoveL(target=0.480282 -0.100428 0.497977 0.647 -179.582 2.637 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("flat",robot_states.q)
        print("flat",robot_states.tcpPoseDes)

        log.info("MoveL:side")
        robot.executePrimitive(
            "MoveL(target=0.480286 -0.100414 0.497982 0.646 -179.585 86.855 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("side",robot_states.q)
        print("side",robot_states.tcpPoseDes)

        log.info("MoveL:inclined")
        robot.executePrimitive(
            "MoveL(target=0.480277 -0.100404 0.497970 0.645 -179.585 116.651 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("inclined",robot_states.q)
        print("inclined",robot_states.tcpPoseDes)

        log.info("MoveL:inclined origin")
        robot.executePrimitive(
            "MoveL(target=0.484466 -0.099501 0.497970 0.645 -179.585 47.761 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
            
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("inclined_origin",robot_states.q)
        print("inclined_origin_current",robot_states.tcpPose)
        print("inclined_origin",robot_states.tcpPoseDes)

        print("here")

        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()

