#!/usr/bin/env python

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
    
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.104"

    log = flexivrdk.Log()
    mode = flexivrdk.Mode
    #IMAGE_PATH = "/home/robot/motion/IMG_2145.JPG"
    #robot_actions.act_leaning(robot, log, gripper, mode, robot_states)

    
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

        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)

        log.info("Executing primitive: MoveJ")
        # degree low position
        q = [-10.99, -10.53, -21.30, 71.14, 6.07, -8.95, -8.99]
        #q = [-12, -16, -22, 68, 8, -6, -11] -0.19180428981781006, -0.1837417185306549, -0.3717230558395386, 1.2416166067123413, 0.10589762777090073, -0.15613622963428497, -0.15685483813285828
        #print(q[0],q[1],q[2],q[3],q[4],q[5],q[6])

        #robot.executePrimitive("MoveJ(target=-12.473 -16.128 -22.768 68.690 8.407 -6.372 -11.489, relative=false)") 
        robot.executePrimitive("MoveJ(target=-10.99 -10.53 -21.30 71.14 6.07 -8.95 -8.99, relative=false)") 
        #robot.executePrimitive("MoveJ(target=q[0] q[1] q[2] q[3] q[4] q[5] q[6], relative=false)") 
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        robot.getRobotStates(robot_states)
        #print("return",robot_states.q)
      
        log.info("Opening gripper")
        gripper.move(0.015, 0.1, 20)
        time.sleep(1)

        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()







