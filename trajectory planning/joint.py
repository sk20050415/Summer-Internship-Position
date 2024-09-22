#!/usr/bin/env python

"""intermediate1_non_realtime_joint_position_control.py

This tutorial runs non-real-time joint position control to hold or sine-sweep all robot joints.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import math
import argparse

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on


def main():
    frequency = 20
    # frequency >= 1 and frequency <= 100
    # Define alias
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.104"
    robot_states = flexivrdk.RobotStates()
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        while not robot.isOperational():
            time.sleep(1)
        log.info("Robot is now operational")

        # Wait for the primitive to finish
        while robot.isBusy():
            time.sleep(1)

        # Switch to non-real-time joint position control mode
        robot.setMode(mode.NRT_JOINT_POSITION)

        period = 1.0 / frequency
        loop_time = 0
        print(
            "Sending command to robot at",
            frequency,
            "Hz, or",
            period,
            "seconds interval",
        )

        # Use current robot joint positions as initial positions
        robot.getRobotStates(robot_states)
        init_pos = robot_states.q.copy()
        print("Initial positions set to: ", init_pos)

        # Robot degrees of freedom
        DOF = len(robot_states.q)
        print("robot_states.q: ", robot_states.q)

        # Initialize target vectors
        target_pos = init_pos.copy()
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF

        # Joint motion constraints
        MAX_VEL = [0.8] * DOF
        MAX_ACC = [1.0] * DOF

        # Joint sine-sweep amplitude [rad]
        SWING_AMP = 0.1

        # TCP sine-sweep frequency [Hz]
        SWING_FREQ = 0.3

        # Send command periodically at user-specified frequency
        while (loop_time < 10):
            # Use sleep to control loop period
            time.sleep(period)

            # Monitor fault on robot server
            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")

            
            for i in range(DOF):
                target_pos[i] = init_pos[i] + SWING_AMP * math.sin(
                    2 * math.pi * SWING_FREQ * loop_time
                )
            # Otherwise all joints will hold at initial positions

            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )

            # Increment loop time
            loop_time += period

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
