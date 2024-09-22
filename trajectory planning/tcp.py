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

SWING_AMP = 0.1

# TCP sine-sweep frequency [Hz]
SWING_FREQ = 0.3

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

        robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        # Set initial pose to current TCP pose
        robot.getRobotStates(robot_states)
        init_pose = robot_states.tcpPose.copy()
        print(
            "Initial TCP pose set to [position 3x1, rotation (quaternion) 4x1]: ",
            init_pose,
        )

        # Periodic Task
        # =========================================================================================
        # Set loop period
        period = 1.0 / frequency
        loop_counter = 0
        print(
            "Sending command to robot at",
            frequency,
            "Hz, or",
            period,
            "seconds interval",
        )

        # Send command periodically at user-specified frequency
        while (loop_counter * period) < 10.0:  # Run for 10 seconds 
            # Use sleep to control loop period
            time.sleep(period)

            # Monitor fault on robot server
            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")

            # Read robot states
            robot.getRobotStates(robot_states)

            # Initialize target pose to initial pose
            target_pose = init_pose.copy()

            target_pose[1] = init_pose[1] + SWING_AMP * math.sin(
                2 * math.pi * SWING_FREQ * loop_counter * period
            )
            robot.sendCartesianMotionForce(target_pose)

            # Increment loop counter
            loop_counter += 1

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
