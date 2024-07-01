# Third Party
import torch
import numpy as np
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

import time
import math
import argparse

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on
from utility import parse_pt_states


def plot_traj(trajectory, dt):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(4, 1)
    q = trajectory.position.cpu().numpy()
    qd = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = trajectory.jerk.cpu().numpy()
    timesteps = [i * dt for i in range(q.shape[0])]
    for i in range(q.shape[-1]):
        axs[0].plot(timesteps, q[:, i], label=str(i))
        axs[1].plot(timesteps, qd[:, i], label=str(i))
        axs[2].plot(timesteps, qdd[:, i], label=str(i))
        axs[3].plot(timesteps, qddd[:, i], label=str(i))

    plt.legend()
    # plt.savefig("test.png")
    plt.show()


def demo_motion_gen(begin_q=None, target_position=None):
    # Standard Library
    PLOT = False
    js = False
    tensor_args = TensorDeviceType()
    world_file = "virtual_test.yml"
    robot_file = "flexiv_plus_de.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
    )

    motion_gen = MotionGen(motion_gen_config)

    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )
    #if begin_q is None:
    #    begin_q = [0.0, -0.2, 0.0, 0.57, 0.0, 0.2, 0.0]
    begin_position = np.array(begin_q)
    begin_cfg = tensor_args.to_device(begin_position)
    begin_state = JointState.from_position(begin_cfg.view(1, -1))

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1))
    goal_state = start_state.clone()
    goal_state.position[..., 3] -= 0.1

    first_position = target_position
    first_orientation=np.array([0.0011660821037366986, -0.40483272075653076, 0.9143546223640442, -0.008044295012950897])
    #first_position=np.array([0.4952899217605591, -0.3900209963321686, 0.5247458219528198])
    #first_orientation=np.array([0.005176813807338476, 0.17775195837020874, 0.9838780760765076, 0.01900942623615265])
    #ee_translation_goal = first_position+ [0, 0, 0.06]
    ee_translation_goal = first_position
    ee_orientation_teleop_goal = first_orientation
    # compute curobo solution:
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
        )
    print(tensor_args.to_device(ee_translation_goal))
    

    if js:
        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            MotionGenPlanConfig(
                max_attempts=1, enable_graph=False, enable_opt=True, enable_finetune_trajopt=True
            ),
        )
    else:
        result = motion_gen.plan_single(
            begin_state, ik_goal, MotionGenPlanConfig(max_attempts=1, enable_graph=False, enable_opt=True, enable_finetune_trajopt=True)
        )
        # print("me")
    traj = result.get_interpolated_plan()
    print("Trajectory Generated: ", result.success, result.solve_time, result.status)
    print("dt",result.interpolation_dt)
    # print("traj",traj)
    if PLOT and result.success.item():
        plot_traj(traj, result.interpolation_dt)
    return traj

def main():
    setup_curobo_logger("error")

    frequency = 100
    # frequency >= 1 and frequency <= 100
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.104"
    robot_states = flexivrdk.RobotStates()
    log = flexivrdk.Log()
    mode = flexivrdk.Mode
    

    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        gripper = flexivrdk.Gripper(robot)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        while not robot.isOperational():
            time.sleep(1)
        log.info("Robot is now operational")

        # Wait for the primitive to finish
        while robot.isBusy():
            time.sleep(1)

        
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:safe_origin")
        robot.executePrimitive(
            "MoveL(target=0.495552 -0.397079 0.526107 177.678 0.163 157.855 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # Switch to non-real-time joint position control mode
        robot.setMode(mode.NRT_JOINT_POSITION)

        period = 1.0 / frequency
        loop_time = 0
        place_time = 0
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

        DOF = len(robot_states.q)
        print("robot_states.q: ", robot_states.q)

        # Initialize target vectors
        target_pos = init_pos.copy()
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF

        # Joint motion constraints
        MAX_VEL = [0.8] * DOF
        MAX_ACC = [1.0] * DOF

        gripper.move(0.02, 0.003, 30)
        #######################################HERE############################################
        q = init_pos.copy()
        print(q)
        '''for i in range(len(q)):
            q[i] = q[i]*np.pi/180
        print(q)'''
        target_place=np.array([0.484466, -0.099501, 0.498583])
        traj = demo_motion_gen(begin_q = q, target_position=target_place)
        art_pos = traj.position.cpu().numpy()
        art_vel = traj.velocity.cpu().numpy()
        loop_counter = len(art_pos)
        count = 0
        print("pos",len(art_pos))
        
        # Send command periodically at user-specified frequency loop_time <= loop_counter * period
        while (loop_time <= loop_counter * period):
            time.sleep(period)

            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")
          
            for i in range(DOF):
                target_pos[i] = art_pos[count][i]

            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )

            count += 1
            loop_time += period
            # print("current", count)


        time.sleep(1)
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:catch")
        robot.executePrimitive(
            "MoveL(target=0.484466 -0.099501 0.460583 0.897 -179.749 47.761 WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.0105, 0.003, 40)
        time.sleep(3)
        
        log.info("MoveL:up")
        robot.executePrimitive(
            "MoveL(target=0.484466 -0.099501 0.516107 0.897 -179.749 47.761 WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("MoveL:move tube")
        robot.executePrimitive(
            "MoveL(target=0.495552 -0.397079 0.516107 177.678 0.163 157.855 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("placehigh",robot_states.tcpPoseDes)

        log.info("MoveL:place tube")
        robot.executePrimitive(
            "MoveL(target=0.495552 -0.397079 0.466107 177.678 0.163 157.855 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
      
        log.info("Opening gripper")
        gripper.move(0.015, 0.1, 20)
        time.sleep(2)

        '''robot.setMode(mode.NRT_JOINT_POSITION)
        # second move
        robot.getRobotStates(robot_states)
        second_pos = robot_states.q.copy()
        print("second_pos",second_pos)
        place_pos = np.array([0.495552, -0.397079, 0.526107])
        place_traj = demo_motion_gen(begin_q = second_pos, target_position = place_pos)
        place_art_pos = place_traj.position.cpu().numpy()
        place_loop_counter = len(place_art_pos)
        place_count = 0
        print("place_pos",len(place_art_pos))
        
        while (place_time <= place_loop_counter * period):
            time.sleep(period)

            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")
          
            for i in range(DOF):
                target_pos[i] = place_art_pos[place_count][i]

            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )

            place_count += 1
            place_time += period
            # print("current", count)

        robot.getRobotStates(robot_states)
        print("place_pos",robot_states.q)
        time.sleep(1)
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:place")
        robot.executePrimitive(
            "MoveL(target=0.495552 -0.397079 0.466107 0.897 -179.749 47.761 WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.015, 0.003, 30)
        time.sleep(2)'''


    except Exception as e:
        log.error(str(e))

if __name__ == "__main__":
    main()