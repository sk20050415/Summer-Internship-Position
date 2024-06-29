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
import threading
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on
from utility import parse_pt_states


def list2str(ls):
    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str


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


def demo_motion_gen(begin_q=None, target_position=None, target_orientation=None):
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
    first_orientation= target_orientation
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

    photo_pose = [0.436954, -0.171356, 0.516743, 179.042, 14.506, 179.717]
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]

    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    pre_catch = [0.507623, -0.097015, 0.536714, -0.896, -179.517, 47.301]
    catch = [0.504629, -0.095521, 0.494666, 0.902, 168.152, 121.188]
    up = [0.497438, -0.081409, 0.536714, -0.896, -179.517, 47.301]
    move_tube = [0.471449, -0.369172, 0.536714, -2.310, 179.831, 47.351]
    place_tube = [0.471449, -0.369172, 0.460266, -2.310, 179.831, 47.351]
    

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

        while robot.isBusy():
            time.sleep(1)
            

        # safe origin
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.executePrimitive("MoveJ(target=-10.99 -10.53 -21.30 71.14 6.07 -8.95 -8.99, relative=false)") 
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        log.info("MoveL:photo")
        robot.executePrimitive(
            f"MoveL(target={list2str(photo_pose)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        time.sleep(7)


        log.info("MoveL:begin_origin")
        robot.executePrimitive(
            f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        
        # Switch to non-real-time joint position control mode
        robot.setMode(mode.NRT_JOINT_POSITION)
        period = 1.0 / frequency
        loop_time = 0
        place_time = 0
        move_time = 0
        print(
            "Sending command to robot at",
            frequency,
            "Hz, or",
            period,
            "seconds interval",
        )
        
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

        gripper.move(0.015, 0.003, 30)
        #######################################HERE############################################
        q = [-0.1576193, 0.12116955, -0.00284479, 1.568902, 0.03851496, -0.1218236, -0.152549535]
        '''for i in range(len(q)):
            q[i] = q[i]*np.pi/180
        print(q)'''
        target_place=np.array([0.510623, -0.097015, 0.536714])
        #target_orient=np.array([-0.0624153316, 0.863970518, -0.488759845, -0.103793360293])
        target_orient=np.array([-0.06241533160209656, 0.8639705181121826, -0.488759845495224, -0.10379336029291153])
        traj = demo_motion_gen(begin_q = q, target_position=target_place, target_orientation=target_orient)
        art_pos = traj.position.cpu().numpy()
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

        time.sleep(3.5)

        
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:catch")
        robot.executePrimitive(
            f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.06)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
            
        gripper.move(0.0087, 0.1, 40)
        time.sleep(2)
        robot.getRobotStates(robot_states)
        print("catch",robot_states.tcpPoseDes)
        q2 = robot_states.q
        print("catchq",q2)
        
        log.info("MoveL:up")
        robot.executePrimitive(
            f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        robot.setMode(mode.NRT_JOINT_POSITION)
        q4 = [-0.055025793612, -0.01993733, 0.0812698975, 1.4333751202, 0.002351583214, -0.13014896214, -0.8009197711944]
        fou_pos = q4
        fou_place=np.array([0.471449, -0.369172, 0.536714])
        fou_orient=np.array([-0.006743834354, -0.40150138736, 0.9156353474, 0.01905272156])
        move_traj = demo_motion_gen(begin_q = fou_pos, target_position=fou_place, target_orientation=fou_orient)
        move_art_pos = move_traj.position.cpu().numpy()
        move_loop_counter = len(move_art_pos)
        move_count = 0
        print("move_pos",len(move_art_pos))
        
        while (move_time <= move_loop_counter * period):
            time.sleep(period)

            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")
          
            for i in range(DOF):
                target_pos[i] = move_art_pos[move_count][i]
            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )
            move_count += 1
            move_time += period
        time.sleep(3.5)

        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:place tube")
        robot.executePrimitive(
            f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.09)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("placedes",robot_states.tcpPoseDes)

        monitoring = False
      
        log.info("Opening gripper")
        gripper.move(0.015, 0.1, 20)
        time.sleep(1.5)

        robot.stop()

    except Exception as e:
        log.error(str(e))

if __name__ == "__main__":
    main()