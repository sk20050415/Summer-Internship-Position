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

    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    pre_catch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]
    catch = [0.507623, -0.097015, 0.489650, 0.377, 166.092, 121.051]
    up = [0.497438, -0.081409, 0.536714, -0.896, -179.517, 47.301]
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    place_tube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846]
    
    posx = []
    posy = []
    posz = []
    desx = []
    desy = []
    desz = []

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
        
        # Switch to non-real-time joint position control mode
        robot.setMode(mode.NRT_JOINT_POSITION)
        period = 1.0 / frequency
        loop_time = 0
        place_time = 0
        up_time = 0
        move_time = 0
        down_time = 0

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
        q = [-10.99, -10.53, -21.30, 71.14, 6.07, -8.95, -8.99]
        for i in range(len(q)):
            q[i] = q[i]*np.pi/180
        print(q)
        target_place=np.array([0.507623, -0.097015, 0.529650])
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

            robot.getRobotStates(robot_states)
            tcp_pose = robot_states.tcpPose
            tcp_des = robot_states.tcpPoseDes
            posx.append(tcp_pose[0])
            posy.append(tcp_pose[1])
            posz.append(tcp_pose[2])
            desx.append(tcp_des[0])
            desy.append(tcp_des[1])
            desz.append(tcp_des[2])
            count += 1
            loop_time += period

        time.sleep(1)


        # second move,down
        q2 = [-0.0533773, 0.008281, 0.10857, 1.46255934, 0.005417, -0.1289172, -0.77157831]
        robot.getRobotStates(robot_states)
        #second_pos = robot_states.q.copy()
        second_pos = q2
        print("second_pos",second_pos)
        sec_place=np.array([0.507623, -0.097015, 0.489650])
        sec_orient=np.array([-0.0624153316, 0.863970518, -0.4887598455, -0.10379336])
        place_traj = demo_motion_gen(begin_q = second_pos, target_position=sec_place, target_orientation=sec_orient)
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
            robot.getRobotStates(robot_states)
            tcp_pose = robot_states.tcpPose
            tcp_des = robot_states.tcpPoseDes
            posx.append(tcp_pose[0])
            posy.append(tcp_pose[1])
            posz.append(tcp_pose[2])
            desx.append(tcp_des[0])
            desy.append(tcp_des[1])
            desz.append(tcp_des[2])
            place_count += 1
            place_time += period
            # print("current", count)
        time.sleep(1)


        # third move,up
        q3 = [-0.107763975859, -0.1128310412, 0.01402801834, 1.39052093029, 0.2001054137945, -0.199992984533, -2.1799564361572]
        third_pos = q3
        thi_place=np.array([0.507623, -0.097015, 0.536714])
        thi_orient=np.array([-0.00699743675068, -0.4011105298996, 0.91598659754, 0.005471404176])
        up_traj = demo_motion_gen(begin_q = third_pos, target_position=thi_place, target_orientation=thi_orient)
        up_art_pos = up_traj.position.cpu().numpy()
        up_loop_counter = len(up_art_pos)
        up_count = 0
        print("up_pos",len(up_art_pos))
        
        while (up_time <= up_loop_counter * period):
            time.sleep(period)

            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")
          
            for i in range(DOF):
                target_pos[i] = up_art_pos[up_count][i]

            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )
            robot.getRobotStates(robot_states)
            tcp_pose = robot_states.tcpPose
            tcp_des = robot_states.tcpPoseDes
            posx.append(tcp_pose[0])
            posy.append(tcp_pose[1])
            posz.append(tcp_pose[2])
            desx.append(tcp_des[0])
            desy.append(tcp_des[1])
            desz.append(tcp_des[2])
            up_count += 1
            up_time += period
        time.sleep(1)
      
      
        # fourth move,move
        q4 = [-0.055025793612, -0.01993733, 0.0812698975, 1.4333751202, 0.002351583214, -0.13014896214, -0.8009197711944]
        fou_pos = q4
        fou_place=np.array([0.471448, -0.369169, 0.536714])
        fou_orient=np.array([0.0053847534582, 0.192058250308, 0.981174767, 0.019512292])
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
            robot.getRobotStates(robot_states)
            tcp_pose = robot_states.tcpPose
            tcp_des = robot_states.tcpPoseDes
            posx.append(tcp_pose[0])
            posy.append(tcp_pose[1])
            posz.append(tcp_pose[2])
            desx.append(tcp_des[0])
            desy.append(tcp_des[1])
            desz.append(tcp_des[2])
            move_count += 1
            move_time += period
        time.sleep(1)

        # fifth move,down
        q5 = [-0.2265537, -0.1866695732, -0.330121845, 1.23663974, 0.09970323, -0.15543203, -0.151190415]
        fif_pos = q5
        fif_place=np.array([0.471448, -0.369169, 0.466266])
        fif_orient=np.array([0.005381352, 0.192069516, 0.981172442436, 0.01951902732])
        down_traj = demo_motion_gen(begin_q = fif_pos, target_position=fif_place, target_orientation=fif_orient)
        down_art_pos = down_traj.position.cpu().numpy()
        down_loop_counter = len(down_art_pos)
        down_count = 0
        print("down_pos",len(down_art_pos))
        
        while (down_time <= down_loop_counter * period):
            time.sleep(period)

            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")
          
            for i in range(DOF):
                target_pos[i] = down_art_pos[down_count][i]
            # Send command
            robot.sendJointPosition(
                target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
            )
            robot.getRobotStates(robot_states)
            tcp_pose = robot_states.tcpPose
            tcp_des = robot_states.tcpPoseDes
            posx.append(tcp_pose[0])
            posy.append(tcp_pose[1])
            posz.append(tcp_pose[2])
            desx.append(tcp_des[0])
            desy.append(tcp_des[1])
            desz.append(tcp_des[2])
            down_count += 1
            down_time += period
        time.sleep(1)

        log.info("Opening gripper")
        gripper.move(0.015, 0.1, 20)
        time.sleep(1.5)

        filename = 'trajectory_total2.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow(['X', 'Y', 'Z','desx','desy','desz'])   
            for x, y, z, xd, yd, zd in zip(posx, posy, posz, desx, desy, desz):
                writer.writerow([x, y, z, xd, yd, zd])
        print(f'Data saved to {filename}')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(posx, posy, posz, color='black', linestyle='-', linewidth=0.5, label='Real Position')
        ax.plot(desx, desy, desz, color='blue', linestyle='--', linewidth=0.5, label='Real Position')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.savefig('trajectory_total2.png')
        plt.show()

        robot.stop()

    except Exception as e:
        log.error(str(e))

if __name__ == "__main__":
    main()
