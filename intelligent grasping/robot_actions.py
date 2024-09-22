import time
from utility import list2str, parse_pt_states

import torch
import numpy as np
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

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

def act_lie(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_catch = [0.507490, -0.091361, 0.506778, -0.886, -179.512, 128.808]
    catch = [0.507490, -0.091361, 0.484320, -0.886, -179.512, 128.808]
    up = [0.512479, -0.091361, 0.536778, -0.886, -179.512, 128.808]
    move_tube = [0.470688, -0.365915, 0.536778, 4.037, -151.671, 128.565]
    place_tube = [0.470688, -0.365915, 0.451999, 4.037, -151.671, 128.565] 

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("MoveL:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Closing gripper")
    gripper.move(0.0087, 0.003, 40)
    time.sleep(3)

    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:move tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_lean(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_catch = [0.501208, -0.131540, 0.532714, 179.718, -14.636, 137.086]
    catch = [0.501208, -0.131540, 0.489686, 179.718, -14.636, 137.086]
    up = [0.501208, -0.131540, 0.536714, 179.718, -14.636, 137.086]
    move_tube = [0.471448, -0.414169, 0.536714, 177.687, 0.176, 157.846]
    place_tube = [0.471448, -0.414169, 0.466266, 177.687, 0.176, 157.846]
    move_up = [0.471448, -0.414169, 0.536714, 177.687, 0.176, 157.846]

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("MoveL:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Closing gripper")
    gripper.move(0.0087, 0.003, 40)
    time.sleep(3)

    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:move tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_twolie(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_push_high = [0.525444, -0.109389, 0.511701, -0.893, -179.518, 128.805]
    pre_push = [0.525444, -0.109389, 0.484701, -0.893, -179.518, 128.805]
    push = [0.513455, -0.097310, 0.482719, -0.891, -179.516, 128.814]
    push_up = [0.513455, -0.097310, 0.501739, -0.891, -179.516, 128.814]
    pre_catch = [0.504466, -0.108769, 0.501128, -0.889, -179.515, 128.814]
    catch = [0.504466, -0.108769, 0.483146, -0.889, -179.515, 128.814]
    up = [0.504466, -0.108769, 0.516128, -0.889, -179.515, 128.814]
    move_tube = [0.470688, -0.428915, 0.536778, 4.037, -151.671, 128.565]
    place_tube = [0.470688, -0.428915, 0.451999, 4.037, -151.671, 128.565]
    move_up = [0.471448, -0.414169, 0.536714, 177.687, 0.176, 157.846]

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre push high")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_push_high)} WORLD WORLD_ORIGIN, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre push")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_push)} WORLD WORLD_ORIGIN, maxVel=0.04)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:push")
    robot.executePrimitive(
        f"MoveL(target={list2str(push)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:push up")
    robot.executePrimitive(
        f"MoveL(target={list2str(push_up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Closing gripper")
    gripper.move(0.0087, 0.003, 40)
    time.sleep(3)

    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:move tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_twocross(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_push_high = [0.507459, -0.120326, 0.511731, -0.899, -179.521, 53.808]
    pre_push = [0.507459, -0.120326, 0.486731, -0.899, -179.521, 53.808]
    push = [0.493457, -0.113331, 0.486731, -0.899, -179.521, 53.808]
    push_up = [0.493457, -0.113331, 0.511731, -0.899, -179.521, 53.808]
    pre_catch = [0.525454, -0.128966, 0.511731, 178.490, 0.681, 114.280]
    catch = [0.525454, -0.128966, 0.482679, 178.490, 0.681, 114.280]
    up = [0.525454, -0.128966, 0.536701, 178.490, 0.681, 114.280]
    move_tube = [0.487450, -0.349270, 0.536701, 179.964, 45.798, 121.597]
    place_tube = [0.487450, -0.349270, 0.442551, 179.964, 45.798, 121.597]
    move_up = [0.471448, -0.414169, 0.536701, 177.687, 0.176, 157.846]

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre push high")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_push_high)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:pre push")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_push)} WORLD WORLD_ORIGIN, maxVel=0.03)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:push")
    robot.executePrimitive(
        f"MoveL(target={list2str(push)} WORLD WORLD_ORIGIN, maxVel=0.02)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    time.sleep(2)
    log.info("Move:push up")
    robot.executePrimitive(
        f"MoveL(target={list2str(push_up)} WORLD WORLD_ORIGIN, maxVel=0.03)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:pre catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.02)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Closing gripper")
    gripper.move(0.0087, 0.003, 40)
    time.sleep(3)

    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.02)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:move tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1.2)

    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_twoapart(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_catch = [0.507490, -0.091361, 0.506778, -0.886, -179.512, 128.808]
    catch = [0.507490, -0.091361, 0.484320, -0.886, -179.512, 128.808]
    up = [0.512479, -0.091361, 0.536778, -0.886, -179.512, 128.808]
    move_tube = [0.470688, -0.365915, 0.536778, 4.037, -151.671, 128.565]
    place_tube = [0.470688, -0.365915, 0.451999, 4.037, -151.671, 128.565] 

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:pre catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("MoveL:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Closing gripper")
    gripper.move(0.0087, 0.003, 40)
    time.sleep(3)

    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:move tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_normal(robot, log, gripper):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    safe_origin = [0.471448, -0.369169, 0.536714, -2.314, 179.830, 47.362]
    pre_catch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.304]
    catch = [0.487420, -0.071394, 0.466700, -0.896, -179.517, 47.304]
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.304]
    move_tube = [0.471449, -0.369172, 0.536714, -2.310, 179.831, 47.351]
    place_tube = [0.471449, -0.369172, 0.460266, -2.310, 179.831, 47.351]
    move_up = [0.471449, -0.369172, 0.536714, -2.310, 179.831, 47.351]

    num = 0
    total = 6
    increment = 0.009

    log.info("Move:begin origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)         
        
    while num < total:
        log.info("Move:pre catch")
        robot.executePrimitive(
            f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )       
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("Move:catch")
        robot.executePrimitive(
            f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        
        log.info("Closing gripper")
        gripper.move(0.0083, 0.003, 45)
        time.sleep(2)
            
        log.info("Move:up")
        robot.executePrimitive(
            f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.08)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("Move:move tube")
        robot.executePrimitive(
            f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("Move:place tube")
        robot.executePrimitive(
            f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.08)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
      
        log.info("Opening gripper")
        gripper.move(0.0121, 0.1, 20)
        time.sleep(1)

        log.info("Move:move up")
        robot.executePrimitive(
            f"MoveL(target={list2str(move_up)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        num += 1
        pre_catch[1] -= increment
        catch[1] -= increment
        up[1] -= increment
        move_tube[1] -= increment
        place_tube[1] -= increment
        move_up[1] -= increment
        print("num", num)

    log.info("Opening gripper")
    gripper.move(0.0125, 0.1, 20)
    time.sleep(2)

def act_lying(robot, log, gripper, mode, robot_states):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_catch = [0.507490, -0.091361, 0.506778, -0.886, -179.512, 128.808]
    catch = [0.507490, -0.091361, 0.484320, -0.886, -179.512, 128.808]
    up = [0.512479, -0.091361, 0.536778, -0.886, -179.512, 128.808]
    move_tube = [0.470688, -0.365915, 0.536778, 4.037, -151.671, 128.565]
    place_tube = [0.470688, -0.365915, 0.451999, 4.037, -151.671, 128.565] 
    period = 0.01
    loop_time = 0
    place_time = 0
    move_time = 0

    log.info("Move:begin_origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
        
    gripper.move(0.015, 0.003, 30)

    log.info("Move:pre_catch")
    robot.setMode(mode.NRT_JOINT_POSITION)
        
    robot.getRobotStates(robot_states)
    init_pos = robot_states.q.copy()
    DOF = len(robot_states.q)
    target_pos = init_pos.copy()
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    MAX_VEL = [0.8] * DOF
    MAX_ACC = [1.0] * DOF

    gripper.move(0.015, 0.003, 30)
    q = [-0.1576193, 0.12116955, -0.00284479, 1.568902, 0.03851496, -0.1218236, -0.152549535]
    target_place=np.array([0.510623, -0.097015, 0.536714])
    target_orient=np.array([-0.06241533160209656, 0.8639705181121826, -0.488759845495224, -0.10379336029291153])
    traj = demo_motion_gen(begin_q = q, target_position=target_place, target_orientation=target_orient)
    art_pos = traj.position.cpu().numpy()
    loop_counter = len(art_pos)
    count = 0

    while (loop_time <= loop_counter * period):
        time.sleep(period)

        if robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")
          
        for i in range(DOF):
            target_pos[i] = art_pos[count][i]

        robot.sendJointPosition(
            target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
        )

        count += 1
        loop_time += period

    time.sleep(3.5)
        
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    log.info("Move:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.06)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
            
    gripper.move(0.0085, 0.07, 50)
    time.sleep(2)
    robot.getRobotStates(robot_states)
    q2 = robot_states.q
        
    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    robot.getRobotStates(robot_states)

    robot.setMode(mode.NRT_JOINT_POSITION)
    robot.getRobotStates(robot_states)
    init_pos = robot_states.q.copy()

    DOF = len(robot_states.q)
    target_pos = init_pos.copy()
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF

    MAX_VEL = [0.8] * DOF
    MAX_ACC = [1.0] * DOF
    q4 = [-0.04931390285, -0.0213746093, 0.0988878086, 1.440139294, -0.019082969, -0.112457253, -2.2027254]
    fou_pos = q4
    fou_place=np.array([0.470688, -0.365915, 0.536778])
    fou_orient=np.array([0.07535227, 0.8767513633, -0.4127150476, 0.235149771])
    move_traj = demo_motion_gen(begin_q = fou_pos, target_position=fou_place, target_orientation=fou_orient)
    move_art_pos = move_traj.position.cpu().numpy()
    move_loop_counter = len(move_art_pos)
    move_count = 0
    log.info("Move:move tube")
    #print("move_loop_counter", move_loop_counter)
        
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
        #print("move_time", move_time)
    time.sleep(3.5)

    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.09)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    robot.getRobotStates(robot_states)
      
    log.info("Opening gripper")
    gripper.move(0.015, 0.1, 20)
    time.sleep(1.5)

def act_leaning(robot, log, gripper, mode, robot_states):
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]
    pre_catch = [0.507490, -0.091361, 0.506778, -0.886, -179.512, 128.808]
    catch = [0.504629, -0.095521, 0.494666, 0.902, 168.152, 121.188]
    up = [0.497438, -0.081409, 0.536714, -0.896, -179.517, 47.301]
    move_tube = [0.471449, -0.369172, 0.536714, -2.310, 179.831, 47.351]
    place_tube = [0.471449, -0.369172, 0.460266, -2.310, 179.831, 47.351]
    period = 0.01
    loop_time = 0
    place_time = 0
    move_time = 0

    log.info("Move:begin_origin")
    robot.executePrimitive(
        f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
        
    gripper.move(0.015, 0.003, 30)

    log.info("Move:pre_catch")
    robot.setMode(mode.NRT_JOINT_POSITION)
        
    robot.getRobotStates(robot_states)
    init_pos = robot_states.q.copy()
    DOF = len(robot_states.q)
    target_pos = init_pos.copy()
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    MAX_VEL = [0.8] * DOF
    MAX_ACC = [1.0] * DOF

    gripper.move(0.015, 0.003, 30)
    q = [-0.1576193, 0.12116955, -0.00284479, 1.568902, 0.03851496, -0.1218236, -0.152549535]
    target_place=np.array([0.510623, -0.097015, 0.536714])
    target_orient=np.array([-0.06241533160209656, 0.8639705181121826, -0.488759845495224, -0.10379336029291153])
    traj = demo_motion_gen(begin_q = q, target_position=target_place, target_orientation=target_orient)
    art_pos = traj.position.cpu().numpy()
    loop_counter = len(art_pos)
    count = 0

    while (loop_time <= loop_counter * period):
        time.sleep(period)

        if robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")
          
        for i in range(DOF):
            target_pos[i] = art_pos[count][i]

        robot.sendJointPosition(
            target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
        )

        count += 1
        loop_time += period

    time.sleep(3.5)
        
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    log.info("Move:catch")
    robot.executePrimitive(
        f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.06)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
            
    gripper.move(0.0085, 0.07, 50)
    time.sleep(2)
    robot.getRobotStates(robot_states)
    q2 = robot_states.q
        
    log.info("Move:up")
    robot.executePrimitive(
        f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    robot.getRobotStates(robot_states)

    robot.setMode(mode.NRT_JOINT_POSITION)

    q4 = [-0.055025793612, -0.01993733, 0.0812698975, 1.4333751202, 0.002351583214, -0.13014896214, -0.8009197711944]
    fou_pos = q4
    fou_place=np.array([0.471449, -0.369172, 0.536714])
    fou_orient=np.array([-0.006743834354, -0.40150138736, 0.9156353474, 0.01905272156])
    move_traj = demo_motion_gen(begin_q = fou_pos, target_position=fou_place, target_orientation=fou_orient)
    move_art_pos = move_traj.position.cpu().numpy()
    move_loop_counter = len(move_art_pos)
    move_count = 0
    log.info("Move:move tube")
    #print("move_loop_counter", move_loop_counter)
        
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
        #print("move_time", move_time)
    time.sleep(3.5)

    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    log.info("Move:place tube")
    robot.executePrimitive(
        f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.09)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    robot.getRobotStates(robot_states)
      
    log.info("Opening gripper")
    gripper.move(0.015, 0.1, 20)
    time.sleep(1.5)