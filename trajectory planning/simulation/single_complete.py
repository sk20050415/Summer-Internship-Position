#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="flexiv_grav_spheretest.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)
args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.tasks import Stacking as BaseStacking
from omni.isaac.core.materials import PhysicsMaterial

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def main():
    # create a curobo motion gen instance:

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage

    # Make a target to follow
    target = cuboid.DynamicCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.01]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.02,
        mass=0.2,
    )
    material = PhysicsMaterial(
        "/World/physics_material/aluminum", 
        dynamic_friction=1.9,
        static_friction=1.1,
        restitution=0.1, 
    )
    target.apply_physics_material(material)
    '''des = cuboid.VisualCuboid(
        "/World/des",
        position=np.array([0.6, 0.3, 0.25]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.01,
    )'''

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None
    des_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = robot.get_articulation_controller()

    gripper = ParallelGripper(
        end_effector_prim_path="/World/rizon4/base_hand",
        joint_prim_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
        joint_opened_positions=np.array([0.03, 0.03]),
        joint_closed_positions=np.array([0, 0]),
        action_deltas=np.array([0.03, 0.03]),
    )
    my_denso = my_world.scene.add(
        SingleManipulator(
            prim_path="/World/rizon4",
            name="rizon4",
            end_effector_prim_name="base_hand",
            gripper=gripper,
        )
    )
    joints_default_positions = np.zeros(9)
    joints_default_positions[7] = 0.01
    joints_default_positions[8] = 0.01
    my_denso.set_joints_default_state(positions=joints_default_positions)

    cmd_js_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

    

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    cmd_plan1 = None
    cmd_idx1 = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    k = 0
    j = 0
    spheres = None
    past_cmd = None
    wait_steps = 8
    flag = True
    actflag = False
    gripper_flag = False
    reach_flag = False
    attach_obj = None
    cube_position = None
    round = 1


    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index

        if step_index < 2:
            my_world.reset()
            # robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        if cube_position is None:
            cube_position, cube_orientation = target.get_world_pose()
        
        obj_position, obj_orientation = target.get_world_pose()
        place_position=np.array([0.6, 0.3, 0.01])
        place_orientation=np.array([0, 1, 0, 0])
        #print("cube_position", cube_position)
        #print("target_pose", target_pose)
        #print("cube_orientation", cube_orientation)
        # place_position, place_orientation = des.get_world_pose()
        # print("place_position_def",type(place_position))

        if des_pose is None:
            des_pose = np.array([0.5, 0.5, 0.5])
        if target_pose is None:
            target_pose = np.array([0.5, 0.5, 0.5])

        # joint_increment = 0.0005
        gripper_positions = my_denso.gripper.get_joint_positions()
        

        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        # print("sim_js_names",sim_js_names)
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities)  * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        robot.enable_gravity()

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 0.2) :
            robot_static = True
        if (
            np.linalg.norm(cube_position - target_pose) > 1e-2
            and robot_static
        ):  # 开始规划
            ee_translation_goal = cube_position+ [0, 0, 0.05]
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )

            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

            succ = result.success.item()  # ik_result.success.item()
            if succ:
                '''cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                
                        # print("get_dof_index",robot.get_dof_index(x)) # 给的序列数字
                        # print("idx_list",idx_list) # 序列数字，从0开始增加
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                # print(common_js_names)'''
                cmd_plan = result.get_interpolated_plan()
                idx_list = [i for i in range(len(cmd_js_names))]
                cmd_plan = cmd_plan.get_ordered_joint_state(cmd_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
            target_pose = cube_position
        

        
        if cmd_plan is not None:    # 有规划则执行规划，夹爪打开
            cmd_state = cmd_plan[cmd_idx]
            
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            
            my_denso.gripper.apply_action(
                ArticulationAction([0.3, 0.3])
            )
            # print("open", gripper_positions)

            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
            flag = False
        else:
            flag = True    # 无规划，flag切为true，可以进行移动操作,夹爪关闭
            if round == 1:
                my_denso.gripper.apply_action(
                    ArticulationAction([0, 0])
                )
            for _ in range(wait_steps):
                my_world.step(render=True)
        gripper_positions = my_denso.gripper.get_joint_positions()
        gripper_velocity = my_denso.gripper.get_angular_velocity()
        # if (np.abs(gripper_positions[0]) < 1e-3) and (np.abs(gripper_positions[1]) < 1e-3):
        if (np.max(np.abs(gripper_velocity)) < 0.05) and (np.max(np.abs(gripper_positions)) < 0.1):
            gripper_flag = True
        # print ("cmd_plan", flag)    
        #print ("gripper_flag", gripper_flag)
        #print ("gripper_velocity", gripper_velocity)
        #print ("gripper_positions", gripper_positions)
        
        # print (sim_js.velocities)
        # 放置物体
            
      

        if (flag and gripper_flag):    # 无规划，可以进行移动操作      
            for _ in range(2):
                my_world.step(render=False)

            sim_js1 = robot.get_joints_state()
            sim_js_names1 = robot.dof_names
            if np.any(np.isnan(sim_js1.positions)):
                log_error("isaac sim has returned NAN joint position values.")
            cu_js1 = JointState(
                position=tensor_args.to_device(sim_js1.positions),
                velocity=tensor_args.to_device(sim_js1.velocities)  * 0.0,
                acceleration=tensor_args.to_device(sim_js1.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js1.velocities) * 0.0,
                joint_names=sim_js_names1,
            )
            
            cu_js1 = cu_js1.get_ordered_joint_state(motion_gen.kinematics.joint_names)
            robot_static1 = False
            if (np.max(np.abs(sim_js1.velocities)) < 0.4) :
                robot_static1 = True
            #print("sim_js1.velocities",sim_js1.velocities)
            print(robot_static1)
            
            if (
                np.linalg.norm(place_position - des_pose) > 1e-2
                and robot_static1
            ):  
                ee_translation_goal1 = place_position+ [0, 0, 0.05]
                ee_orientation_teleop_goal1 = place_orientation

                ik_goal1 = Pose(
                    position=tensor_args.to_device(ee_translation_goal1),
                    quaternion=tensor_args.to_device(ee_orientation_teleop_goal1),
                )
                result1 = motion_gen.plan_single(cu_js1.unsqueeze(0), ik_goal1, plan_config)
                # print(cu_js1.unsqueeze(0),"ik", ik_goal1.position)
                successful = result1.success.item() 
                if successful:
                    cmd_plan1 = result1.get_interpolated_plan()
                    cmd_plan1 = motion_gen.get_full_js(cmd_plan1)
                    idx_list1 = []
                    com_js_names = []
                    '''for x in sim_js_names1:
                        if x in cmd_plan1.joint_names:
                            idx_list1.append(robot.get_dof_index(x))
                            com_js_names.append(x)
                    print("plan")
                    cmd_plan1 = cmd_plan1.get_ordered_joint_state(com_js_names)'''
                    print("plan")
                    cmd_plan1 = result1.get_interpolated_plan()
                    idx_list1 = [i for i in range(len(cmd_js_names))]
                    cmd_plan1 = cmd_plan1.get_ordered_joint_state(cmd_js_names)
                    
                    
                    cmd_idx1 = 0
                else:
                    carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
                des_pose = place_position
    

            if cmd_plan1 is not None:    # 执行放置规划
                actflag = True
                print("actflag", actflag) #在执行
                cmd_state1 = cmd_plan1[cmd_idx1]
            
                art_action1 = ArticulationAction(
                    cmd_state1.position.cpu().numpy(),
                    cmd_state1.velocity.cpu().numpy(),
                    joint_indices=idx_list1,
                )
                articulation_controller.apply_action(art_action1)

                my_denso.gripper.apply_action(
                    ArticulationAction([0, 0])
                )
                
                round = 2
                cmd_idx1 += 1
                for _ in range(2):
                    my_world.step(render=False)
                if cmd_idx1 >= len(cmd_plan1.position):
                    cmd_idx1 = 0
                    cmd_plan1 = None
            else:
                if (np.linalg.norm(obj_position - place_position) < 1e-2):
                    reach_flag = True
                    j += 1
                print("reach_flag", reach_flag)
                print("j", j)
                if reach_flag:
                    my_denso.gripper.apply_action(
                        ArticulationAction([0.3, 0.3])
                    )
                    
                      
            print ("obj_position", obj_position)
            print ("place_position", place_position)

    simulation_app.close()

if __name__ == "__main__":
    main()
