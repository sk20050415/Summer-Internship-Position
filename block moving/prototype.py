# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# fmt: on

# Third Party
import torch
import pyzed.sl as sl
import numpy as np
import cv2
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from utility import parse_pt_states
import gpt

import time
import base64
from PIL import Image
import os
import io
from openai import OpenAI

os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"#若打开vpn时此处填写代理端口

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def invoke_with_image(query, image_file=None):
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    
    if image_file is not None:
        image = Image.open(image_file)
        base64_image = encode_image_to_base64(image)
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
        messages[0]["content"].append(image_message)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content

#result = invoke_with_image(query="YOUR TEXT",image_file="IMAGE_PATH")

# 将列表转换为字符串，元素之间用空格分隔
def list2str(ls):
    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str

# 绘制关节轨迹图
def plot_traj(trajectory, dt):#trajectory:pytorch张量
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(4, 1)#创建一个四行一列的多图

    #.cpu:张量从gpu->cpu   .numpy:张量-->numpy的数组
    q = trajectory.position.cpu().numpy()#获取位置
    qd = trajectory.velocity.cpu().numpy()#获取速度
    qdd = trajectory.acceleration.cpu().numpy()#获取加速度
    qddd = trajectory.jerk.cpu().numpy()#获取跃度：da/dt加加速度

    timesteps = [i * dt for i in range(q.shape[0])]#.shape[0]获取数组长度
    for i in range(q.shape[-1]):
        axs[0].plot(timesteps, q[:, i], label=str(i))
        axs[1].plot(timesteps, qd[:, i], label=str(i))
        axs[2].plot(timesteps, qdd[:, i], label=str(i))
        axs[3].plot(timesteps, qddd[:, i], label=str(i))

    plt.legend()
    # plt.savefig("test.png")
    plt.show()

# 生成运动轨迹
def demo_motion_gen(begin_q=None, target_position=None, target_orientation=None):
    # 标准库
    PLOT = False  # 是否绘制轨迹
    js = False  # 是否使用关节空间规划
    tensor_args = TensorDeviceType()  # 张量设备类型
    world_file = "virtual_test.yml"  # 世界配置文件
    robot_file = "flexiv_plus_de.yml"  # 机器人配置文件
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,  # 插值时间步长
    )

    motion_gen = MotionGen(motion_gen_config)  # 初始化运动生成器

    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js)  # 预热运动生成器
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)  # 从字典加载机器人配置
    retract_cfg = motion_gen.get_retract_config()  # 获取缩回配置
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )  # 计算缩回状态的运动学

    # 如果没有提供初始关节角，则使用缩回配置
    #begin_position = np.array(begin_q)
    #begin_cfg = tensor_args.to_device(begin_position)
    #begin_state = JointState.from_position(begin_cfg.view(1, -1))

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1))
    goal_state = start_state.clone()
    goal_state.position[..., 3] -= 0.1

    first_position = target_position
    first_orientation = target_orientation
    ee_translation_goal = first_position
    ee_orientation_teleop_goal = first_orientation
    # 计算 CuRobo 解决方案：
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
            start_state, ik_goal, MotionGenPlanConfig(max_attempts=1, enable_graph=False, enable_opt=True, enable_finetune_trajopt=True)
        )
    traj = result.get_interpolated_plan()
    print("轨迹生成: ", result.success, result.solve_time, result.status)
    print("dt", result.interpolation_dt)
    # print("traj", traj)
    if PLOT and result.success.item():
        plot_traj(traj, result.interpolation_dt)
    return traj


def main():
    frequency = 100
    # frequency >= 1 and frequency <= 100
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.106"
    robot_states = flexivrdk.RobotStates()
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    #相机初始化
    '''zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.depth_minimum_distance = 0.3

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    image = sl.Mat()
    runtime = sl.RuntimeParameters(enable_fill_mode = True)'''

    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    photo_pose = [0.436954, -0.171356, 0.516743, 179.042, 14.506, 179.717]
    pre_catch = [0.487438, -0.071409, 0.3, -0.896, -179.517, 47.301]
    '''catch = [0.487420, -0.071394, 0.491700, -0.896, -179.517, 47.304]
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]
    move_cube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    place_cube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846]'''
    
    catch = [0.487420, -0.071394, 0.09, -0.896, -179.517, 47.304]
    up = [0.487438, -0.071409, 0.3, -0.896, -179.517, 47.301]
    move_cube = [0.471448, -0.369169, 0.3, 177.687, 0.176, 157.846]
    place_cube = [0.471448, -0.369169, 0.09, 177.687, 0.176, 157.846]


    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        gripper = flexivrdk.Gripper(robot)#gripper.move(WIDTH,速度,FORCE)

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

        '''# Switch to non-real-time joint position control mode
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
        target_place=np.array([0.487438, -0.071409, 0.536714])
        target_orient=np.array([0.0011660821037366986, -0.40483272075653076, 0.9143546223640442, -0.008044295012950897])
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
            # print("current", count)'''
        

        time.sleep(1)
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("MoveL:safe origin")
        robot.executePrimitive(
            f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.0125, 0.01, 30)
        time.sleep(1)

        log.info("MoveL:photo")#移动到拍摄位置
        robot.executePrimitive(
            f"MoveL(target={list2str(photo_pose)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        time.sleep(1)

        #拍摄保存
        '''if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            img = image.get_data()
            cv2.imwrite('cube_test.jpg', img)
            zed.close()

        text = "There are two cubes in the photo. One is green, the other is red.If the green cube is the left one, return 0. If the red cube is the left one, return 1.You can only return the figure. No other words."
        file = "/home/robot/motion/cube_test.jpg"
        result = invoke_with_image(query=text,image_file=file)'''
        time.sleep(3)
        result = gpt.gpt2()
        log.info("result: "+result)
        #初始情况默认绿色在左边
        if result == '1':#若红色在左边，自动交换红色和绿色的位置
            tmp=catch
            catch=place_cube
            place_cube=tmp
            tmp=pre_catch
            pre_catch=move_cube
            move_cube=tmp

        catch[2]=0.06

        log.info("MoveL:move2pre_catch")
        robot.executePrimitive(
            f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        
        gripper.move(0.045, 0.01, 30)

        log.info("MoveL:catch")
        robot.executePrimitive(
            f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.032, 0.1, 40)
        time.sleep(2)
        
        log.info("MoveL:up")
        robot.executePrimitive(
            f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("MoveL:move cube")
        robot.executePrimitive(
            f"MoveL(target={list2str(move_cube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("placehigh",robot_states.tcpPoseDes)

        log.info("MoveL:place cube")
        robot.executePrimitive(
            f"MoveL(target={list2str(place_cube)} WORLD WORLD_ORIGIN, maxVel=0.07)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        log.info("Opening gripper")
        gripper.move(0.045, 0.1, 20)
        time.sleep(1.5)

    except Exception as e:
        log.error(str(e))

if __name__ == "__main__":
    main()