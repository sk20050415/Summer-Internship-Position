# 第三方库
import torch
import numpy as np
# CuRobo 库
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

# 导入Flexiv RDK Python库
# 格式关闭
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk
# 格式开启
from utility import parse_pt_states

# 初始化数据列表存储机器人TCP位置、目标位置及旋转角度
posx = []
posy = []
posz = []
desx = []
desy = []
desz = []
rx = []
ry = []
rz = []

# 将列表转换为字符串的函数
def list2str(ls):
    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str

# 绘制轨迹图的函数
def plot_traj(trajectory, dt):
    _, axs = plt.subplots(4, 1)
    q = trajectory.position.cpu().numpy() # 位置
    qd = trajectory.velocity.cpu().numpy() # 速度
    qdd = trajectory.acceleration.cpu().numpy() # 加速度
    qddd = trajectory.jerk.cpu().numpy() # 冲击力
    timesteps = [i * dt for i in range(q.shape[0])] # 时间步长
    for i in range(q.shape[-1]):
        axs[0].plot(timesteps, q[:, i], label=str(i))
        axs[1].plot(timesteps, qd[:, i], label=str(i))
        axs[2].plot(timesteps, qdd[:, i], label=str(i))
        axs[3].plot(timesteps, qddd[:, i], label=str(i))

    plt.legend()
    plt.show()

# 运动生成函数
def demo_motion_gen(begin_q=None, target_position=None, target_orientation=None):
    # 使用标准库
    PLOT = False  # 是否绘制轨迹
    js = False  # 是否使用关节空间规划
    tensor_args = TensorDeviceType()  # 设备类型配置
    world_file = "virtual_test.yml"  # 世界配置文件
    robot_file = "flexiv_plus_de.yml"  # 机器人配置文件
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
    )

    motion_gen = MotionGen(motion_gen_config)  # 初始化运动规划器

    # 预热运动规划器，准备计算
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)  # 加载机器人配置
    retract_cfg = motion_gen.get_retract_config()  # 获取缩回配置
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )  # 计算动力学状态

    # 如果没有指定起始关节位置，则默认为零位
    #begin_q = [0.0, -0.2, 0.0, 0.57, 0.0, 0.2, 0.0] if begin_q is None else begin_q
    begin_position = np.array(begin_q)
    begin_cfg = tensor_args.to_device(begin_position)  # 将起始位置转换为设备上的张量
    begin_state = JointState.from_position(begin_cfg.view(1, -1))  # 创建关节状态

    # 计算末端执行器缩回位置
    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1))  # 创建起始状态
    goal_state = start_state.clone()  # 复制起始状态作为目标状态
    goal_state.position[..., 3] -= 0.1  # 修改目标状态中的某个关节位置

    # 目标位置和方向
    first_position = target_position
    first_orientation = target_orientation
    ee_translation_goal = first_position
    ee_orientation_teleop_goal = first_orientation

    # 构建逆运动学目标
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
    )
    print(tensor_args.to_device(ee_translation_goal))  # 输出目标位置

    # 运动规划
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
            begin_state,
            ik_goal,
            MotionGenPlanConfig(max_attempts=1, enable_graph=False, enable_opt=True, enable_finetune_trajopt=True)
        )

    # 输出轨迹生成结果
    traj = result.get_interpolated_plan()
    print("Trajectory Generated: ", result.success, result.solve_time, result.status)
    print("dt", result.interpolation_dt)

    # 如果启用绘图且轨迹生成成功，则绘制轨迹
    if PLOT and result.success.item():
        plot_traj(traj, result.interpolation_dt)

    return traj  # 返回生成的轨迹

# 打印机器人状态的函数
def print_robot_states(robot, monitoring):
    while monitoring:
        robot.getRobotStates(robot_states)
        tcp_pose = robot_states.tcpPose
        tcp_des = robot_states.tcpPoseDes
        posx.append(tcp_pose[0])
        posy.append(tcp_pose[1])
        posz.append(tcp_pose[2])
        desx.append(tcp_des[0])
        desy.append(tcp_des[1])
        desz.append(tcp_des[2])
        rx.append(tcp_pose[3])
        ry.append(tcp_pose[4])
        rz.append(tcp_pose[5])
        time.sleep(0.01)

# 主程序
def main():
    setup_curobo_logger("error")

    frequency = 100
    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.104"
    robot_states = flexivrdk.RobotStates()
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # 定义各个操作点的位置
    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    pre_catch = [0.507623, -0.097015, 0.536714, -0.896, -179.517, 47.301]
    catch = [0.507623, -0.097015, 0.489650, 0.377, 166.092, 121.051]
    up = [0.497438, -0.081409, 0.536714, -0.896, -179.517, 47.301]
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    place_tube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846]

    # 初始化机器人
    robot = flexivrdk.Robot(robot_ip, local_ip)
    gripper = flexivrdk.Gripper(robot)

    # 清除故障
    if robot.isFault():
        log.warn("在机器人服务器上发生了故障，尝试清除...")
        robot.clearFault()
        time.sleep(2)
        if robot.isFault():
            log.error("无法清除故障，退出...")
            return
        log.info("机器人服务器上的故障已清除")

    # 启用机器人
    log.info("正在启用机器人...")
    robot.enable()

    while not robot.isOperational():
        time.sleep(1)
    log.info("机器人现在可操作")

    while robot.isBusy():
        time.sleep(1)

    # 安全原点
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.executePrimitive("MoveJ(target=-10.99 -10.53 -21.30 71.14 6.07 -8.95 -8.99, relative=false)")
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)

    # 切换到非实时关节位置控制模式
    robot.setMode(mode.NRT_JOINT_POSITION)
    period = 1.0 / frequency
    loop_time = 0
    print(f"以{frequency}Hz或{period}秒间隔向机器人发送命令")

    robot.getRobotStates(robot_states)
    init_pos = robot_states.q.copy()
    print("初始位置设定为:", init_pos)

    DOF = len(robot_states.q)
    print("robot_states.q:", robot_states.q)

    # 初始化目标向量
    target_pos = init_pos.copy()
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF

    # 关节运动约束
    MAX_VEL = [0.8] * DOF
    MAX_ACC = [1.0] * DOF

    # 移动至预设位置
    gripper.move(0.015, 0.003, 30)
    q = [-10.99, -10.53, -21.30, 71.14, 6.07, -8.95, -8.99]
    for i in range(len(q)):
        q[i] = q[i]*np.pi/180
    print(q)

    target_place = np.array([0.510623, -0.097015, 0.536714])
    target_orient = np.array([-0.06241533160209656, 0.8639705181121826, -0.488759845495224, -0.10379336029291153])

    # 运动生成
    traj = demo_motion_gen(begin_q=q, target_position=target_place, target_orientation=target_orient)
    art_pos = traj.position.cpu().numpy()
    loop_counter = len(art_pos)
    count = 0
    print("pos", len(art_pos))

    monitoring = True
    print_thread = threading.Thread(target=print_robot_states, args=[robot, monitoring])
    print_thread.start()

    # 循环发送命令
    while loop_time <= loop_counter * period:
        time.sleep(period)
        if robot.isFault():
            raise Exception("在机器人服务器上发生了故障，退出...")
        for i in range(DOF):
            target_pos[i] = art_pos[count][i]
        robot.sendJointPosition(target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC)
        count += 1
        loop_time += period

    time.sleep(3.5)

    # 执行抓取动作
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    log.info("MoveL:抓取")
    robot.executePrimitive(f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.06)")
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    gripper.move(0.0087, 0.1, 40)
    time.sleep(2)
   # 获取机器人当前状态，包括TCP（工具中心点）的期望位置和关节角度
robot.getRobotStates(robot_states)
print("抓取位置的期望TCP位置:", robot_states.tcpPoseDes)
q2 = robot_states.q
print("抓取位置的关节角度:", q2)

# 信息日志：移动到抬升位置
log.info("MoveL:抬升")
# 执行原始指令移动到抬升位置
robot.executePrimitive(f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)")
# 等待直到机器人到达目标位置
while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
    time.sleep(1)

# 设置机器人模式为非实时关节位置控制
robot.setMode(mode.NRT_JOINT_POSITION)
# 定义新的起始关节角度
q4 = [-0.055025793612, -0.01993733, 0.0812698975, 1.4333751202, 0.002351583214, -0.13014896214, -0.8009197711944]
fou_pos = q4
# 目标位置和方向
fou_place = np.array([0.475248, -0.369169, 0.536714])
fou_orient = np.array([0.0053847534582, 0.192058250308, 0.981174767, 0.019512292])
# 生成移动轨迹
move_traj = demo_motion_gen(begin_q=fou_pos, target_position=fou_place, target_orientation=fou_orient)
move_art_pos = move_traj.position.cpu().numpy()
move_loop_counter = len(move_art_pos)
move_count = 0
print("移动轨迹点数量:", len(move_art_pos))

# 循环发送关节位置命令，使机器人沿轨迹移动
while move_time <= move_loop_counter * period:
    time.sleep(period)

    # 检查是否有故障发生
    if robot.isFault():
        raise Exception("机器人服务器发生故障，退出...")

    # 更新目标位置
    for i in range(DOF):
        target_pos[i] = move_art_pos[move_count][i]
    # 发送关节位置命令
    robot.sendJointPosition(target_pos, target_vel, target_acc, MAX_VEL, MAX_ACC)
    
    # 更新计数器和时间变量
    move_count += 1
    move_time += period
time.sleep(3.5)

# 设置机器人模式为原始指令执行
robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
# 信息日志：移动到放置管子位置
log.info("MoveL:放置管子")
# 执行原始指令移动到放置管子位置
robot.executePrimitive(f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.09)")
# 等待直到机器人到达目标位置
while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
    time.sleep(1)
# 获取放置位置的TCP期望位置
robot.getRobotStates(robot_states)
print("放置位置的期望TCP位置:", robot_states.tcpPoseDes)

# 停止监控线程
monitoring = False

# 信息日志：打开夹爪
log.info("打开夹爪")
# 打开夹爪
gripper.move(0.015, 0.1, 20)
time.sleep(1.5)

# 停止机器人
robot.stop()

# 将记录的数据写入CSV文件
filename = 'trajectory_test.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入列标题
    writer.writerow(['X', 'Y', 'Z', 'desx', 'desy', 'desz', 'rx', 'ry', 'rz'])
    # 写入数据
    for x, y, z, xd, yd, zd, rax, ray, raz in zip(posx, posy, posz, desx, desy, desz, rx, ry, rz):
        writer.writerow([x, y, z, xd, yd, zd, rax, ray, raz])
print(f"数据已保存到 {filename}")

# 绘制3D轨迹图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(posx, posy, posz, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# 保存并显示图形
plt.savefig('trajectory_test.png')
plt.show()

# 异常捕获
except Exception as e:
    # 记录错误信息
    log.error(str(e))

# 主函数入口
if __name__ == "__main__":
    # 调用主函数
    main()
