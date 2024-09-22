# 导入必要的库
import time
import argparse
import pyzed.sl as sl
import numpy as np
import cv2
import math

# 导入自定义的工具方法

from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

# Flexiv RDK相关库的导入
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk

def main():
    # 设置机器人和本地IP地址
    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.104"

    # 初始化日志记录器和模式
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # 定义各种机器人姿态位置
    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]# 安全起始位置
    photo_pose = [0.436954, -0.171356, 0.516743, 179.042, 14.506, 179.717]# 拍照位置
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]# 开始位置

    pre_catch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]# 预抓取位置
    catch = [0.487420, -0.071394, 0.466700, -0.896, -179.517, 47.304]# 抓取位置
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]# 抬起位置
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]# 移动管子位置
    place_tube = [0.471448, -0.369169, 0.461266, 177.687, 0.176, 157.846]# 放置管子位置
    move_up = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]# 移动上抬位置

    # 对某些位置进行x轴增量调整
    x_increment = 0
    for pose in [pre_catch, catch, up, move_tube, place_tube, move_up]:
        pose[0] += x_increment

    # 初始化ZED相机
    zed = sl.Camera()
    init_params = sl.InitParameters()  # 创建初始化参数对象
    # 设置相机分辨率、帧率、深度模式等
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.3

    # 打开相机
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("打开相机失败，退出程序。")
        exit(1)

    # 创建图像矩阵和运行时参数
    image = sl.Mat()
    runtime = sl.RuntimeParameters(enable_fill_mode=True)

    # 尝试连接并控制机器人
    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # 清除机器人服务器上的任何故障
        if robot.isFault():
            log.warn("检测到机器人服务器上的故障，尝试清除...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("无法清除故障，退出...")
                return
            log.info("成功清除机器人服务器上的故障")

        # 启用机器人
        log.info("正在启用机器人...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)
        log.info("机器人现在处于操作状态")

        # 设置机器人执行模式为实时原始执行
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)

        # 设置循环次数和增量
        num = 0
        total = 2
        increment = 0.009

        # 移动到安全起始位置
        log.info("移动到安全起始位置")
        robot.executePrimitive(f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.1)")
        # 等待直到到达目标
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.0125, 0.01, 30)
        time.sleep(1)

        # 移动到拍照位置并拍照
        log.info("移动到拍照位置")
        robot.executePrimitive(f"MoveL(target={list2str(photo_pose)} WORLD WORLD_ORIGIN, maxVel=0.1)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        time.sleep(1)
        # 捕获图像并保存
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            img = image.get_data()
            cv2.imwrite('tube_zed2.jpg', img)
            zed.close()  # 关闭相机

        # 等待一段时间并打印信息
        time.sleep(6)
        print('''
        0
        []
        normal pattern''')

        # 移动到开始位置
        log.info("移动到开始位置")
        robot.executePrimitive(f"MoveL(target={list2str(begin_pose)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # 循环执行抓取和放置管子的动作
        while num < total:
            # 移动到预抓取位置
            log.info("移动到预抓取位置")
            robot.executePrimitive(f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.1)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 移动到抓取位置
            log.info("移动到抓取位置")
            robot.executePrimitive(f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 闭合夹爪
            log.info("闭合夹爪")
            gripper.move(0.0088, 0.003, 45)
            time.sleep(2)

            # 移动到抬起位置
            log.info("移动到抬起位置")
            robot.executePrimitive(f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.08)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 移动到移动管子的位置
            log.info("移动到移动管子的位置")
            robot.executePrimitive(f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.1)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 移动到放置管子的位置
            log.info("移动到放置管子的位置")
            robot.executePrimitive(f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.08)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 打开夹爪
            log.info("打开夹爪")
            gripper.move(0.0125, 0.1, 20)
            time.sleep(1)

            # 移动到移动上抬位置
            log.info("移动到移动上抬位置")
            robot.executePrimitive(f"MoveL(target={list2str(move_up)} WORLD WORLD_ORIGIN, maxVel=0.1)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 增加计数并调整位置
            num += 1
            for pose in [pre_catch, catch, up, move_tube, place_tube, move_up]:
                pose[1] -= increment
            print("pre_catch[1]", pre_catch[1])
            print("num", num)

        # 停止机器人
        robot.stop()

    except Exception as e:
        # 打印异常错误信息
        log.error(str(e))

# 如果是直接运行此脚本，则调用主函数
if __name__ == "__main__":
    main()

