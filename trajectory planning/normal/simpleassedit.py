#!/usr/bin/env python
# 该代码实现了原地抓取放下的功能，属于测试代码
import time
import argparse

# 导入辅助方法和Flexiv RDK库
from utility import quat2eulerZYX, parse_pt_states, list2str
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk

def main():
    # 设置机器人IP和本地IP
    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.104"

    # 初始化日志记录和模式选择
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # 定义各个运动点的坐标
    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]  # 安全起始点
    pre_catch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]  # 抓取前位置
    catch = [0.487420, -0.071394, 0.466700, -0.896, -179.517, 47.304]      # 抓取点
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]         # 抬起位置
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]   # 移动管子位置
    place_tube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846]  # 放置管子位置
    move_up = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]     # 移动到上方位置

    try:
        # 创建机器人对象
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # 清除机器人服务器上的故障（如果存在）
        if robot.isFault():
            log.warn("机器人服务器上发生故障，尝试清除...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("无法清除故障，退出...")
                return
            log.info("机器人服务器上的故障已清除")

        # 启用机器人
        log.info("启用机器人...")
        robot.enable()

        # 等待机器人变为操作状态
        while not robot.isOperational():
            time.sleep(1)

        log.info("机器人现在可操作")

        # 设置为实时原始执行模式
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)

        # 移动到安全起始点
        log.info("MoveL:安全起始点")
        robot.executePrimitive(f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
  
        # 打开机械爪
        log.info("打开机械爪")
        gripper.move(0.0125, 0.01, 30)
        time.sleep(2)

        # 调整x和y坐标增量
        x_increment = 0
        y_increment = 0
        pre_catch[0] += x_increment
        catch[0] += x_increment
        up[0] += x_increment
        move_tube[0] += x_increment
        place_tube[0] += x_increment
        pre_catch[1] -= y_increment
        catch[1] -= y_increment
        up[1] -= y_increment
        move_tube[1] -= y_increment
        place_tube[1] -= y_increment

        # 移动到抓取前的位置
        log.info("MoveL:抓取前的位置")
        robot.executePrimitive(f"MoveL(target={list2str(pre_catch)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # 获取当前TCP姿态
        robot.getRobotStates(robot_states)
        print(robot_states.tcpPose)
        print("抓取前目标姿态:", robot_states.tcpPoseDes)

        # 移动到抓取点
        log.info("MoveL:抓取点")
        robot.executePrimitive(f"MoveL(target={list2str(catch)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # 关闭机械爪
        log.info("关闭机械爪")
        gripper.move(0.0087, 0.003, 40)
        time.sleep(3)

        # 抬起
        log.info("MoveL:抬起")
        robot.executePrimitive(f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("抬起目标姿态:", robot_states.tcpPoseDes)

        # 移动管子
        log.info("MoveL:移动管子")
        robot.executePrimitive(f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        robot.getRobotStates(robot_states)
        print("放置高点目标姿态:", robot_states.tcpPoseDes)

        # 放置管子
        log.info("MoveL:放置管子")
        robot.executePrimitive(f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # 打开机械爪
        log.info("打开机械爪")
        gripper.move(0.0125, 0.1, 20)
        time.sleep(2)

        # 停止机器人
        robot.stop()

    except Exception as e:
        # 打印异常错误信息
        log.error(str(e))

# 当脚本被直接运行时，调用main函数
if __name__ == "__main__":
    main()
