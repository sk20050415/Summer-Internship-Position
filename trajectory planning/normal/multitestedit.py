#!/usr/bin/env python
# 单行抓取（此操作实现了单行的抓取，但未实现移动的操作，也并未用到camera，属于测试的代码）

import time
import argparse

# 工具方法
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

# 添加Flexiv RDK库路径
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk

def main():
    # 设置机器人和本地IP地址
    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.104"

    # 初始化日志和运动模式
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # 定义各位置坐标
    # 安全起始位置
    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    # 预抓取位置
    precatch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]
    # 抓取位置
    catch = [0.487420, -0.071394, 0.466700, -0.896, -179.517, 47.304]
    # 抬升位置
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301]
    # 移动管子位置
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    # 放置管子位置
    place_tube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846]
    # 移动上抬位置
    move_up = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]

    # X轴增量
    x_increment = 0
    # 更新所有位置的X坐标
    precatch[0] += x_increment
    catch[0] += x_increment
    up[0] += x_increment
    move_tube[0] += x_increment
    place_tube[0] += x_increment
    move_up[0] += x_increment

    # 尝试连接并控制机器人
    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # 清除机器人服务器上的任何故障
        if robot.isFault():
            log.warn("机器人服务器发生故障，尝试清除...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("无法清除故障，退出...")
                return
            log.info("机器人服务器上的故障已清除")

        # 启用机器人
        log.info("正在启用机器人...")
        robot.enable()

        # 等待机器人变为可操作状态
        while not robot.isOperational():
            time.sleep(1)
        log.info("机器人现在可以操作")

        # 设置为实时原始执行模式
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)

        # 设置循环次数
        num = 1
        total = 5
        increment = 0.009

        # 移动到安全起始位置
        log.info("移动到安全起始位置")
        robot.executePrimitive(f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.1)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        
        # 调整夹爪宽度
        gripper.move(0.0125, 0.01, 30)
        time.sleep(2)

        # 循环抓取
        while num < total:
            # 移动到预抓取位置
            log.info("移动到预抓取位置")
            robot.executePrimitive(f"MoveL(target={list2str(precatch)} WORLD WORLD_ORIGIN, maxVel=0.1)")
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

            # 抬升
            log.info("抬升")
            robot.executePrimitive(f"MoveL(target={list2str(up)} WORLD WORLD_ORIGIN, maxVel=0.08)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 打开夹爪
            log.info("打开夹爪")
            gripper.move(0.0125, 0.1, 20)
            time.sleep(1)

            # 增加计数器，更新位置
            num += 1
            precatch[1] -= increment
            catch[1] -= increment
            up[1] -= increment
            move_tube[1] -= increment
            place_tube[1] -= increment
            move_up[1] -= increment
            print("precatch[1]:", precatch[1])
            print("num:", num)

        # 停止机器人
        robot.stop()

    except Exception as e:
        # 打印异常错误信息
        log.error(str(e))

if __name__ == "__main__":
    main()
