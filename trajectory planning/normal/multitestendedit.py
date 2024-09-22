#!/usr/bin/env python
# 这是一个测试的函数，测试夹爪的开合以及机械臂的移动
import time
import argparse
from utility import quat2eulerZYX, parse_pt_states, list2str
import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk

def main():
    # 定义机器人和本地IP地址
    robot_ip = "192.168.2.100"
    local_ip = "192.168.2.104"

    # 日志和模式初始化
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # 初始化一些关键点的位置坐标，单位为米和度
    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846] # 安全起始位置
    pre_catch = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301] # 预抓取位置
    catch = [0.487420, -0.071394, 0.466700, -0.896, -179.517, 47.304] # 抓取位置
    up = [0.487438, -0.071409, 0.536714, -0.896, -179.517, 47.301] # 抬升位置
    move_tube = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846] # 移动试管位置
    place_tube = [0.471448, -0.369169, 0.466266, 177.687, 0.176, 157.846] # 放置试管位置
    move_up = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846] # 再次抬升位置

    # 增量调整所有位置的X轴坐标
    x_increment = 0
    for pos in [pre_catch, catch, up, move_tube, place_tube, move_up]:
        pos[0] += x_increment

    try:
        # 连接至机器人
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # 清除机器人的故障状态
        if robot.isFault():
            log.warn("检测到机器人服务器上的故障，尝试清除...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("无法清除故障，退出程序...")
                return
            log.info("机器人服务器上的故障已清除")

        # 启用机器人
        log.info("正在启用机器人...")
        robot.enable()

        # 等待机器人进入操作状态
        while not robot.isOperational():
            time.sleep(1)
        log.info("机器人现在可操作")

        # 设置为实时原始执行模式
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)

        # 初始化循环变量
        num = 0
        total = 5
        increment = 0.009

        # 移动到安全起始位置
        log.info("移动到安全起始位置")
        robot.executePrimitive(f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.1)")
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        
        # 调整机械手开合
        gripper.move(0.0125, 0.01, 30)
        time.sleep(2)

        gripper.move(0.0088, 0.01, 30)
        time.sleep(2)

        # 循环执行任务
        while num < total:
            # 移动至移动试管位置
            log.info("移动至移动试管位置")
            robot.executePrimitive(f"MoveL(target={list2str(move_tube)} WORLD WORLD_ORIGIN, maxVel=0.07)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 移动至放置试管位置
            log.info("移动至放置试管位置")
            robot.executePrimitive(f"MoveL(target={list2str(place_tube)} WORLD WORLD_ORIGIN, maxVel=0.08)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 再次抬升
            log.info("再次抬升")
            robot.executePrimitive(f"MoveL(target={list2str(move_up)} WORLD WORLD_ORIGIN, maxVel=0.08)")
            while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
                time.sleep(1)

            # 更新计数器和位置增量
            num += 1
            for pos in [pre_catch, catch, up, move_tube, place_tube, move_up]:
                pos[1] -= increment
            print("预抓取Y坐标:", pre_catch[1])
            print("当前循环次数:", num)

        # 结束任务
        robot.stop()

    except Exception as e:
        # 打印异常错误信息
        log.error(str(e))

# 主函数入口
if __name__ == "__main__":
    main()
