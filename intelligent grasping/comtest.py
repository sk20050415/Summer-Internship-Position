#!/usr/bin/env python
import time
import argparse
import pyzed.sl as sl
import numpy as np
import cv2
import math
from openai import OpenAI
import json
import os
import requests
import base64

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

import sys
sys.path.insert(0, "/home/robot/motion/flexiv_rdk/lib_py")
import flexivrdk

import robot_actions
MY_API = "API"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    robot_ip = "192.168.2.100"            
    local_ip = "192.168.2.104"

    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    safe_origin = [0.471448, -0.369169, 0.536714, 177.687, 0.176, 157.846]
    photo_pose = [0.436954, -0.171356, 0.516743, 179.042, 14.506, 179.717]
    begin_pose = [0.436954, -0.171356, 0.536743, 177.687, 0.176, 179.717]

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.depth_minimum_distance = 0.3

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    image = sl.Mat()
    runtime = sl.RuntimeParameters(enable_fill_mode = True)
    
    
    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)
        robot_states = flexivrdk.RobotStates()

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot
        log.info("Enabling robot ...")
        robot.enable()

        while not robot.isOperational():
            time.sleep(1)
        log.info("Robot is now operational")

        # primitive execution mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        gripper = flexivrdk.Gripper(robot)
        
        log.info("MoveL:safe origin")
        robot.executePrimitive(
            f"MoveL(target={list2str(safe_origin)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        gripper.move(0.0125, 0.01, 30)
        time.sleep(1)

        log.info("MoveL:photo")
        robot.executePrimitive(
            f"MoveL(target={list2str(photo_pose)} WORLD WORLD_ORIGIN, maxVel=0.1)"
        )
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        time.sleep(1)
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            img = image.get_data()
            cv2.imwrite('tube_test.jpg', img)
            zed.close() # Close the camera
        
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

        client= OpenAI(
            api_key=MY_API,
        )
        
        IMAGE_PATH = "/home/robot/motion/tube_test.jpg"
        base64_image = encode_image(IMAGE_PATH)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":"user",
                    "content":[
                        {"type":"text","text":"\nThe image is a tube rack filling with tubes. However, there may be some abnormal tubes which are not in the right place. Understand this scene and generate a scenery description. Information about environments is given as python dictionary. \nExample1:\n{\n\"objects\": [\n\t\"<tubes>\",\n\t\"<abnormal tube>\",\n\t\"<tube rack>\"],\n\"number of abnormal tubes\": 2,\n\"state of abnormal tubes\": [\"lying_down\", \"lying_down\"],\n\"spatial relation between abnormal tubes\": [\"in contact\", \"crossed\"],\n\"your_explanation\": \"There are many tubes in the tube rack, and there are two abnormal tubes on the rack. The two abnormal tubes are in contact, and are crossed with each other.\"\n}\n- The \"objects\" field denotes the list of objects. Enclose the object names with '<' and '>'.Connect the words without spaces, using underscores instead. You should list all the objects related to the tubes. \n- Distinguish the \"tubes\" and  \"abnormal tube\". The \"tubes\" refer to the normal tubes placed correctly in the tube rack. The \"abnormal tube\" refers to the tube which is inclined or lying down, not in the slots of the rack. \n- Please list the number of abnormal tubes in the \"number of abnormal tubes\" field. If there is no abnormal tubes, the number is 0.\n- If the number of abnormal tubes is not zero, describe the state of abnormal tubes with \"lying down\" or \"leaning\" in the \"state of abnormal tubes\" field. Distinguish \"lying down\" and  \"leaning\". \"lying_down\" means the abnormal tube is lying down atop the normal upright tubes. \"leaning\" means the abnormal tube is leaning against other tubes and lying horizontally atop the rack. If there are two abnormal tubes, describe the state respectively.\n- If the number of abnormal tubes is more than one, describe the spatial relationship between the abnormal tubes with two keywords in the \"spatial relation between abnormal tubes\" field.The spatial relation means they are in touch or stay apart. \n- As to the first keyword in the \"spatial relation between abnormal tubes\" field, if the two abnormal tubes are in touch with each other, describe as\"in contact\". If the abnormal tubes stay apart, describe the relation as \"separate\".\n- As to the second keyword in the \"spatial relation between abnormal tubes\" field, if the two abnormal tubes are \"in contact\", describe the spatial relation with \"crossed\" or \"parallel\". If not, describe as \"none\".\n- describe the objects and the spatial relation between the abnormal tubes in one or two sentences in the \"your explanation\" field.\nPlease take note of the following.\n1. The response should be a Python dictionary only, without any explanatory text (e.g., Do not include a sentence like \"here is the environment\").\n2. Insert \"```python\" at the beginning and insert \"```\" at end of your response.\n\n"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}
                        }
                    ],
                }
            ],
            temperature=0.0,
        )

        output1 = response.choices[0].message.content
        output1 = output1[len("```python"): -len("```")]

        response_dict = json.loads(output1)
        objects = response_dict["objects"]
        num_abnormal_tubes = response_dict["number of abnormal tubes"]
        abnormal_tube_state = response_dict["state of abnormal tubes"]
        spatial_relation = response_dict["spatial relation between abnormal tubes"]
        print(num_abnormal_tubes)
        if num_abnormal_tubes == 1:
            print(abnormal_tube_state)
        if len(spatial_relation) > 1:
            relationship = spatial_relation[1]

        if num_abnormal_tubes > 1:
            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role":"user",
                        "content":[
                            {"type":"text","text":"\nThe image is a tube rack filling with tubes. However, there are some abnormal tubes which are not in the right place. Understand this scene and generate a scenery description as a python dictionary. For example: \n{\n\"spatial relation\": \" \",\n\"explanation\": \" \"\n} \n- Distinguish the \"tubes\" and  \"abnormal tube\". The \"tubes\" refer to the normal tubes placed correctly in the tube rack. The \"abnormal tube\" refers to the tube which is inclined or lying down, not in the slots of the rack.\n- Please describe the spatial relation between the abnormal tubes. The spatial relation means they are \"in touch\" or stay \"apart\". The spatial relation is between abnormal tubes, not between abnormal tube and normal tubes.\n- The response should be a Python dictionary only, without any explanatory text (e.g., Do not include a sentence like \"here is the environment\"). The keys in dictionary only include \"spatial relation\" and \"explanation\".\n- Please give the spatial relation in the \"spatial relation\" field, with \"in touch\" or \"apart\" to describe. Describe the scene in the \"explanation\" field.\n"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"}
                            }
                        ],
                    }
                ],
                temperature=0.0,
            )
            output2 = response2.choices[0].message.content
            output2 = output2.strip("```python").strip("```")
            response_dict2 = json.loads(output2)
            space_dis = response_dict2["spatial relation"]
            print(space_dis)
            if space_dis == "in touch":
                print(abnormal_tube_state)
                print(relationship)
            if space_dis == "apart":
                print(abnormal_tube_state)


        if num_abnormal_tubes == 0:
            print("normal pattern")
            robot_actions.act_normal(robot, log, gripper)
        if num_abnormal_tubes == 1:
            if abnormal_tube_state[0] == "lying_down":
                print("grasp in the middle")
                robot_actions.act_lie(robot, log, gripper)
            elif abnormal_tube_state[0] == "leaning":
                print("grasp from the top")
                robot_actions.act_lean(robot, log, gripper)
        if num_abnormal_tubes == 2:
            if space_dis == "apart":
                print("grasp either one in the middle")
                robot_actions.act_twoapart(robot, log, gripper)
            elif space_dis == "in touch":
                if relationship == "crossed":
                    print("push the one above and grasp")
                    robot_actions.act_twocross(robot, log, gripper)
                elif relationship == "parallel":
                    print("push one from the top and grasp")
                    robot_actions.act_twolie(robot, log, gripper)

        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
