import pyzed.sl as sl
import numpy as np
import cv2

from utility import parse_pt_states

import time
import base64
from PIL import Image
import os
import io
from openai import OpenAI

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

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

def gpt2():
    #相机初始化
    zed = sl.Camera()
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
    runtime = sl.RuntimeParameters(enable_fill_mode = True)

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.RIGHT)
        img = image.get_data()
        cv2.imwrite('cube_test.jpg', img)
        zed.close()

    text = "There are two cubes in the photo. One is green, the other is red.If the green cube is the left one, return 0. If the red cube is the left one, return 1.You can only return the figure. No other words."
    file = "/home/robot/motion/cube_test.jpg"
    result = invoke_with_image(query=text,image_file=file)
    time.sleep(3)
    return result

'''if __name__ == "__main__":
    main()'''
print(gpt2())