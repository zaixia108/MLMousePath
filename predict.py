import random
import time
import os
import numpy as np
import onnxruntime
import cv2
import ctypes

import pyautogui
import win32con
import pydirectinput
from mouse_ghub import mouse_xy


# driver = ctypes.CDLL(r'.\MouseControl.dll')

def mouse_move_rel(dx, dy, duration=0.01, stpes=10):
    # 使用 pydirectinput 实现相对移动
    # pydirectinput.move(dx, dy, relative=True)
    mouse_xy(int(dx), int(dy))
    time.sleep(0.01)


path = os.getcwd()
dll = os.path.join(path, "mouse.dll")
print(os.path.exists(dll))  # 应该返回 True
print(f'从 {dll} 读取模型')
def load_from_dll(dll_path):
    dll = ctypes.WinDLL(dll_path)
    dll.get_model_data.restype = ctypes.POINTER(ctypes.c_ubyte)
    dll.get_model_size.restype = ctypes.c_size_t
    ptr = dll.get_model_data()
    size = dll.get_model_size()
    buffer = ctypes.create_string_buffer(size)
    ctypes.memmove(buffer, ptr, size)
    return buffer.raw
model = load_from_dll(dll)
model2 = 'mouse.onnx'

ort_session = onnxruntime.InferenceSession(model)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


def predict(dx_dy):
    return ort_session.run([output_name], {'input': dx_dy})[0]


def mouse_output(start, end, absulute=False):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    input_tensor = np.array([[[dx, dy]]], dtype=np.float32)
    result = ort_session.run([output_name], {'input': input_tensor})[0][0]
    ret_list = []
    if absulute:
        for item in result:
            x = int(item[0]) + start[0]
            y = int(item[1]) + start[1]
            ret_list.append((x, y))
        ret_list.append(end)
    else:
        last_point = [0, 0]
        for item in result:
            x = int(item[0])
            y = int(item[1])
            if last_point == [0, 0]:
                last_point = [x, y]
                ret_list.append((x, y))
            else:
                delta_x = x - last_point[0]
                delta_y = y - last_point[1]
                last_point = [x, y]
                ret_list.append((delta_x, delta_y))
        ret_list.append(((end[0]-last_point[0]), (end[1]-last_point[1])))
    return ret_list


def test():
    x = random.random() * 200 * random.choice([-1, 1])
    y = random.random() * 200 * random.choice([-1, 1])
    input_tensor = np.array([[[x, y]]], dtype=np.float32)
    t1 = time.time()
    result = predict(input_tensor)
    t2 = time.time()
    print(f"Prediction time: {(t2 - t1)*1000} ms")
    img = np.zeros((400, 400, 3), np.uint8)
    last_point = [0,0]
    for item in result[0]:
        x = int(item[0])
        y = int(item[1])
        print(x, y)
        if last_point == [0,0]:
            last_point = [x, y]
            cv2.circle(img, (x + 200, y + 200), 1, (255, 255, 255), -1)
        else:
            cv2.line(img, (last_point[0] + 200, last_point[1] + 200), (x + 200, y + 200), (255, 255, 255), 1)
            last_point = [x, y]
    cv2.imshow('image', img)
    cv2.waitKey(1000)

total_time_1 = time.time()
for i in range(1):
    t1 = time.time()
    path = mouse_output((0,0), (0, 200), absulute=True)
    t2 = time.time()
    img = np.zeros((400, 400, 3), np.uint8)
    last_point = [0,0]
    # print(path)
    for item in path:
        x = int(item[0])
        y = int(item[1])
        # print(x, y)
        if last_point == [0,0]:
            last_point = [x, y]
            cv2.circle(img, (x + 200, y + 200), 1, (255, 255, 255), -1)
        else:
            cv2.line(img, (last_point[0] + 200, last_point[1] + 200), (x + 200, y + 200), (255, 255, 255), 1)
            last_point = [x, y]

    pos11 = pyautogui.position()
    for point in path:
        print(point)
        pos1 = pyautogui.position()
        # mouse_move_rel(point[0], point[1])
        pos2 = pyautogui.position()
        print(f'移动距离：{pos2[0] - pos1[0]}, {pos2[1] - pos1[1]}')

    pos12 = pyautogui.position()

    print(f'移动距离：{pos12[0] - pos11[0]}, {pos12[1] - pos11[1]}')
    cv2.imshow('img', img)
    cv2.waitKey(0)
    print(f"Prediction time: {(t2 - t1)*1000} ms")
# total_time_2 = time.time()
# print(f"千次推理用时: {(total_time_2 - total_time_1)*1000} ms")
# test()


