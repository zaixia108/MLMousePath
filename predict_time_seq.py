import random
import time
import os
import numpy as np
import onnxruntime
import cv2
import pyautogui
import ctypes
import win32con

def mouse_move_rel(dx, dy, duration):
    ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    duration = float(duration)
    time.sleep(duration)

model = 'mouse_model.onnx'

ort_session = onnxruntime.InferenceSession(model)
input_name = ort_session.get_inputs()[0]
print(input_name)
output_name = ort_session.get_outputs()[0]
print(output_name)


def predict(dx_dy):
    print(dx_dy)
    return ort_session.run([output_name.name], {'input': dx_dy})[0]



# x = random.random() * 200 * random.choice([-1, 1])
# y = random.random() * 200 * random.choice([-1, 1])
ix = 200
iy = 200
input_tensor = np.array([[ix, iy, 0.1]], dtype=np.float32)
# input_data = input_tensor.squeeze(1)
t1 = time.time()
result = predict(input_tensor)[0]
t2 = time.time()
print(result)
print(f"Prediction time: {(t2 - t1)*1000} ms")
last_point = [0, 0]
last_time = None
ti1 = time.time()
now_mouse_pos1 = pyautogui.position()
print(now_mouse_pos1)
for item in result:
    # print(item)
    x = int(item[0])
    y = int(item[1])
    times = item[2]/1000
    if last_time:
        utime = times - last_time
        last_time = times
    else:
        utime = times
        last_time = times
    # print(times)
    if utime < 0:
        utime = 0.001
    print(utime)
    if last_point == [0, 0]:
        last_point = [x, y]
        # pyautogui.moveRel(x, y, duration=times)
        print(x, y)
        mouse_move_rel(x, y, utime)
    else:
        delta_x = int(x - last_point[0])
        delta_y = int(y - last_point[1])
        print(delta_x, delta_y)
        # print(delta_x, delta_y)
        last_point = [x, y]
        # pyautogui.moveRel(delta_x, delta_y, duration=times)
        mouse_move_rel(delta_x, delta_y, utime)
now_mouse_pos2 = pyautogui.position()
print(now_mouse_pos2)
ti2 = time.time()
print(f'移动距离：{now_mouse_pos2[0] - now_mouse_pos1[0]}, {now_mouse_pos2[1] - now_mouse_pos1[1]}')
print(f"Prediction time: {(ti2 - ti1)*1000} ms")
    # x = int(item[0]) + 500
    # y = int(item[1]) + 500
    # pyautogui.moveTo(x, y, duration=times)
# img = np.zeros((400, 400, 3), np.uint8)
# last_point = [0,0]
# for item in result[0]:
#     x = int(item[0])
#     y = int(item[1])
#     print(x, y)
#     if last_point == [0,0]:
#         last_point = [x, y]
#         cv2.circle(img, (x + 200, y + 200), 1, (255, 255, 255), -1)
#     else:
#         cv2.line(img, (last_point[0] + 200, last_point[1] + 200), (x + 200, y + 200), (255, 255, 255), 1)
#         last_point = [x, y]
# cv2.imshow('image', img)
# cv2.waitKey(1000)




