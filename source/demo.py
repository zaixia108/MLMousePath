import os
import numpy as np
import onnxruntime
import ctypes


path = os.getcwd()
dll = os.path.join(path, "mouse.dll")
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


ort_session = onnxruntime.InferenceSession(model)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


def mouse_output(start, end, absulute=False) -> list:
    """

    :param start: 鼠标开始位置
    :param end: 鼠标结束位置
    :param absulute: 是否输出绝对坐标, False 时 输出每次移动的距离, default False
    :return:
    """
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


if __name__ == '__main__':
    start = (0, 0)
    end = (100, 100)
    print(mouse_output(start, end, absulute=False))
