import ctypes
import win32con


def disable_mouse_acceleration():
    # 获取当前鼠标参数
    mouse_params = ctypes.c_int(0)
    ctypes.windll.user32.SystemParametersInfoA(win32con.SPI_GETMOUSE, 0, ctypes.byref(mouse_params), 0)

    # 创建新参数，禁用加速
    new_params = (ctypes.c_int * 3)(0, 0, 0)

    # 应用新参数
    ctypes.windll.user32.SystemParametersInfoA(win32con.SPI_SETMOUSE, 0, new_params, 0)


def restore_mouse_acceleration(original_params):
    # 恢复原始鼠标参数
    ctypes.windll.user32.SystemParametersInfoA(win32con.SPI_SETMOUSE, 0, original_params, 0)
