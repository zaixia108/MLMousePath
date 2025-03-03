import math
import random
import tkinter as tk
import matplotlib.pyplot as plt
import csv
import time
from tkinter import Label

# 创建窗口
root = tk.Tk()
root.attributes('-fullscreen', True)  # 全屏显示

label_n = Label(root, text="n: 0", font=("Helvetica", 16))
label_n.pack()

csv_file_path = "mouse_data_time_seq.csv"

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 设置小球的初始位置
ball1_pos = (screen_width / 2, screen_height / 2)
ball2_pos = (ball1_pos[0] + random.randint(-500, 500), ball1_pos[1] + random.randint(-500, 500))

# 设置小球的半径
ball_radius = 20

# 设置鼠标记录状态
recording = False
mouse_path = []
time_stamps = []  # 新增：记录时间戳

n = 0


# 鼠标移动事件处理函数
def motion(event):
    global recording, mouse_path, time_stamps
    if recording:
        current_time = time.time() * 1000  # 转换为毫秒
        mouse_path.append((event.x, event.y))
        time_stamps.append(current_time)


# 鼠标点击事件处理函数
def mouse_click(event):
    global recording, mouse_path, ball2_pos, n, time_stamps

    if (event.x >= ball1_pos[0] - ball_radius and
            event.x <= ball1_pos[0] + ball_radius and
            event.y >= ball1_pos[1] - ball_radius and
            event.y <= ball1_pos[1] + ball_radius):
        recording = True
        current_time = time.time() * 1000  # 转换为毫秒
        mouse_path = [(event.x, event.y)]
        time_stamps = [current_time]
    elif (event.x >= ball2_pos[0] - ball_radius and
          event.x <= ball2_pos[0] + ball_radius and
          event.y >= ball2_pos[1] - ball_radius and
          event.y <= ball2_pos[1] + ball_radius):
        recording = False
        canvas.delete("ball2")
        save_to_csv(mouse_path, time_stamps)
        n = n + 1
        if n == 100:
            root.destroy()
        label_n.config(text=f"n: {n}")
        mouse_path = []
        time_stamps = []

        # 重新生成第二个小球的位置
        ball2_pos = (ball1_pos[0] + random.randint(-500, 500),
                     ball1_pos[1] + random.randint(-500, 500))

        # 绘制新的第二个小球
        canvas.create_oval(ball2_pos[0] - ball_radius, ball2_pos[1] - ball_radius,
                           ball2_pos[0] + ball_radius, ball2_pos[1] + ball_radius,
                           fill="blue", tags="ball2")


# 键盘事件处理函数
def key(event):
    if event.keysym == "Escape":
        root.destroy()


def save_to_csv(path, timestamps):
    # 确保至少有一个点
    if not path or not timestamps:
        return

    # 将路径坐标转换为相对于起点的坐标
    x_rel = [px - path[0][0] for px, py in path]
    y_rel = [-(py - path[0][1]) for px, py in path]

    # 将时间戳转换为相对时间（相对于第一个点的时间）
    relative_times = [t - timestamps[0] for t in timestamps]

    # 选择10个关键点
    num_points = len(path)
    key_points_indices = [int(i) for i in range(0, num_points, max(1, num_points // 10))][:10]

    # 如果点数不足10个，补充最后一个点
    while len(key_points_indices) < 10:
        key_points_indices.append(key_points_indices[-1])

    # 提取关键点的数据
    key_points = []
    for i in key_points_indices:
        x = x_rel[i]
        y = y_rel[i]
        t = relative_times[i]
        key_points.append(f"{x},{y},{t}")

    # 添加终点的相对坐标和时间
    final_point = f"{x_rel[-1]},{y_rel[-1]},{relative_times[-1]}"

    # 写入CSV文件
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([final_point] + key_points)


def visualize_path(path, timestamps):
    if not path or not timestamps:
        return

    # 将路径坐标转换为相对于起点的坐标
    x_rel = [px - path[0][0] for px, py in path]
    y_rel = [-(py - path[0][1]) for px, py in path]

    # 计算相对时间
    relative_times = [t - timestamps[0] for t in timestamps]

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D路径
    scatter = ax.scatter(x_rel, y_rel, relative_times,
                         c=relative_times, cmap='viridis')

    # 设置轴标签
    ax.set_xlabel('X relative')
    ax.set_ylabel('Y relative')
    ax.set_zlabel('Time (ms)')

    # 添加颜色条
    plt.colorbar(scatter, label='Time (ms)')

    plt.show()


# 创建画布和小球
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()
canvas.create_oval(ball1_pos[0] - ball_radius, ball1_pos[1] - ball_radius,
                   ball1_pos[0] + ball_radius, ball1_pos[1] + ball_radius,
                   fill="red")
canvas.create_oval(ball2_pos[0] - ball_radius, ball2_pos[1] - ball_radius,
                   ball2_pos[0] + ball_radius, ball2_pos[1] + ball_radius,
                   fill="blue", tags="ball2")

# 绑定事件
canvas.bind('<Motion>', motion)
canvas.bind('<Button-1>', mouse_click)
root.bind('<Key>', key)

# 运行窗口
root.mainloop()