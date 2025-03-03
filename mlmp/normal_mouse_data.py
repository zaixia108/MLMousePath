import math
import random
import tkinter as tk
import csv
import os


class MouseDataCollector:
    BALL_RADIUS = 30
    OFFSET_RANGE = 500
    TARGET_SAMPLES = 100
    CSV_HEADER = ['target'] + [f'kp{i}' for i in range(10)]

    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # 状态变量
        self.recording = False
        self.mouse_path = []
        self.sample_count = 0

        # 初始化界面元素
        self.create_widgets()
        self.setup_balls()
        self.setup_file_handler()

        # 事件绑定
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_mouse_click)
        self.root.bind('<Key>', self.on_key_press)
        self.root.mainloop()

    def create_widgets(self):
        """创建界面组件"""
        self.status_label = tk.Label(
            self.root,
            text=f"Collected: {self.sample_count}/{self.TARGET_SAMPLES}",
            font=("Helvetica", 16)
        )
        self.status_label.pack()

        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height)
        self.canvas.pack()

    def setup_balls(self):
        """初始化球体位置"""
        self.start_ball_pos = (self.screen_width // 2, self.screen_height // 2)
        self.target_ball_pos = self.generate_valid_position()

        # 绘制初始球体
        self.draw_ball(*self.start_ball_pos, "red")
        self.draw_ball(*self.target_ball_pos, "blue", "target")

    def setup_file_handler(self):
        """初始化CSV文件处理"""
        self.csv_path = "mouse_data.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.CSV_HEADER)

    def draw_ball(self, x, y, color, tag=None):
        """绘制球体辅助方法"""
        return self.canvas.create_oval(
            x - self.BALL_RADIUS,
            y - self.BALL_RADIUS,
            x + self.BALL_RADIUS,
            y + self.BALL_RADIUS,
            fill=color,
            tags=tag
        )

    def generate_valid_position(self):
        """生成符合条件的目标球位置"""
        while True:
            dx = random.randint(-self.OFFSET_RANGE, self.OFFSET_RANGE)
            dy = random.randint(-self.OFFSET_RANGE, self.OFFSET_RANGE)
            new_x = self.start_ball_pos[0] + dx
            new_y = self.start_ball_pos[1] + dy

            # 边界检查
            in_bounds = (
                    self.BALL_RADIUS <= new_x <= self.screen_width - self.BALL_RADIUS and
                    self.BALL_RADIUS <= new_y <= self.screen_height - self.BALL_RADIUS
            )

            # 距离检查（至少2倍半径）
            distance = math.hypot(
                new_x - self.start_ball_pos[0],
                new_y - self.start_ball_pos[1]
            )
            if in_bounds and distance >= 2 * self.BALL_RADIUS:
                return (new_x, new_y)

    def on_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.recording:
            self.mouse_path.append((event.x, event.y))

    def on_mouse_click(self, event):
        """处理鼠标点击事件"""
        if self.is_clicked_ball(event, self.start_ball_pos):
            self.start_recording(event)
        elif self.is_clicked_ball(event, self.target_ball_pos):
            self.stop_recording()

    def is_clicked_ball(self, event, ball_pos):
        """判断是否点击到球体"""
        x, y = event.x, event.y
        return (
                ball_pos[0] - self.BALL_RADIUS <= x <= ball_pos[0] + self.BALL_RADIUS and
                ball_pos[1] - self.BALL_RADIUS <= y <= ball_pos[1] + self.BALL_RADIUS
        )

    def start_recording(self, event):
        """开始记录路径"""
        self.recording = True
        self.mouse_path = [(event.x, event.y)]

    def stop_recording(self):
        """停止记录并保存数据"""
        self.recording = False
        if len(self.mouse_path) >= 2:  # 过滤无效点击
            self.save_data()
            self.sample_count += 1
            self.update_ui()
            self.reset_target_ball()

        if self.sample_count >= self.TARGET_SAMPLES:
            self.root.destroy()

    def save_data(self):
        """处理并保存数据到CSV"""
        try:
            # 转换坐标系
            base_x, base_y = self.mouse_path[0]
            x_rel = [x - base_x for x, y in self.mouse_path]
            y_rel = [base_y - y for x, y in self.mouse_path]  # 转换为笛卡尔坐标系

            # 均匀采样关键点
            key_points = self.sample_key_points(x_rel, y_rel)
            self.write_to_csv(key_points)
        except Exception as e:
            print(f"Error saving data: {e}")

    def sample_key_points(self, x, y):
        """均匀采样10个关键点（带插值）"""
        num_points = 10
        indices = []
        path_length = len(x)

        if path_length >= num_points:
            step = max(1, (path_length - 1) / (num_points - 1))
            indices = [int(i * step) for i in range(num_points)]
        else:
            # 对于短路径进行线性插值
            indices = list(range(path_length))
            indices += [path_length - 1] * (num_points - path_length)

        return [
            (x[i], y[i])
            for i in indices[:num_points]  # 确保正好10个点
        ]

    def write_to_csv(self, key_points):
        """写入CSV文件"""
        target = f"{key_points[-1][0]},{key_points[-1][1]}"
        kp_data = [f"{x},{y}" for x, y in key_points]

        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([target] + kp_data)

    def update_ui(self):
        """更新界面状态"""
        self.status_label.config(
            text=f"Collected: {self.sample_count}/{self.TARGET_SAMPLES}")
        self.canvas.update()

    def reset_target_ball(self):
        """重置目标球位置"""
        self.target_ball_pos = self.generate_valid_position()
        self.canvas.delete("target")
        self.draw_ball(*self.target_ball_pos, "blue", "target")

    def on_key_press(self, event):
        """处理键盘事件"""
        if event.keysym == "Escape":
            self.root.destroy()


if __name__ == "__main__":
    MouseDataCollector()