import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# 设置使用cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MouseDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 解析终点数据
        end_point = torch.tensor([float(x) for x in row[0].split(',')], dtype=torch.float32)

        # 解析序列数据
        sequence = torch.zeros((10, 3), dtype=torch.float32)
        for i in range(10):
            point_data = [float(x) for x in row[i + 1].split(',')]
            sequence[i] = torch.tensor(point_data, dtype=torch.float32)

        return end_point, sequence


class MouseTrajectoryModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(MouseTrajectoryModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 输出x, y, time
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 编码输入
        x = self.encoder(x)
        x = x.unsqueeze(1).repeat(1, 10, 1)

        # LSTM处理
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 解码输出序列
        output = self.decoder(lstm_out)
        return output


def train_model():
    # 加载数据
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MouseDataset('mouse_data_time_seq.csv')
    test_dataset = MouseDataset('mouse_data_time_seq_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = MouseTrajectoryModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练循环
    num_epochs = 1000
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for end_points, sequences in train_loader:
            optimizer.zero_grad()
            outputs = model(end_points)
            loss = criterion(outputs, sequences)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for end_points, sequences in test_loader:
                outputs = model(end_points)
                loss = criterion(outputs, sequences)
                val_loss += loss.item()

        # 打印训练状态
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {avg_val_loss:.6f}')

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_mouse_model.pth')

        # 导出ONNX模型
        model.eval()
        dummy_input = torch.randn(1, 3)
        torch.onnx.export(model, dummy_input, 'mouse_model.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})


if __name__ == '__main__':
    train_model()