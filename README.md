# Machine Learning Mouse Path

mlmp (Machine Learning Mouse Path) 是一个用于鼠标路径控制和训练的 Python 包，使用mlmp推理出的鼠标路径移动鼠标，让其看起来更像人类。它提供了鼠标路径预测功能，并支持通过数据收集和训练来自定义鼠标行为

## 功能

- **鼠标路径预测**：基于预训练模型或自定义模型文件，预测鼠标路径。
- **数据收集**：收集鼠标移动数据，用于训练自定义模型。
- **训练模型**：使用收集的数据训练鼠标行为模型。
- **时间序列数据支持**：支持时间序列数据的收集和训练（需启用开发功能）。<BR> **正处于开发阶段，功能随时可能发生变化** <BR> <span style="color: rgb(255, 169, 0);">由于没有足够的时间，所以如果真的希望使用这个功能，可以自己收集足够的数据之后测试，个人测试，同样200组数据不能达到需求的效果，建议增加数据量至数千组或以上进行训练</span>

## 安装

通过 `pip` 安装 mlmp：

```bash
pip install mlmp
```

## 使用示例

### 鼠标路径预测

```python
from mlmp import mouse

# 初始化鼠标对象
m = mouse()

# 预测鼠标路径
start = (0, 0)
end = (100, 100)
result = m.mouse_output(start, end)
print(result)

# 输出示例
[(0, 0), (6, 5), (11, 17), (17, 19), (21, 20), (14, 11), (11, 11), (5, 4), (4, 3), (3, 5), (8, 5)]
```

### 数据收集和训练

```python
from mlmp import TrainBySelf

# 初始化训练器
trainer = TrainBySelf()

# 收集数据
trainer.collect_data()
# 训练数据是点击红色球后再点击一次蓝色球，视为一次数据收集，每轮最多记录100次
# 建议收集5轮次及以上的数据后，在开始训练


# 训练模型
# 将收集好的数据（mouse_data.csv）抽出一部分，放入（mouse_data_test.csv）
# 或重新收集1-2轮次，保证两个文件可访问
trainer.train()
# 训练完成后，会生成一个onnx模型文件
```

### 时间序列功能（开发功能）

```python
from mlmp import TrainBySelf

# 初始化训练器并启用开发功能
trainer = TrainBySelf()
trainer.dev_features = True

# 收集时间序列数据
trainer.time_seq_mouse_data()
# 与之前方式相似，名字改为 mouse_data_time_seq.csv
# 训练集名字改为 mouse_data_time_seq_test.csv

# 训练时间序列模型
trainer.time_seq_train()

### 补充时间序列预测的推理功能
import mlmp
mouse_path = mlmp.dev_mouse(model_path = "你的模型位置", dll=False)
start = (0, 0)
end = (200, 100)
print(mouse_path.mouse_output(start, end, absulute=False))
```

## 开发功能

时间序列数据收集和训练功能默认为关闭状态。如果需要使用，请在初始化 `TrainBySelf` 时启用开发功能：

```python
trainer = TrainBySelf()
trainer.dev_features = True
```

## 依赖

- `onnxruntime`：用于模型推理。

## 许可证

本项目基于 MIT 许可证发布。详情请参阅 [LICENSE](LICENSE) 文件。

## 项目主页

访问 [GitHub 仓库](https://github.com/zaixia108/MLMousePath) 获取最新版本和源代码。

## 致谢
本项目的部分代码借鉴了 [suixin1424/mouse_control](https://github.com/suixin1424/mouse_control) 的实现