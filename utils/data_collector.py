import json
import numpy as np

class DataCollector:
    def __init__(self):
        self.data = []

    def collect(self, state, reward):
        # 将 state 中的所有 NumPy 数组转换为列表
        state_converted = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        self.data.append((state_converted, reward))

    def save_data(self, file_path):
        # 读取原文件内容
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        # 添加新数据
        data.append(self.data)

        # 将更新后的数据写回文件
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def reset(self):
        self.data = []
