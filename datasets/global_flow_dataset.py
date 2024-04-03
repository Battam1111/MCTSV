import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os

class GlobalFlowDataset(Dataset):
    def __init__(self, data_file, model_config):
        """
        初始化GlobalFlowDataset

        :param data_file: 包含所有环境信息和价值评估矩阵的JSON文件。
        :param model_config: 模型配置字典。
        """
        self.environment_matrices = []
        self.value_matrices = []
        self.model_config = model_config  # 添加模型配置属性
        
        # 加载数据文件
        with open(data_file, 'r') as file:
            data = json.load(file)
        
        # 提取环境矩阵和价值矩阵
        for key in sorted(data.keys()):
            environment_matrix = np.array(data[key]['environment'])
            value_matrix = np.array(data[key]['value_matrix'])
            self.environment_matrices.append(environment_matrix)
            self.value_matrices.append(value_matrix)

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.environment_matrices)

    def __getitem__(self, idx):
        environment_matrix = self.environment_matrices[idx]
        value_matrix = self.value_matrices[idx]
        
        environment_tensor = torch.tensor(environment_matrix, dtype=torch.float).flatten()
        value_tensor = torch.tensor(value_matrix, dtype=torch.float)

        return environment_tensor, value_tensor

# 使用示例
if __name__ == "__main__":
    # 假设已经加载了模型配置
    model_config = {'hidden_dim': 10}  # 示例配置，根据实际情况调整

    dataset = GlobalFlowDataset("data/processed/all_environments.json", model_config)
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        environment, value = dataset[i]
        print(f"Sample {i}: Environment shape: {environment.shape}, Value shape: {value.shape}")
