import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

class LocalFlowDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            all_data = json.load(file)
            for env_key, env_data in all_data.items():
                for pos_key, pos_data in env_data.items():
                    if 'local_data' not in pos_data:
                        print(f"Missing 'local_data' in {pos_key}")
                        continue
                    local_data = pos_data['local_data']
                    value_matrix = pos_data['local_value_matrix']
                    self.data.append({
                        'signals': np.array(local_data['signals']),
                        'obstacles': np.array(local_data['obstacles']),
                        'value_matrix': np.array(value_matrix)
                    })

    def __len__(self):
        # 数据集长度为所有加载的局部感知数据项的数量
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引为idx的数据项。

        :param idx: 数据项的索引。
        :return: 一个字典，包含局部信号点、障碍物和价值矩阵。
        """
        data_item = self.data[idx]
        # print(data_item)
        return {
            'signals': torch.tensor(data_item['signals'], dtype=torch.float32),
            'obstacles': torch.tensor(data_item['obstacles'], dtype=torch.float32),
            'value_matrix': torch.tensor(data_item['value_matrix'], dtype=torch.float32)
        }

def custom_collate_fn(batch):
    signals = torch.stack([item['signals'] for item in batch])
    obstacles = torch.stack([item['obstacles'] for item in batch])
    value_matrices = torch.stack([item['value_matrix'] for item in batch])
    return signals, obstacles, value_matrices

# if __name__ == "__main__":
#     file_path = 'data/processed/all_local_data.json'

#     dataset = LocalFlowDataset(file_path)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

#     for signals, obstacles, value_matrices in dataloader:
#         print(signals.shape, obstacles.shape, value_matrices.shape)

if __name__ == "__main__":
    file_path = 'data/processed/all_local_data.json'
    dataset = LocalFlowDataset(file_path)
    print(len(dataset))
    # 尝试直接从Dataset中获取单个数据项
    sample = dataset[0]
    print(sample['signals'].shape, sample['obstacles'].shape, sample['value_matrix'].shape)
