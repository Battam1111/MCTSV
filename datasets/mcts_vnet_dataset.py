import torch
from torch.utils.data import Dataset

class MCTSVNetDataset(Dataset):
    def __init__(self, data_file):
        """
        初始化 MCTSVNetDataset。

        :param data_file: 包含训练数据的文件路径，数据应该是一个列表，其中每个元素是一个字典，包含
                          'global_value_matrix', 'local_value_matrix', 'battery' 和 'reward' 四个键。
        """
        self.data = torch.load(data_file)

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引为 idx 的样本。

        :param idx: 样本的索引。
        :return: 一个包含全局价值矩阵、局部价值矩阵和电量信息的元组，以及对应的奖励。
        """
        sample = self.data[idx]
        global_value_matrix = torch.tensor(sample['global_value_matrix'], dtype=torch.float32)
        local_value_matrix = torch.tensor(sample['local_value_matrix'], dtype=torch.float32)
        battery = torch.tensor([sample['battery']], dtype=torch.float32)
        reward = torch.tensor([sample['reward']], dtype=torch.float32)

        return (global_value_matrix, local_value_matrix, battery), reward
