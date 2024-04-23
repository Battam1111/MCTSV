import json
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import yaml

class GlobalFlowModel(nn.Module):
    def __init__(self, config):
        super(GlobalFlowModel, self).__init__()
        # 配置模型参数
        self.config = config
        self.input_linear = nn.Linear(config['environment_size'] * config['environment_size'], config['hidden_dim'])
        
        # 初始化 Transformer 编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['transformer_heads'],
            dim_feedforward=config['hidden_dim'] * 4,  # 一般设置为隐藏层维度的4倍
            dropout=config["dropout"],  # dropout 比率
            batch_first=True  # 重要：保证输入的第一个维度是批次大小
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config['transformer_layers'])
        
        # 输出层调整，以匹配环境矩阵的形状
        self.output_layer = nn.Linear(config['hidden_dim'], config['environment_size'] * config['environment_size'])
        
        # 如果需要，添加激活函数
        self.activation = nn.ReLU()

    def forward(self, x):
        # 输入 x 的形状应为 (batch_size, seq_length, feature_size)
        x = self.input_linear(x.view(1, -1))
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        # 调整输出形状以匹配环境矩阵
        x = x.view(-1, self.config['environment_size'], self.config['environment_size'])
        # 根据需要应用激活函数
        x = self.activation(x)
        return x

if __name__ == '__main__':
    # 读取配置文件
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # 初始化模型
    global_flow_model = GlobalFlowModel(config['global_flow_model'])

    # 示例：打印模型结构
    print(global_flow_model)
