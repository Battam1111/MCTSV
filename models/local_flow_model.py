import math
import torch
import torch.nn as nn
import yaml
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys

class LocalFlowTransformer(nn.Module):
    def __init__(self, config):
        super(LocalFlowTransformer, self).__init__()
        self.model_dim = config["model_dim"] * 2
        self.perception_range = config["perception_range"]
        self.output_size = (self.perception_range * 2 + 1) ** 2  # 计算价值矩阵的元素数量

        encoder_layer = TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=config["num_heads"],
            dim_feedforward=config["feedforward_dim"] * 2,
            dropout=config["dropout"],
            batch_first=True # 重要：保证输入的第一个维度是批次大小
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config["num_encoder_layers"])
        self.src_tok_emb = nn.Linear(self.perception_range * 2 + 1, self.model_dim)
        self.positional_encoding = PositionalEncoding(self.model_dim, dropout=config["dropout"])

        # 输出层，用于预测价值矩阵
        self.output_layer = nn.Linear((self.perception_range * 2 + 1) * self.model_dim, self.output_size)

    def forward(self, src):
        # print(src.shape)
        # import pdb; pdb.set_trace()
        src = self.src_tok_emb(src)
        
        # print(src.shape)
        
        src = src.view(-1, self.perception_range * 2 + 1, self.model_dim)
        # print(src.shape)
        # sys.exit(0)

        src = self.positional_encoding(src)
        
        output = self.transformer_encoder(src)
        # print(output.shape)
        output = output.view(output.size(0), -1)  # Now shape should be [batch_size, 5*25]
        # print(output.shape)
        output = self.output_layer(output)
        
        # 确保重新塑形前后元素数量一致
        output = output.view(-1, self.perception_range * 2 + 1, self.perception_range * 2 + 1)  # 应该自动匹配
        # print(output.shape)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe[:max_len, :])

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)
        return self.dropout(x)

if __name__ == '__main__':
    with open('config/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    local_flow_model = LocalFlowTransformer(config['models']['local_flow_model'])
    print(local_flow_model)
