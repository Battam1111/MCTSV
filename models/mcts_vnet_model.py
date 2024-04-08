import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.norm1(self.linear1(x)))
        out = self.dropout(out)
        out = self.norm2(self.linear2(out))
        out += residual
        out = F.relu(out)
        return out

class MCTSVNet(nn.Module):
    def __init__(self, config):
        super(MCTSVNet, self).__init__()
        self.config = config
        dim_observation = config["environment_size"] ** 2
        dim_action = config["available_actions"]
        hidden_dims = config["hidden_layers"]
        dropout_rate = config["dropout"]

        # 残差块处理全局观察和局部观察的组合
        input_dim = dim_observation + (2*config["perception_range"]+1)**2
        self.encoder = nn.Sequential(
            ResidualBlock(input_dim, hidden_dims[0], dropout_rate),
            *[ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_rate) for i in range(len(hidden_dims)-1)]
        )
        
        # 动作生成头
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], dim_action),
            nn.Tanh()
        )
        
        # 价值评估头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, global_flow_output, local_flow_output):
        # 合并全局观察和局部观察
        combined_input = torch.cat((global_flow_output.view(global_flow_output.size(0), -1), 
                                     local_flow_output.view(local_flow_output.size(0), -1)), dim=1)

        # 通过残差块编码器
        x = self.encoder(combined_input)

        # 生成动作和价值
        policy = F.softmax(self.actor_head(x), dim=1)
        value = self.value_head(x)

        return policy, value

if __name__ == '__main__':
    # 示例配置参数
    config = {
        "input_dim": 1024,  # 输入层维度
        "hidden_layers": [512, 256, 128],  # 隐藏层维度列表
        "dropout": 0.5,  # Dropout 率
        "available_actions": 10  # 可选动作数量
    }

    # 初始化模型
    mcts_vnet_model = MCTSVNet(config)

    print(mcts_vnet_model)
