import torch
import torch.nn as nn
import torch.nn.functional as F

class MCTSVNet(nn.Module):
    def __init__(self, config):
        super(MCTSVNet, self).__init__()
        self.available_actions = config["available_actions"]
        
        # 假设 global_flow_output 和 local_flow_output 的大小分别是 [10, 10] 和 [5, 5]
        global_input_dim = config["environment_size"]**2  # 因为 [10, 10] 的矩阵被平铺后的大小
        local_input_dim = (2*config["perception_range"]+1)**2  # 因为 [5, 5] 的矩阵被平铺后的大小
        total_input_dim = global_input_dim + local_input_dim  # 融合后的总输入维度

        layers = []

        # 使用融合后的输入维度作为第一层的输入
        input_dim = total_input_dim
        hidden_layers = config["hidden_layers"]
        dropout_rate = config.get("dropout", 0.5)

        # 构建隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        
        self.policy_head = nn.Linear(input_dim, self.available_actions)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, global_flow_output, local_flow_output):
        global_flow_flat = global_flow_output.view(global_flow_output.size(0), -1)
        local_flow_flat = local_flow_output.view(local_flow_output.size(0), -1)
        combined_input = torch.cat((global_flow_flat, local_flow_flat), dim=1)

        # 在将combined_input传递给网络之前，增加一个批维度
        # combined_input = combined_input.unsqueeze(0)  # 这使combined_input成为2D输入

        x = self.shared_layers(combined_input)

        policy = F.softmax(self.policy_head(x), dim=1)  # 注意调整softmax的dim参数为1
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
