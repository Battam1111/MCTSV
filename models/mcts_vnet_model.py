import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(EnhancedResidualBlock, self).__init__()
        self.adjust_dim = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        ) if input_dim != output_dim else nn.Identity()

        self.norm1 = nn.LayerNorm(output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = self.adjust_dim(x)  # 调整维度后的结果应用于残差连接
        x = self.adjust_dim(x)  # 这是关键改动：确保调整维度后的x被用于后续的LayerNorm和线性层
        out = F.relu(self.norm1(x))
        out = self.dropout1(out)
        out = self.linear1(out)  # 根据错误栈，此处应首先应用线性变换再应用ReLU
        out = F.relu(self.norm2(out))
        out = self.dropout2(out)
        out = self.linear2(out)  # 同上，调整顺序
        out += residual
        return F.relu(out)


class AttentionMechanism(nn.Module):
    def __init__(self, dim):
        super(AttentionMechanism, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)

class MCTSVNet(nn.Module):
    def __init__(self, config):
        super(MCTSVNet, self).__init__()
        self.config = config
        dim_observation = config["environment_size"] ** 2
        dim_action = config["available_actions"]
        hidden_dims = config["hidden_layers"]
        dropout_rate = config["dropout"]

        input_dim = dim_observation + (2*config["perception_range"]+1)**2
        
        self.encoder = nn.Sequential(
            EnhancedResidualBlock(input_dim, hidden_dims[0], dropout_rate),
            AttentionMechanism(hidden_dims[0]),
            *[nn.Sequential(
                EnhancedResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_rate),
                AttentionMechanism(hidden_dims[i+1])
              ) for i in range(len(hidden_dims)-1)]
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], dim_action),
            nn.Tanh()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, global_flow_output, local_flow_output):
        combined_input = torch.cat((
            global_flow_output.view(global_flow_output.size(0), -1), 
            local_flow_output.view(local_flow_output.size(0), -1)
        ), dim=1)
        x = self.encoder(combined_input)
        policy = F.softmax(self.actor_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value
