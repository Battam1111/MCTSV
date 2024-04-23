import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, device='cuda:0'):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.device = device

        # 创建网格
        y_position = torch.arange(height, device=device).unsqueeze(1).unsqueeze(2).repeat(1, width, 1)  # [height, width, 1]
        x_position = torch.arange(width, device=device).unsqueeze(0).unsqueeze(2).repeat(height, 1, 1)  # [height, width, 1]

        # 维度划分
        half_dim = d_model // 4
        div_term = torch.exp(torch.arange(0., half_dim, 2, device=device) * -(math.log(10000.0) / half_dim))  # [half_dim/2]

        # 计算位置编码
        pe = torch.zeros(height, width, d_model, device=device)
        pe[:, :, 0:half_dim:2] = torch.sin(x_position * div_term)
        pe[:, :, 1:half_dim:2] = torch.cos(x_position * div_term)
        pe[:, :, half_dim:2*half_dim:2] = torch.sin(y_position * div_term)
        pe[:, :, half_dim+1:2*half_dim:2] = torch.cos(y_position * div_term)

        pe = pe.permute(2, 0, 1).unsqueeze(0)  # Permute to make it [1, d_model, height, width]
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, d_model, height, width]
        """
        x = x + self.pe  # Add positional encoding to the input
        return x

class CombinedNormActivation(nn.Module):
    def __init__(self, channels, negative_slope=0.01, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.negative_slope = negative_slope
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.instance_norm = nn.InstanceNorm2d(channels, eps=self.eps, affine=True)

    def forward(self, x):
        x = self.instance_norm(x)
        return self.leaky_relu(x)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, expansion=4, dropout_rate=0.1, norm_activation=None):
        super().__init__()
        mid_channels = int(output_dim * expansion)
        self.adjust_dim = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))

        # 确保 norm_activation 接收正确的 channels 参数
        if norm_activation is None:
            norm_activation = CombinedNormActivation(mid_channels)  # 这里使用 mid_channels 作为 channels

        self.layers = nn.Sequential(
            norm_activation,
            nn.Conv2d(output_dim, mid_channels, kernel_size=(3, 3), padding=1),
            norm_activation,
            nn.Conv2d(mid_channels, output_dim, kernel_size=(3, 3), padding=1),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        residual = self.adjust_dim(x)
        out = self.layers(residual)
        out += residual
        return out

class DynamicSparseAttention(nn.Module):
    def __init__(self, num_heads=4, sparsity=0.1, device='cuda:0'):
        super(DynamicSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.device = device
        self.scale = None
        self.query = nn.ModuleDict()
        self.key = nn.ModuleDict()
        self.value = nn.ModuleDict()
        self.out = nn.ModuleDict()
        self.distill = nn.Linear(num_heads * num_heads, num_heads * num_heads)  # 蒸馏层
        self.gate = nn.ModuleDict()  # 门控机制
        self.gate_projection = nn.ModuleDict()  # 门控映射层

    def _init_layers(self, dim):
        self.dim = dim
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        dim_key = str(dim)
        if dim_key not in self.query:
            self.query[dim_key] = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)
            self.key[dim_key] = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)
            self.value[dim_key] = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)
            self.out[dim_key] = nn.Linear(self.head_dim * self.num_heads, dim).to(self.device)
            self.gate[dim_key] = nn.Sequential(
                nn.Linear(dim, self.num_heads * self.num_heads),  # 根据输入维度调整门控参数的大小
                nn.Sigmoid()
            ).to(self.device)
            self.gate_projection[dim_key] = nn.Linear(self.num_heads * self.num_heads, self.head_dim * self.num_heads).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = x.reshape(x.size(0), x.size(1), -1)
        batch_size, seq_length, dim = x.size()
        self._init_layers(dim)
        dim_key = str(dim)
        Q = self.query[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        k = max(int(attention_scores.size(-1) * self.sparsity), 1)
        threshold = torch.topk(attention_scores.view(-1, attention_scores.size(-1)), k=k, dim=-1)[0][..., -1, None]
        print(threshold.shape)
        threshold = threshold.view(batch_size, self.num_heads, 1, 1)
        mask = attention_scores >= threshold
        masked_attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_probs = F.softmax(masked_attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)
        distilled_attention_output = self.distill(attention_output.view(-1, self.num_heads * self.num_heads))
        distilled_attention_output = distilled_attention_output.view(batch_size, -1, self.head_dim * self.num_heads)
        gate = self.gate[dim_key](x.mean(dim=1))  # 使用输入的平均值来计算门控参数
        gate = self.gate_projection[dim_key](gate)  # 将门控参数映射到正确的维度
        gate = gate.unsqueeze(1).expand(-1, x.size(1), -1)  # 扩展gate的维度以匹配attention_output
        return self.out[dim_key](gate * distilled_attention_output + (1 - gate) * attention_output)

class GenericEncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_rate, expansion, sparsity, negative_slope, eps, shared_attention, shared_norm_activation):
        super().__init__()
        # Ensure the norm_activation is properly configured
        norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, negative_slope, eps)
        self.residual_block = EnhancedResidualBlock(input_dim, output_dim, expansion, dropout_rate, norm_activation)
        self.attention = shared_attention if shared_attention is not None else DynamicSparseAttention(num_heads=num_heads, sparsity=sparsity)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.attention(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads, dropout_rate, expansion, sparsity, negative_slope, eps, shared_attention, shared_norm_activation, positional_encoding):
        super().__init__()
        self.input_adjustment = nn.Conv2d(1, hidden_dims[0], kernel_size=(1, 1))
        self.positional_encoding = positional_encoding if positional_encoding is not None else PositionalEncoding2D(hidden_dims[0])

        self.layers = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            if shared_norm_activation is not None:
                shared_norm_activation = CombinedNormActivation(dim)  # 确保此处传入的是正确的 channels 数
            self.layers.append(GenericEncoderLayer(input_dim=in_dim, output_dim=dim, num_heads=num_heads, expansion=expansion, sparsity=sparsity, dropout_rate=dropout_rate, negative_slope=negative_slope, eps=eps, shared_attention=shared_attention, shared_norm_activation=shared_norm_activation))

    def forward(self, x):
        x = self.input_adjustment(x)  # x现在应该是[batch_size, channels, height, width]
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout_rate=0.1, eps=1e-5, sparsity=0.1, shared_attention=None, shared_norm_activation=None):
        super().__init__()
        self.dynamic_attention = shared_attention if shared_attention is not None else DynamicSparseAttention(num_heads=num_heads, sparsity=sparsity)
        self.combined_norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, eps=eps)
        self.proj_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.final_layer = nn.Linear(output_dim, 1)
    
    def forward(self, x):
        x = self.dynamic_attention(x)
        x = self.proj_layer(x)
        x = self.combined_norm_activation(x)
        value = self.final_layer(x)
        return x, value

class MCTSVNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        global_height, global_width = config["environment_size"], config["environment_size"]
        local_height, local_width = 2*config["perception_range"]+1, 2*config["perception_range"]+1

        positional_encoding_global = PositionalEncoding2D(config["hidden_layers"][0], global_height, global_width)
        positional_encoding_local = PositionalEncoding2D(config["hidden_layers"][0], local_height, local_width)

        # 设置共享层
        if config["shared_attention"]:
            shared_attention = DynamicSparseAttention(num_heads=config["num_heads"], sparsity=config["sparsity"])
        else:
            shared_attention = None
        if config["shared_norm_activation"]:
            shared_norm_activation = CombinedNormActivation(config["hidden_layers"][0], config["negative_slope"], config["eps"])
        else:
            shared_norm_activation = None

        # 网络结构
        self.encoder_global = EncoderBlock(
            1,
            config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention=shared_attention,
            shared_norm_activation=shared_norm_activation,
            positional_encoding=positional_encoding_global
        )
        self.encoder_local = EncoderBlock(
            1,
            config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention=shared_attention,
            shared_norm_activation=shared_norm_activation,
            positional_encoding=positional_encoding_local
        )
        self.decoder = AttentionDecoder(
            config["hidden_layers"][-1] * 2 * config["environment_size"] * config["environment_size"],  # 假设解码器需要一维输入
            config["available_actions"], config["num_heads"],
            config["dropout"], config["eps"], config["sparsity"],
            shared_attention=shared_attention, shared_norm_activation=shared_norm_activation
        )

    def forward(self, global_flow_output, local_flow_output):
        # 编码全局和局部信息
        encoded_global = self.encoder_global(global_flow_output)
        encoded_local = self.encoder_local(local_flow_output)
        
        # 结合全局和局部编码输出，展平以适应解码器
        combined_output = torch.cat((encoded_global.flatten(1), encoded_local.flatten(1)), dim=1)
        
        # 解码处理合并后的输出，调整以接收两个输出：输出张量x和预测的价值
        output, value = self.decoder(combined_output)

        # 将解码器的输出张量处理成策略，使用softmax进行归一化
        policy = F.softmax(output, dim=1)
        
        # 返回策略和价值
        return policy, value
