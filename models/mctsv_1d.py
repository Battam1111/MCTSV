import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1, max_len=10000, device='cuda:0'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        
        # 预先计算位置编码
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不计算梯度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # if x.size(1) != 25:
            # 根据批次大小动态重复位置编码
        x = x + self.pe[:, :x.size(1)]
        return x

class CombinedNormActivation(nn.Module):
    def __init__(self, normalized_shape=512, negative_slope=0.01, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.negative_slope = negative_slope
        # 预定义LayerNorm层，尽量避免在forward中重复创建
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=self.eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        # 确保layer_norm能够处理输入x的最后一个维度
        if self.layer_norm.normalized_shape[0] != x.size(-1):
            self.layer_norm = nn.LayerNorm(x.size(-1), eps=self.eps).to(x.device)
        x = self.layer_norm(x)
        return self.leaky_relu(x)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, expansion=4, dropout_rate=0.1, norm_activation=None):
        super().__init__()
        mid_dim = output_dim * expansion

        self.adjust_dim = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            norm_activation
        ) if input_dim != output_dim else nn.Identity()

        self.layers = nn.Sequential(
            norm_activation,
            nn.Linear(output_dim, mid_dim),
            norm_activation,
            nn.Linear(mid_dim, output_dim),
            nn.Dropout(dropout_rate)
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
        self.head_dim = None
        
        # 初始化共享的Q, K, V层但不在这里设置参数
        self.query = nn.Linear(1, 1).to(self.device)  # 临时占位，将在_forward_init中更新
        self.key = nn.Linear(1, 1).to(self.device)    # 临时占位，将在_forward_init中更新
        self.value = nn.Linear(1, 1).to(self.device)  # 临时占位，将在_forward_init中更新

        # 输出层和门控机制，维持动态
        self.out = nn.ModuleDict()
        self.distill = nn.Linear(num_heads * num_heads, num_heads * num_heads)  # 蒸馏层
        self.gate = nn.ModuleDict()  # 门控机制
        self.gate_projection = nn.ModuleDict()  # 门控映射层

    def _forward_init(self, dim):
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # 更新Q, K, V层的参数
        self.query = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)
        self.key = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)
        self.value = nn.Linear(dim, self.head_dim * self.num_heads).to(self.device)

        dim_key = str(dim)
        if dim_key not in self.out:
            self.out[dim_key] = nn.Linear(self.head_dim * self.num_heads, dim).to(self.device)
            self.gate[dim_key] = nn.Sequential(
                nn.Linear(dim, self.num_heads * self.num_heads),
                nn.Sigmoid()
            ).to(self.device)
            self.gate_projection[dim_key] = nn.Linear(self.num_heads * self.num_heads, self.head_dim * self.num_heads).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, seq_length, dim = x.size()
        self._forward_init(dim)
        dim_key = str(dim)
        Q = self.query(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 优化阈值计算方法：采用近似计算
        percentile_value = 100 - (self.sparsity * 100)
        threshold = torch.quantile(attention_scores, percentile_value / 100, dim=-1, keepdim=True)
        
        # 使用mask实现稀疏操作
        mask = attention_scores >= threshold
        masked_attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_probs = F.softmax(masked_attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)

        distilled_attention_output = self.distill(attention_output.view(-1, self.num_heads * self.num_heads))
        distilled_attention_output = distilled_attention_output.view(batch_size, -1, self.head_dim * self.num_heads)

        gate = self.gate[dim_key](x.mean(dim=1))
        gate = self.gate_projection[dim_key](gate)
        gate = gate.unsqueeze(1).expand(-1, x.size(1), -1)

        return self.out[dim_key](gate * distilled_attention_output + (1 - gate) * attention_output)


class GenericEncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_rate, expansion, sparsity, negative_slope, eps, shared_attention, shared_norm_activation):
        super().__init__()
        norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, negative_slope, eps)
        self.residual_block = EnhancedResidualBlock(input_dim, output_dim, expansion, dropout_rate, norm_activation)
        self.attention = shared_attention if shared_attention is not None else DynamicSparseAttention(num_heads=num_heads, sparsity=sparsity)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.attention(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads, dropout_rate, expansion, sparsity, negative_slope, eps, shared_attention, shared_norm_activation, shared_positional_encoding):
        super().__init__()
        # 2D可能效果更好，但是为了泛用性，这里使用1D卷积（因为任何维度的数据都可以展平成1维向量？），第一个输入为d_model，暂时硬编码为1
        self.input_adjustment = nn.Conv1d(1, hidden_dims[0], kernel_size=1) if input_dim != hidden_dims[0] else nn.Identity()
        self.positional_encoding = shared_positional_encoding if shared_positional_encoding is not None else PositionalEncoding(hidden_dims[0])
        self.layers = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            self.layers.append(GenericEncoderLayer(input_dim=in_dim, output_dim=dim, num_heads=num_heads, expansion=expansion, sparsity=sparsity, dropout_rate=dropout_rate, negative_slope=negative_slope, eps=eps, shared_attention=shared_attention, shared_norm_activation=shared_norm_activation))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)
        x = x.transpose(1, 2)
        x = self.input_adjustment(x)
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x

class MultiScaleAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=8, scales=[1, 2, 4], sparsity=0.1, shared_attention=None):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionPooling(input_dim, pool_size=scale, stride=scale) for scale in scales
        ])
        self.shared_attention = shared_attention if shared_attention is not None else DynamicSparseAttention(num_heads=num_heads, sparsity=sparsity)

    def forward(self, x):
        # Apply shared attention first, then pool at multiple scales and concatenate.
        x = self.shared_attention(x)
        pooled_outputs = [layer(x) for layer in self.layers]
        return torch.cat(pooled_outputs, dim=1)  # Concatenate along feature dimension

class AttentionPooling(nn.Module):
    def __init__(self, input_dim=32, pool_size=1, stride=1):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)  # 生成每个元素的权重
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        weights = torch.softmax(self.query(x), dim=1)  # 应用softmax获取归一化的权重
        # 根据pool_size和stride执行池化操作
        pooled = F.avg_pool1d(weights * x, kernel_size=self.pool_size, stride=self.stride)
        return torch.sum(pooled, dim=1)  # 加权求和

class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout_rate=0.1, eps=1e-5, scales=[1, 2, 4], shared_attention=None, shared_norm_activation=None):
        super().__init__()
        self.scale_factor = len(scales)  # Depending on how many scales we use
        scaled_input_dim = input_dim * self.scale_factor

        self.global_attention_pooling = MultiScaleAttentionPooling(input_dim, num_heads=num_heads, scales=scales, shared_attention=shared_attention)
        self.local_attention_pooling = MultiScaleAttentionPooling(input_dim, num_heads=num_heads, scales=scales, shared_attention=shared_attention)

        self.global_proj_layer = nn.Sequential(
            nn.Linear(scaled_input_dim, output_dim),  # Adjusted input dimension
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.local_proj_layer = nn.Sequential(
            nn.Linear(scaled_input_dim, output_dim),  # Adjusted input dimension
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.combined_norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, eps=eps)
        self.final_layer = nn.Linear(output_dim, 1)

        self.policy_network = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),  # 融合全局和局部信息
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

        self.value_network = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 1)
        )

    def forward(self, Eglobal, Elocal):
        Eglobal = self.global_attention_pooling(Eglobal)
        Elocal = self.local_attention_pooling(Elocal)

        Eglobal = self.global_proj_layer(Eglobal)
        Elocal = self.local_proj_layer(Elocal)

        Eglobal = self.combined_norm_activation(Eglobal)
        Elocal = self.combined_norm_activation(Elocal)

        combined_features = torch.cat([Eglobal, Elocal], dim=-1)
        policy = self.policy_network(combined_features)
        value = self.value_network(combined_features)

        policy = F.softmax(policy, dim=-1)
        return policy, value

class MCTSVNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 共享层设定
        if config["shared_attention"]:
            shared_attention = DynamicSparseAttention(num_heads=config["num_heads"], sparsity=config["sparsity"])
        else:
            shared_attention = None
        if config["shared_norm_activation"]:
            shared_norm_activation = CombinedNormActivation((config["hidden_layers"][0],), config["negative_slope"], config["eps"])
        else:
            shared_norm_activation = None
        if config["shared_positional_encoding"]:
            # shared_positional_encoding = PositionalEncoding(config["hidden_layers"][0])
            # 定制维度
            shared_positional_encoding = PositionalEncoding()
        else:
            shared_positional_encoding = None
        # 网络结构
        self.encoder_global = EncoderBlock(
            config["environment_size"] ** 2, config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention=shared_attention, shared_norm_activation=shared_norm_activation,
            shared_positional_encoding=shared_positional_encoding
        )
        self.encoder_local = EncoderBlock(
            (2 * config["perception_range"] + 1) ** 2, config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention=shared_attention, shared_norm_activation=shared_norm_activation,
            shared_positional_encoding=shared_positional_encoding
        )
        self.decoder = AttentionDecoder(
            input_dim=config["hidden_layers"][0], output_dim=config["available_actions"], num_heads=config["num_heads"],
            dropout_rate=config["dropout"], eps=config["eps"], scales=[1], shared_attention=shared_attention, shared_norm_activation=shared_norm_activation,
        )

    def forward(self, global_flow_output, local_flow_output):
        # 重新整形全局和局部输出以匹配网络输入要求
        global_flow_output = global_flow_output.view(global_flow_output.size(0), -1)
        local_flow_output = local_flow_output.view(local_flow_output.size(0), -1)
        
        # 编码全局和局部信息
        encoded_global = self.encoder_global(global_flow_output)
        encoded_local = self.encoder_local(local_flow_output)
        
        # 使用解码器处理合并后的输出
        # 注意调整以接收两个输出：输出张量x和预测的价值
        policy, value = self.decoder(encoded_global, encoded_local)
        
        # 返回策略和价值
        return policy, value