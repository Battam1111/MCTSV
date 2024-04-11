import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CombinedNormActivation(nn.Module):
    def __init__(self, normalized_shape=None, negative_slope=0.01, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.negative_slope = negative_slope
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.layer_norm = None

    def forward(self, x):
        if self.layer_norm is None or self.layer_norm.normalized_shape[0] != x.size(-1):
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
    def __init__(self, num_heads=8, sparsity=0.1, device='cuda:0'):
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
        batch_size, seq_length, dim = x.size()
        self._init_layers(dim)
        dim_key = str(dim)
        Q = self.query[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value[dim_key](x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        k = max(int(attention_scores.size(-1) * self.sparsity), 1)
        threshold = torch.topk(attention_scores.view(-1, attention_scores.size(-1)), k=k, dim=-1)[0][..., -1, None]
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
    def __init__(self, input_dim, output_dim, expansion, dropout_rate, negative_slope, eps, shared_attention, shared_norm_activation):
        super().__init__()
        norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, negative_slope, eps)
        self.residual_block = EnhancedResidualBlock(input_dim, output_dim, expansion, dropout_rate, norm_activation)
        self.attention = shared_attention if shared_attention is not None else DynamicSparseAttention(output_dim)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.attention(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads, dropout_rate, expansion, sparsity, negative_slope, eps, shared_attention, shared_norm_activation, shared_positional_encoding):
        super().__init__()
        # 2D可能效果更好，但是为了泛用性，这里使用1D卷积（因为任何维度的数据都可以展平成1维向量？）
        self.input_adjustment = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=1) if input_dim != hidden_dims[0] else nn.Identity()
        self.positional_encoding = shared_positional_encoding if shared_positional_encoding is not None else PositionalEncoding(hidden_dims[0])
        self.layers = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            self.layers.append(GenericEncoderLayer(in_dim, dim, expansion, dropout_rate, negative_slope, eps, shared_attention, shared_norm_activation))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.input_adjustment(x)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout_rate=0.1, eps=1e-5, sparsity=0.1, shared_attention=None, shared_norm_activation=None):
        super().__init__()
        self.dynamic_attention = shared_attention if shared_attention is not None else DynamicSparseAttention(input_dim, num_heads=num_heads, sparsity=sparsity)
        self.combined_norm_activation = shared_norm_activation if shared_norm_activation is not None else CombinedNormActivation(output_dim, eps=eps)
        self.proj_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = self.dynamic_attention(x)
        x = self.proj_layer(x)
        x = self.combined_norm_activation(x)
        return x

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
            shared_positional_encoding = PositionalEncoding(config["hidden_layers"][0])
        else:
            shared_positional_encoding = None
        # 网络结构
        self.encoder_global = EncoderBlock(
            config["environment_size"] ** 2, config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention, shared_norm_activation,
            shared_positional_encoding
        )
        self.encoder_local = EncoderBlock(
            (2 * config["perception_range"] + 1) ** 2, config["hidden_layers"], config["num_heads"],
            config["dropout"], config["expansion"], config["sparsity"],
            config["negative_slope"], config["eps"], shared_attention, shared_norm_activation,
            shared_positional_encoding
        )
        self.decoder = AttentionDecoder(
            config["hidden_layers"][-1] * 2, config["available_actions"], config["num_heads"],
            config["dropout"], config["eps"], config["sparsity"], shared_attention, shared_norm_activation
        )
        self.value_head = nn.Linear(config["hidden_layers"][-1] * 2, 1)

    def forward(self, global_flow_output, local_flow_output):
        global_flow_output = global_flow_output.view(global_flow_output.size(0), -1)
        local_flow_output = local_flow_output.view(local_flow_output.size(0), -1)
        encoded_global = self.encoder_global(global_flow_output)
        encoded_local = self.encoder_local(local_flow_output)
        combined_output = torch.cat((encoded_global, encoded_local), dim=-1)
        policy = F.softmax(self.decoder(combined_output), dim=-1)
        value = self.value_head(combined_output)
        return policy, value