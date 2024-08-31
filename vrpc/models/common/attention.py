from collections import OrderedDict
from functools import partial
import math
from torch.nn import functional as F
import torch.nn as nn
import torch

from .mlp import MLPBlock, AttMLPBlock


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scaling_factor = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

    def _reshape_to_batches(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _reshape_from_batches(self, x: torch.Tensor):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ):
        query = self._reshape_to_batches(self.q_proj(query))
        key = self._reshape_to_batches(self.k_proj(key))
        value = self._reshape_to_batches(self.v_proj(value))

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, value)
        output = self._reshape_from_batches(output)
        return self.out_proj(output)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = ScaledDotProductAttention(
            hidden_dim, num_heads, attention_dropout
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = AttMLPBlock(hidden_dim, mlp_dim, hidden_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduce_ratio=16, dropout: float = 0.0):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = MLPBlock(
            in_channels, in_channels // reduce_ratio, in_channels, dropout
        )

    def _sigmoid(self, x):
        return 2 / (1 + torch.exp(-x))

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.max_pool(x).view(b, c)
        out = self.fc(out)
        out = self._sigmoid(out)
        out = x * out.view(b, c, 1, 1)
        return out


class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log2(channels) / gamma) + b / gamma))
        k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(24)
    input = torch.randn(50, 49, 512)
    sa = ScaledDotProductAttention(512, 8)
    output = sa(input, input, input)
    print(output)
