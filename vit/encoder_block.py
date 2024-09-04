import torch
from torch import nn

from vit.mlp_block import MLPBlock


class EncoderBlock(nn.Module):
    """Encoder block used in the Vision Transformer encoder"""

    def __init__(
        self, num_heads: int, hidden_dim: int, embed_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # MSA
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)  # What is hidden dim from?
        self.msa = nn.MultiheadAttention(
            num_heads=self.num_heads, embed_dim=self.embed_dim
        )
        self.dropout = nn.Dropout(dropout)

        # MLP
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.mlp = MLPBlock(self.hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # MSA
        x = self.layer_norm1(input)
        x = self.msa(x)
        x = self.dropout(x)
        x = x + input

        # MLP
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y
