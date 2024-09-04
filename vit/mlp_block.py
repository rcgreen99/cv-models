import torch
from torch import nn


class MLPBlock(nn.Module):
    """Multi-layer perceptron block used in the Vision Transformer encoder block"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear1(input)
        x = nn.GELU(x)
        x = self.linear2(x)
        x = nn.Dropout(self.dropout)
        return x
