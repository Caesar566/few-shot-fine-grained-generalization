"""Matching Networks full-context embeddings (Vinyals et al., Appendix A)."""

import torch
from torch import nn


class FullContextEmbedding(nn.Module):
    """Support BiLSTM residuals and independent query attention LSTM reads.

    The LSTMCell input concatenates query and read vectors. Its affine gates are
    equivalent to the paper's concatenation of previous hidden and read vectors:
    both compute independent linear maps of query, hidden and support readout.
    Query hidden states start at the CNN features; cell states start at zero.
    """

    def __init__(self, dimension: int = 512, steps: int = 5):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("fce_steps 必须是正整数")
        self.dimension = dimension
        self.steps = steps
        self.support_lstm = nn.LSTM(dimension, dimension, bidirectional=True, batch_first=True)
        self.query_cell = nn.LSTMCell(2 * dimension, dimension)

    def forward(
        self, support: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (support.ndim != 2 or query.ndim != 2 or support.shape[-1] != self.dimension
                or query.shape[-1] != self.dimension or support.shape[0] == 0):
            raise ValueError(f"full-context embedding 需要非空 support 和 {self.dimension} 维特征")
        sequence, _ = self.support_lstm(support.unsqueeze(0))
        forward, backward = sequence.squeeze(0).chunk(2, dim=-1)
        contextual_support = support + forward + backward
        hidden = query
        cell = torch.zeros_like(query)
        for _ in range(self.steps):
            attention = (hidden @ contextual_support.T).softmax(dim=-1)
            read = attention @ contextual_support
            hidden, cell = self.query_cell(torch.cat((query, read), dim=-1), (hidden, cell))
            hidden = hidden + query
        return contextual_support, hidden
