import math

import torch
import torch.nn as nn


class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(AdaptivePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # generate position encoding dynamically based on sequence length
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)

        return self.dropout(x + pe)
