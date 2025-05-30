import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """positional encoding for the transformer model.

    adds positional information to the input embeddings to provide
    sequence order information to the self-attention mechanism.
    """

    def __init__(self, d_model: int, max_seq_length: int = 64, dropout: float = 0.1):
        """initialize the positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension and register as buffer (persistent state)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """add positional encoding to the input embeddings.
        :arg x: input embeddings [batch_size, seq_length, d_model]

        returns positional embeddings
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
