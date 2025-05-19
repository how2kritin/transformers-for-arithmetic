import math
from typing import Literal

import torch
import torch.nn as nn

from models.Encoder.EncoderLayer import EncoderLayer
from models.FactoryFunctions.get_positional_encoding import get_positional_encoding


class Encoder(nn.Module):
    """the transformer encoder.

    consists of an embedding layer, positional encoding, and multiple encoder layers.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_length: int,
                 dropout: float = 0.1, pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
        """initialize the encoder.

        :arg vocab_size: size of the vocabulary
        :arg d_model: dimensionality of the embeddings
        :arg num_layers: number of encoder layers
        :arg num_heads: number of attention heads
        :arg d_ff: dimensionality of the feed-forward network
        :arg max_seq_length: maximum sequence length
        :arg dropout: dropout probability
        :arg pos_encoding_type: type of positional encoding ('standard' or 'adaptive')
        """
        super(Encoder, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # positional encoding
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, max_seq_length, dropout)

        # stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, padding_mask: torch.Tensor = None):
        """process input through the encoder.

        :arg src: input tensor [batch_size, src_len]
        :arg src_mask: mask for self-attention (optional)
        :arg padding_mask: mask for padding tokens (optional)

        returns encoder output tensor
        """
        # apply embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)

        # process through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask, padding_mask)

        # apply final normalization
        src = self.norm(src)

        return src
