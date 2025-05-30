import math
from typing import Literal

import torch
import torch.nn as nn

from models.ArithmeticTransformer.Decoder.DecoderLayer import DecoderLayer
from models.ArithmeticTransformer.FactoryFunctions.get_positional_encoding import get_positional_encoding


class Decoder(nn.Module):
    """the transformer decoder.

    consists of an embedding layer, positional encoding, and multiple decoder layers.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_length: int,
                 dropout: float = 0.1, pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
        """initialize the decoder.

        :arg vocab_size: size of the vocabulary
        :arg d_model: dimensionality of the embeddings
        :arg num_layers: number of decoder layers
        :arg num_heads: number of attention heads
        :arg d_ff: dimensionality of the feed-forward network
        :arg max_seq_length: maximum sequence length
        :arg dropout: dropout probability
        :arg pos_encoding_type: type of positional encoding ('standard' or 'adaptive')
        """
        super(Decoder, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # positional encoding
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, max_seq_length, dropout)

        # stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # layer normalization
        self.norm = nn.LayerNorm(d_model)

        # output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None,
                memory_padding_mask: torch.Tensor = None):
        """process input through the decoder.

        :arg tgt: Input tensor [batch_size, tgt_len]
        :arg memory: Encoder output [batch_size, src_len, d_model]
        :arg tgt_mask: mask for self-attention (optional)
        :arg memory_mask: mask for cross-attention (optional)
        :arg tgt_padding_mask: mask for padding tokens in target (optional)
        :arg memory_padding_mask: mask for padding tokens in memory (optional)

        returns output tensor of vocabulary size logits
        """
        # apply embedding and positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoder(tgt)

        # process through each decoder layer
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)

        # apply final normalization
        tgt = self.norm(tgt)

        # project to vocabulary size
        output = self.output_projection(tgt)

        return output
