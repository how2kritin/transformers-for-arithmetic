from typing import Literal

import torch
import torch.nn as nn

from models.Decoder.Decoder import Decoder
from models.Encoder.Encoder import Encoder


class ArithmeticTransformer(nn.Module):
    """transformer model for arithmetic operations.
    an encoder-decoder transformer model for sequence-to-sequence arithmetic tasks.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 num_heads: int = 8, d_ff: int = 512,
                 max_seq_length: int = 64, dropout: float = 0.1,
                 pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
        """initialize the transformer model.

        :arg vocab_size: size of the vocabulary
        :arg d_model: dimensionality of the embeddings
        :arg num_encoder_layers: number of encoder layers
        :arg num_decoder_layers: number of decoder layers
        :arg num_heads: number of attention heads
        :arg d_ff: dimensionality of the feed-forward network
        :arg max_seq_length: maximum sequence length
        :arg dropout: dropout probability
        :arg pos_encoding_type: type of positional encoding ('standard' or 'adaptive')
        """
        super(ArithmeticTransformer, self).__init__()

        # encoder
        self.encoder = Encoder(vocab_size, d_model, num_encoder_layers, num_heads, d_ff, max_seq_length, dropout,
                               pos_encoding_type)

        # decoder
        self.decoder = Decoder(vocab_size, d_model, num_decoder_layers, num_heads, d_ff, max_seq_length, dropout,
                               pos_encoding_type)

        self._init_parameters()

    def _init_parameters(self):
        """initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        """generate a square mask for the sequence to mask future positions.

        :arg sz: the size of the square mask

        returns square mask of size [sz, sz]
        """
        # create a mask that prevents attending to future positions
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def create_padding_mask(src: torch.Tensor, pad_idx: int = 0):
        """create a mask for padding tokens.

        :arg src: input tensor [batch_size, seq_len]
        :arg pad_idx: index of the padding token

        returns a padding mask of size [batch_size, seq_len]
        """
        # mark padding positions with True
        return src == pad_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None):
        """forward pass through the transformer model.

        :arg src: source sequence [batch_size, src_len]
        :arg tgt: target sequence [batch_size, tgt_len]
        :arg src_padding_mask: mask for padding tokens in source (optional)
        :arg tgt_padding_mask: mask for padding tokens in target (optional)

        returns the output logits [batch_size, tgt_len, vocab_size]
        """
        # create padding masks if not provided
        if src_padding_mask is None:
            src_padding_mask = self.create_padding_mask(src)
        if tgt_padding_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)

        # create causal mask for decoder self-attention
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # encode source sequence
        memory = self.encoder(src, padding_mask=src_padding_mask)

        # decode target sequence
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                              memory_padding_mask=src_padding_mask)

        return output
