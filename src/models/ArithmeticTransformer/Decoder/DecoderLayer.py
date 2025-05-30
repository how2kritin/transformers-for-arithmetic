import torch

import torch.nn as nn


class DecoderLayer(nn.Module):
    """unitary transformer decoder layer.

    has masked multi-head self-attention, multi-head cross-attention with encoder output,
    and a position-wise feed-forward network.
    each sub-layer has a residual connection and is followed by layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """initialize the decoder layer.

        :arg d_model: dimensionality of the embeddings
        :arg num_heads: number of attention heads
        :arg d_ff: dimensionality of the feed-forward network
        :arg dropout: dropout probability
        """
        super(DecoderLayer, self).__init__()

        # multi-head self-attention (masked)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # multi-head cross-attention with encoder output
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # feed-forward network
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_ff, d_model))

        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None,
                memory_padding_mask: torch.Tensor = None):
        """process input through the decoder layer.

        :arg tgt: input tensor [batch_size, tgt_len, d_model]
        :arg memory: encoder output [batch_size, src_len, d_model]
        :arg tgt_mask: mask for self-attention (optional)
        :arg memory_mask: mask for cross-attention (optional)
        :arg tgt_padding_mask: mask for padding tokens in target (optional)
        :arg memory_padding_mask: mask for padding tokens in memory (optional)

        returns output tensor after self-attention, cross-attention, and feed-forward
        """
        # self-attention block with residual connection and layer norm
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        tgt = tgt + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        # cross-attention block with residual connection and layer norm
        cross_attn_output, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                               key_padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt
