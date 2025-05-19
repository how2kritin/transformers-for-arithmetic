import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """unitary transformer encoder layer.

    has multi-head self-attention, followed by a position-wise feed-forward network.
    each sub-layer has a residual connection (to smoothen gradient flows) and is followed by layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """initialize the encoder layer.

        :arg d_model: the dimensionality of the embeddings
        :arg num_heads: number of attention heads
        :arg d_ff: dimensionality of the feed-forward network
        :arg dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()

        # multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # feed-forward network
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_ff, d_model))

        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, padding_mask: torch.Tensor = None):
        """process input through the encoder layer.
        :arg src: input tensor [batch_size, src_len, d_model]
        :arg src_mask: mask for self-attention (optional)
        :arg padding_mask: mask for padding tokens (optional)

        returns output tensor after self-attention and feed-forward
        """
        # self-attention block with residual connection and layer norm
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
