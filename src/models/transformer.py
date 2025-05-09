import math
from typing import Literal

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


def get_positional_encoding(encoding_type: Literal["standard", "adaptive"], d_model: int, max_seq_length: int = 64, dropout: float = 0.1):
    """factory function to create the specified positional encoding.

    :arg encoding_type: string specifying the type of encoding ('standard' or 'adaptive')
    :arg d_model: the dimensionality of the embeddings
    :arg max_seq_length: maximum sequence length
    :arg dropout: dropout probability

    returns a positional encoding module
    """
    if encoding_type.lower() == 'standard':
        return PositionalEncoding(d_model, max_seq_length, dropout)
    elif encoding_type.lower() == 'adaptive':
        return AdaptivePositionalEncoding(d_model, dropout)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Use 'standard' or 'adaptive'.")


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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None, memory_padding_mask: torch.Tensor = None):
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


class Encoder(nn.Module):
    """the transformer encoder.

    consists of an embedding layer, positional encoding, and multiple encoder layers.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_length: int, dropout: float = 0.1,
                 pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
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


class Decoder(nn.Module):
    """the transformer decoder.

    consists of an embedding layer, positional encoding, and multiple decoder layers.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_length: int, dropout: float = 0.1,
                 pos_encoding_type: str = 'standard'):
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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None, memory_padding_mask: torch.Tensor = None):
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


class ArithmeticTransformer(nn.Module):
    """transformer model for arithmetic operations.
    an encoder-decoder transformer model for sequence-to-sequence arithmetic tasks.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, num_encoder_layers: int = 3, num_decoder_layers: int = 3, num_heads: int = 8, d_ff: int = 512,
                 max_seq_length: int = 64, dropout: int = 0.1, pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
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

    def generate_square_subsequent_mask(self, sz: int):
        """generate a square mask for the sequence to mask future positions.

        :arg sz: the size of the square mask

        returns square mask of size [sz, sz]
        """
        # create a mask that prevents attending to future positions
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_padding_mask(self, src: torch.Tensor, pad_idx: int = 0):
        """create a mask for padding tokens.

        :arg src: input tensor [batch_size, seq_len]
        :arg pad_idx: index of the padding token

        returns a padding mask of size [batch_size, seq_len]
        """
        # mark padding positions with True
        return src == pad_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None):
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
