import math
from typing import Literal

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model.

    This adds positional information to the input embeddings to provide
    sequence order information to the self-attention mechanism.
    """

    def __init__(self, d_model, max_seq_length=64, dropout=0.1):
        """Initialize the positional encoding.

        Args:
            d_model: The dimensionality of the embeddings
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (persistent state)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input embeddings.

        Args:
            x: Input embeddings [batch_size, seq_length, d_model]

        Returns:
            Embeddings with positional information added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AdaptivePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
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


# Create a factory function to get the appropriate positional encoding
def get_positional_encoding(encoding_type, d_model, max_seq_length=64, dropout=0.1):
    """Factory function to create the specified positional encoding.

    Args:
        encoding_type: String specifying the type of encoding ('standard' or 'adaptive')
        d_model: The dimensionality of the embeddings
        max_seq_length: Maximum sequence length
        dropout: Dropout probability

    Returns:
        A positional encoding module
    """
    if encoding_type.lower() == 'standard':
        return PositionalEncoding(d_model, max_seq_length, dropout)
    elif encoding_type.lower() == 'adaptive':
        return AdaptivePositionalEncoding(d_model, dropout)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Use 'standard' or 'adaptive'.")


class EncoderLayer(nn.Module):
    """A single layer of the transformer encoder.

    Consists of multi-head self-attention, followed by a position-wise feed-forward network.
    Each sub-layer has a residual connection and is followed by layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Initialize the encoder layer.

        Args:
            d_model: The dimensionality of the embeddings
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_ff, d_model))

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, padding_mask=None):
        """Process input through the encoder layer.

        Args:
            src: Input tensor [batch_size, src_len, d_model]
            src_mask: Optional mask for self-attention
            padding_mask: Optional mask for padding tokens

        Returns:
            Output tensor after self-attention and feed-forward
        """
        # Self-attention block with residual connection and layer norm
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class DecoderLayer(nn.Module):
    """A single layer of the transformer decoder.

    Consists of masked multi-head self-attention, multi-head cross-attention with encoder output,
    and a position-wise feed-forward network. Each sub-layer has a residual connection and
    is followed by layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Initialize the decoder layer.

        Args:
            d_model: The dimensionality of the embeddings
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()

        # Multi-head self-attention (masked)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Multi-head cross-attention with encoder output
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_ff, d_model))

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        """Process input through the decoder layer.

        Args:
            tgt: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Optional mask for self-attention
            memory_mask: Optional mask for cross-attention
            tgt_padding_mask: Optional mask for padding tokens in target
            memory_padding_mask: Optional mask for padding tokens in memory

        Returns:
            Output tensor after self-attention, cross-attention, and feed-forward
        """
        # Self-attention block with residual connection and layer norm
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        tgt = tgt + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        # Cross-attention block with residual connection and layer norm
        cross_attn_output, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                               key_padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt


class Encoder(nn.Module):
    """The transformer encoder.

    Consists of an embedding layer, positional encoding, and multiple encoder layers.
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1,
                 pos_encoding_type: Literal['standard', 'adaptive'] = 'standard'):
        """Initialize the encoder.

        Args:
            vocab_size: Size of the vocabulary
            d_model: The dimensionality of the embeddings
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            pos_encoding_type: Type of positional encoding ('standard' or 'adaptive')
        """
        super(Encoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, max_seq_length, dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, padding_mask=None):
        """Process input through the encoder.

        Args:
            src: Input tensor [batch_size, src_len]
            src_mask: Optional mask for self-attention
            padding_mask: Optional mask for padding tokens

        Returns:
            Encoder output tensor
        """
        # Apply embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)

        # Process through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask, padding_mask)

        # Apply final normalization
        src = self.norm(src)

        return src


class Decoder(nn.Module):
    """The transformer decoder.

    Consists of an embedding layer, positional encoding, and multiple decoder layers.
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1,
                 pos_encoding_type='standard'):
        """Initialize the decoder.

        Args:
            vocab_size: Size of the vocabulary
            d_model: The dimensionality of the embeddings
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            pos_encoding_type: Type of positional encoding ('standard' or 'adaptive')
        """
        super(Decoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, max_seq_length, dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        """Process input through the decoder.

        Args:
            tgt: Input tensor [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Optional mask for self-attention
            memory_mask: Optional mask for cross-attention
            tgt_padding_mask: Optional mask for padding tokens in target
            memory_padding_mask: Optional mask for padding tokens in memory

        Returns:
            Output tensor of vocabulary size logits
        """
        # Apply embedding and positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoder(tgt)

        # Process through each decoder layer
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)

        # Apply final normalization
        tgt = self.norm(tgt)

        # Project to vocabulary size
        output = self.output_projection(tgt)

        return output


class ArithmeticTransformer(nn.Module):
    """Transformer model for arithmetic operations.

    An encoder-decoder transformer model for sequence-to-sequence arithmetic tasks.
    """

    def __init__(self, vocab_size, d_model=128, num_encoder_layers=3, num_decoder_layers=3, num_heads=8, d_ff=512,
                 max_seq_length=64, dropout=0.1, pos_encoding_type='standard'):
        """Initialize the transformer model.

        Args:
            vocab_size: Size of the vocabulary
            d_model: The dimensionality of the embeddings
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            pos_encoding_type: Type of positional encoding ('standard' or 'adaptive')
        """
        super(ArithmeticTransformer, self).__init__()

        # Encoder
        self.encoder = Encoder(vocab_size, d_model, num_encoder_layers, num_heads, d_ff, max_seq_length, dropout,
                               pos_encoding_type)

        # Decoder
        self.decoder = Decoder(vocab_size, d_model, num_decoder_layers, num_heads, d_ff, max_seq_length, dropout,
                               pos_encoding_type)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence to mask future positions.

        Args:
            sz: The size of the square mask

        Returns:
            Square mask of size [sz, sz]
        """
        # Create a mask that prevents attending to future positions
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_padding_mask(self, src, pad_idx=0):
        """Create a mask for padding tokens.

        Args:
            src: Input tensor [batch_size, seq_len]
            pad_idx: Index of the padding token

        Returns:
            Padding mask of size [batch_size, seq_len]
        """
        # Mark padding positions with True
        return src == pad_idx

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """Forward pass through the transformer model.

        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_padding_mask: Optional mask for padding tokens in source
            tgt_padding_mask: Optional mask for padding tokens in target

        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """
        # Create padding masks if not provided
        if src_padding_mask is None:
            src_padding_mask = self.create_padding_mask(src)
        if tgt_padding_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)

        # Create causal mask for decoder self-attention
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # Encode source sequence
        memory = self.encoder(src, padding_mask=src_padding_mask)

        # Decode target sequence
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                              memory_padding_mask=src_padding_mask)

        return output
