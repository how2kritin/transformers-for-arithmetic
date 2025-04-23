"""Configuration for the Arithmetic Transformer model."""


class ArithmeticTransformerConfig:
    """Configuration for the Arithmetic Transformer model.

    Contains hyperparameters for the transformer model and justification for choices.
    """

    # we need tokens for digits 0-9, operators +/-, special tokens, etc.
    vocab_size = 16
    d_model = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    num_heads = 8
    d_ff = 512  # 4 * d_model
    max_seq_length = 64
    dropout = 0.1
    pos_encoding_type = 'standard'
