class ArithmeticTransformerConfig:
    """configuration for the transformer. contains default hyperparams.
    """

    # vocab size is 16, as we need tokens for digits 0-9, operators +/-, special tokens, etc. check the tokenizer at src/data/processing/tokenizer.py.
    vocab_size = 16
    d_model = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    num_heads = 8
    d_ff = 512  # 4 * d_model
    max_seq_length = 64
    dropout = 0.1
    pos_encoding_type = 'standard'
