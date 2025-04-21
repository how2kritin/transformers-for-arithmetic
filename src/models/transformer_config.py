"""Configuration for the Arithmetic Transformer model."""


class ArithmeticTransformerConfig:
    """Configuration for the Arithmetic Transformer model.

    Contains hyperparameters for the transformer model and justification for choices.
    """

    # Vocabulary size (based on tokenizer vocabulary)
    # We need tokens for digits 0-9, operators +/-, special tokens, etc.
    vocab_size = 16  # From the ArithmeticTokenizer class

    # Model dimension (embedding size)
    # Smaller than standard NLP transformers since our task is simpler
    # and the vocabulary is much smaller than natural language
    d_model = 128  # 128-dimensional embeddings

    # Number of encoder and decoder layers
    # Fewer than standard NLP transformers since arithmetic has less complex patterns
    num_encoder_layers = 3  # 3 encoder layers
    num_decoder_layers = 3  # 3 decoder layers

    # Number of attention heads
    # Multiple heads allow the model to focus on different parts of the sequence
    # 8 heads split the 128-dim embedding into 16-dim per head, which is reasonable
    num_heads = 8

    # Feed-forward network dimension
    # Standard practice is 4x the embedding dimension but for our simpler task,
    # we can use a smaller multiple (e.g., 4x)
    d_ff = 512  # 4 * d_model

    # Maximum sequence length
    # Must be large enough for the longest arithmetic expression + result
    # Expressions like "-999-999" plus special tokens would need ~10-12 tokens,
    # but we set it higher for safety margin and potential generalization
    max_seq_length = 64

    # Dropout probability
    # Moderate dropout to prevent overfitting but not so high that learning is impaired
    dropout = 0.1

    # Justification for hyperparameter choices:
    """
    The hyperparameters for the Arithmetic Transformer are chosen with the following considerations:

    1. Task Complexity: Arithmetic operations are structurally simpler than natural language tasks, 
       requiring pattern recognition on a small vocabulary of digits and operators. This allows us 
       to use a smaller model than typical NLP transformers.

    2. Model Size vs. Performance: We balance model size with performance by:
       - Using a moderate embedding dimension (128) instead of larger sizes like 512 or 768
       - Using fewer layers (3 encoder, 3 decoder) instead of 6+ layers
       - Maintaining a reasonable feed-forward dimension (512) for pattern capture

    3. Attention Mechanism: We use 8 attention heads to allow the model to focus on different 
       portions of the sequence, which is important for arithmetic operations where position and 
       relationships between digits are critical.

    4. Sequence Length: Maximum sequence length of 64 provides ample space for arithmetic expressions 
       and results, even for generalization to longer numbers than seen in training.

    5. Regularization: A dropout probability of 0.1 helps prevent overfitting while maintaining 
       strong pattern recognition.

    These choices result in a model with approximately:
    - 3 encoder layers × [(128×128×8 for attention) + (128×512×2 for FF)] = ~787K parameters in encoder
    - 3 decoder layers × [(128×128×8×2 for attention) + (128×512×2 for FF)] = ~1.18M parameters in decoder
    - Plus embeddings and output projection: ~4K parameters

    Total: ~2M parameters - A reasonable size that can be trained efficiently on most hardware
    while having sufficient capacity to learn arithmetic operations.
    """