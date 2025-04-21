import torch
from models.transformer import ArithmeticTransformer
from models.transformer_config import ArithmeticTransformerConfig
from data.processing.tokenizer import ArithmeticTokenizer


def initialize_model():
    """Initialize the Arithmetic Transformer model with configuration parameters."""

    # Create tokenizer instance
    tokenizer = ArithmeticTokenizer()

    # Get vocabulary size from tokenizer
    vocab_size = tokenizer.get_vocab_size()

    # Load configuration
    config = ArithmeticTransformerConfig

    # Create model instance
    model = ArithmeticTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )

    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Initialized Arithmetic Transformer with:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Embedding dimension: {config.d_model}")
    print(f"- Encoder layers: {config.num_encoder_layers}")
    print(f"- Decoder layers: {config.num_decoder_layers}")
    print(f"- Attention heads: {config.num_heads}")
    print(f"- Feed-forward dimension: {config.d_ff}")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = initialize_model()

    # Example of model usage
    src = torch.randint(0, tokenizer.get_vocab_size(), (2, 10))  # Batch of 2, length 10
    tgt = torch.randint(0, tokenizer.get_vocab_size(), (2, 8))  # Batch of 2, length 8

    # Forward pass
    output = model(src, tgt)
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")