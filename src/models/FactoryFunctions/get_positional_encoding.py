from typing import Literal

from models.Encodings.AdaptivePositionalEncoding import AdaptivePositionalEncoding
from models.Encodings.PositionalEncoding import PositionalEncoding


def get_positional_encoding(encoding_type: Literal["standard", "adaptive"], d_model: int, max_seq_length: int = 64,
                            dropout: float = 0.1):
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
