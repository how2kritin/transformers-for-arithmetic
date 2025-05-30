import argparse
import torch
from typing import Dict, Any

from src.models.ArithmeticTransformer.ArithmeticTransformerConfig import ArithmeticTransformerConfig


def add_common_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """add common model arguments to parser.

    :arg parser: argument parser to extend

    returns extended parser
    """
    parser.add_argument("--max_length", type=int, default=None,
                        help="maximum length of the generated sequence")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for processing")
    parser.add_argument("--no_cuda", action="store_true",
                        help="disable cuda")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for data loading")
    return parser


def add_model_architecture_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """add model architecture arguments.

    :arg parser: argument parser to extend

    returns extended parser
    """
    parser.add_argument("--d_model", type=int, default=ArithmeticTransformerConfig.d_model,
                        help="embedding dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=ArithmeticTransformerConfig.num_encoder_layers,
                        help="number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=ArithmeticTransformerConfig.num_decoder_layers,
                        help="number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=ArithmeticTransformerConfig.num_heads,
                        help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=ArithmeticTransformerConfig.d_ff,
                        help="feedforward dimension")
    parser.add_argument("--max_seq_length", type=int, default=ArithmeticTransformerConfig.max_seq_length,
                        help="maximum sequence length")
    parser.add_argument("--dropout", type=float, default=ArithmeticTransformerConfig.dropout,
                        help="dropout rate")
    parser.add_argument('--pos_encoding_type', type=str, default=ArithmeticTransformerConfig.pos_encoding_type,
                        help="type of positional encoding to use (standard or adaptive)")
    return parser


def print_model_info(model: torch.nn.Module, config_args: Dict[str, Any]) -> None:
    """print model information in a standardized way.

    :arg model: the model to print info about
    :arg config_args: configuration arguments dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"initialized arithmetic transformer with:")
    print(f"- total parameters: {total_params:,}")
    print(f"- trainable parameters: {trainable_params:,}")
    print(f"- embedding dimension: {config_args.get('d_model', 'n/a')}")
    print(f"- encoder layers: {config_args.get('num_encoder_layers', 'n/a')}")
    print(f"- decoder layers: {config_args.get('num_decoder_layers', 'n/a')}")
    print(f"- attention heads: {config_args.get('num_heads', 'n/a')}")
    print(f"- feed-forward dimension: {config_args.get('d_ff', 'n/a')}")