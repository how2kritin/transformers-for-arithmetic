import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data.processing.tokenizer import ArithmeticTokenizer
from src.models.transformer import ArithmeticTransformer
from src.models.transformer_config import ArithmeticTransformerConfig


class ArithmeticTransformerInference:
    def __init__(
            self,
            model,
            tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the inference module for ArithmeticTransformer.

        Args:
            model: The transformer model
            tokenizer: The tokenizer for encoding/decoding
            device: Device to use for inference
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.pad_idx = tokenizer.get_pad_token_id()
        self.sos_token = tokenizer.get_sos_token_id()
        self.eos_token = tokenizer.get_eos_token_id()

        # Ensure model is in evaluation mode
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_config=None, max_seq_length=None, device=None):
        """Create inference module from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_config: Configuration for the model (optional)
            max_seq_length: Maximum sequence length for tokenizer (optional)
            device: Device to use for inference (optional)

        Returns:
            ArithmeticTransformerInference instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model configuration
        if model_config is None:
            model_config = ArithmeticTransformerConfig()

        if max_seq_length is None:
            max_seq_length = model_config.max_seq_length

        # Create tokenizer
        tokenizer = ArithmeticTokenizer(max_length=max_seq_length)

        # Initialize empty model
        model = ArithmeticTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_config.d_model,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            max_seq_length=max_seq_length,
            dropout=model_config.dropout
        )

        # Load model weights
        try:
            # First try with weights_only=True (new default in PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Error loading with weights_only=True: {e}")
            print("Trying with weights_only=False (legacy mode)...")
            # If that fails, try with weights_only=False (old behavior)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from checkpoint: {checkpoint_path}")

        return cls(model, tokenizer, device)

    def predict_single(self, input_text, max_len=None):
        """Generate prediction for a single input.

        Args:
            input_text: Input text string
            max_len: Maximum length of the generated sequence

        Returns:
            Predicted text
        """
        if max_len is None:
            max_len = self.tokenizer.max_length

        # Tokenize input
        encoded = self.tokenizer.encode(input_text, padding='max_length')
        src = torch.tensor([encoded['input_ids']], dtype=torch.long, device=self.device)

        self.model.eval()

        # Create src padding mask
        src_padding_mask = self.model.create_padding_mask(src, self.pad_idx).to(self.device)

        # Encode source sequence
        memory = self.model.encoder(src, padding_mask=src_padding_mask)

        # Initialize target sequence with SOS token
        tgt = torch.full((1, 1), self.sos_token, dtype=torch.long, device=self.device)

        # Generate sequence
        for i in range(max_len - 1):
            # Create target padding mask
            tgt_padding_mask = self.model.create_padding_mask(tgt, self.pad_idx).to(self.device)

            # Create causal mask
            tgt_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_len).to(self.device)

            # Decode
            output = self.model.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=src_padding_mask
            )

            # Get next token prediction
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=1, keepdim=True)

            # Concatenate with target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check if we've generated an EOS token
            if next_token.item() == self.eos_token:
                break

        # Find the position of EOS token, if any
        eos_pos = (tgt[0] == self.eos_token).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            # Only keep tokens up to the first EOS
            pred_tokens = tgt[0, 1:eos_pos[0]].cpu().numpy().tolist()  # Skip SOS token
        else:
            # Keep all tokens if no EOS is found
            pred_tokens = tgt[0, 1:].cpu().numpy().tolist()  # Skip SOS token

        # Convert tokens to text
        pred_text = self.tokenizer.decode(pred_tokens)
        return pred_text

    def predict(self, input_text, max_len=None, batch_size=16):
        """Generate predictions using greedy decoding.

        Args:
            input_text: Input text or list of input texts
            max_len: Maximum length of the generated sequence
            batch_size: Size of batches to process at once (to avoid OOM)

        Returns:
            Predicted text or list of predicted texts
        """
        # Handle single input or list of inputs
        is_single_input = not isinstance(input_text, list)
        input_texts = [input_text] if is_single_input else input_text

        # Process inputs in batches to avoid OOM
        all_predictions = []

        # Use tqdm to show progress across batches
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Processing batches"):
            batch_inputs = input_texts[i:i + batch_size]

            # Process each input in the batch individually to avoid GPU OOM
            batch_predictions = []
            for single_input in tqdm(batch_inputs, desc="Generating", leave=False):
                pred = self.predict_single(single_input, max_len=max_len)
                batch_predictions.append(pred)

            all_predictions.extend(batch_predictions)

        # Return a single prediction or list of predictions
        return all_predictions[0] if is_single_input else all_predictions

    def evaluate_batch(self, inputs, targets, batch_size=16):
        """Evaluate the model on a batch of inputs and targets.

        Args:
            inputs: List of input texts
            targets: List of target texts
            batch_size: Size of batches to process at once

        Returns:
            Dictionary of metrics
        """
        # Generate predictions
        predictions = self.predict(inputs, batch_size=batch_size)

        # Calculate metrics
        exact_matches = 0
        for pred, target in zip(predictions, targets):
            if pred.strip() == target.strip():
                exact_matches += 1

        accuracy = exact_matches / len(inputs) if len(inputs) > 0 else 0

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "num_samples": len(inputs)
        }


def main():
    """Run inference with a trained arithmetic transformer model."""
    parser = argparse.ArgumentParser(description="Run inference with Arithmetic Transformer")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--input", type=str, required=False,
                        help="Input text for inference")
    parser.add_argument("--input_file", type=str, required=False,
                        help="File containing input texts, one per line")
    parser.add_argument("--output_file", type=str, required=False,
                        help="File to save output predictions")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum length of the generated sequence")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for processing multiple inputs")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")

    args = parser.parse_args()

    # Check that either input or input_file is provided
    if not args.input and not args.input_file:
        parser.error("Either --input or --input_file must be provided")

    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load model for inference
    inference_model = ArithmeticTransformerInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_seq_length=args.max_length,
        device=device
    )

    # Perform inference
    if args.input:
        # Single input
        prediction = inference_model.predict(args.input, max_len=args.max_length)
        print(f"Input: {args.input}")
        print(f"Prediction: {prediction}")

    elif args.input_file:
        # Multiple inputs from file
        with open(args.input_file, 'r') as f:
            input_texts = [line.strip() for line in f.readlines()]

        print(f"Running inference on {len(input_texts)} inputs...")
        predictions = inference_model.predict(
            input_texts,
            max_len=args.max_length,
            batch_size=args.batch_size
        )

        # Save predictions if output file is provided
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for inp, pred in zip(input_texts, predictions):
                    f.write(f"Input: {inp}\nPrediction: {pred}\n\n")
            print(f"Predictions saved to {args.output_file}")
        else:
            # Print predictions to console (limit to first 20 for readability)
            for i, (inp, pred) in enumerate(zip(input_texts[:20], predictions[:20])):
                print(f"Example {i + 1}:")
                print(f"  Input: {inp}")
                print(f"  Prediction: {pred}")
                print("")

            if len(input_texts) > 20:
                print(f"... and {len(input_texts) - 20} more predictions (not shown)")


if __name__ == "__main__":
    main()