import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, SequentialSampler

from src.data.processing.tokenizer import ArithmeticTokenizer
from src.models.transformer import ArithmeticTransformer
from src.models.transformer_config import ArithmeticTransformerConfig
from src.data.processing.dataset import ArithmeticDataset


class ArithmeticTransformerInference:
    def __init__(
            self,
            model,
            tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the inference module for ArithmeticTransformer."""
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
        """Create inference module from a checkpoint file."""
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
        """Generate prediction for a single input."""
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

    def evaluate_csv_file_faster(self, csv_file, max_len=None, batch_size=32, num_workers=4):
        """Evaluate the model on data from a CSV file using parallel batch processing."""
        if max_len is None:
            max_len = self.tokenizer.max_length

        # Create a dataset directly using ArithmeticDataset
        dataset = ArithmeticDataset(
            csv_file=csv_file,
            tokenizer=self.tokenizer,
            max_length=max_len
        )

        # Create dataloader for efficient batch processing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Loaded {len(dataset)} samples from {csv_file}")
        print(f"Processing in {len(dataloader)} batches")

        # Ensure model is in evaluation mode
        self.model.eval()

        # Initialize metrics tracking
        total_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0
        all_predictions = []
        all_inputs = []
        all_targets = []

        # Set up criterion
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        with torch.no_grad():
            # Process batches with tqdm progress bar
            pbar = tqdm(dataloader, desc="Evaluating")
            for batch in pbar:
                # Extract input_ids and labels from batch
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Store original inputs and targets for reporting
                for i in range(input_ids.size(0)):
                    # Get tokens up to padding or EOS
                    input_tokens = input_ids[i].cpu().numpy().tolist()
                    eos_pos = input_tokens.index(self.eos_token) if self.eos_token in input_tokens else -1
                    pad_pos = input_tokens.index(self.pad_idx) if self.pad_idx in input_tokens else len(input_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(input_tokens), pad_pos)
                    input_text = self.tokenizer.decode(input_tokens[:end_pos])
                    all_inputs.append(input_text)

                    # Get target tokens up to padding or EOS
                    target_tokens = labels[i].cpu().numpy().tolist()
                    eos_pos = target_tokens.index(self.eos_token) if self.eos_token in target_tokens else -1
                    pad_pos = target_tokens.index(self.pad_idx) if self.pad_idx in target_tokens else len(target_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(target_tokens), pad_pos)
                    target_text = self.tokenizer.decode(target_tokens[1:end_pos])  # Skip SOS token
                    all_targets.append(target_text)

                # Create input/output for teacher forcing evaluation
                src = input_ids
                tgt = labels[:, :-1]  # Remove last token for decoder input
                tgt_output = labels[:, 1:]  # Remove first token (SOS) for targets

                # Forward pass
                outputs = self.model(src, tgt)

                # Reshape for loss calculation
                batch_size_current, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.contiguous().view(batch_size_current * seq_len, vocab_size)
                tgt_output_flat = tgt_output.contiguous().view(-1)

                # Calculate loss
                loss = criterion(outputs_flat, tgt_output_flat)
                current_loss = loss.item()
                total_loss += current_loss

                # Reshape back for accuracy calculation
                outputs = outputs.view(batch_size_current, seq_len, vocab_size)
                tgt_output = tgt_output.view(batch_size_current, seq_len)

                # Calculate predictions
                predictions = outputs.argmax(dim=2)

                # Store predictions for reporting
                for i in range(predictions.size(0)):
                    pred_tokens = predictions[i].cpu().numpy().tolist()
                    mask = (tgt_output[i] != self.pad_idx).cpu().numpy()
                    # Get only valid tokens (before padding)
                    valid_length = sum(mask)
                    valid_pred_tokens = [t for t, m in zip(pred_tokens[:valid_length], mask[:valid_length]) if m]
                    pred_text = self.tokenizer.decode(valid_pred_tokens)
                    all_predictions.append(pred_text)

                # Calculate metrics
                mask = (tgt_output != self.pad_idx)

                # Exact match (entire sequence correct)
                correct_sequences = ((predictions == tgt_output) | ~mask).all(dim=1).sum().item()
                all_correct_sequences += correct_sequences
                total_sequences += batch_size_current

                # Character-level accuracy
                correct_chars = ((predictions == tgt_output) & mask).sum().item()
                total_chars += mask.sum().item()
                all_correct_chars += correct_chars

                # Update progress bar
                pbar.set_postfix(loss=f"{current_loss:.4f}")

        # Calculate final metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

        # Print results
        print(f"Evaluation on {csv_file}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Character Accuracy: {char_accuracy:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "char_accuracy": char_accuracy,
            "perplexity": perplexity,
            "predictions": all_predictions,
            "inputs": all_inputs,
            "targets": all_targets,
            "num_samples": total_sequences
        }


def main():
    """Run inference with a trained arithmetic transformer model."""
    parser = argparse.ArgumentParser(description="Run inference with Arithmetic Transformer")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--input", type=str, required=False,
                        help="Input text for inference (e.g., '2+3')")
    parser.add_argument("--csv_file", type=str, required=False,
                        help="CSV file containing 'expression' and 'result' columns")
    parser.add_argument("--output_file", type=str, required=False,
                        help="File to save output predictions")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum length of the generated sequence")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for processing multiple inputs")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading in fast evaluation mode")

    args = parser.parse_args()

    # Check for valid input options
    if not args.input and not args.csv_file:
        parser.error("At least one of --input or --csv_file must be provided")

    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load model for inference
    inference_model = ArithmeticTransformerInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_seq_length=args.max_length,
        device=device
    )

    # Perform inference based on input type
    if args.csv_file:
        # Use a specific CSV file with fast_eval by default
        print(f"Evaluating model on {args.csv_file}...")
        results = inference_model.evaluate_csv_file_faster(
            csv_file=args.csv_file,
            max_len=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Save predictions if output file is provided
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Character Accuracy: {results['char_accuracy']:.4f}\n")
                f.write(f"Perplexity: {results['perplexity']:.4f}\n\n")

                # Write individual examples
                for i, (inp, target, pred) in enumerate(zip(
                        results['inputs'],
                        results['targets'],
                        results['predictions'])):
                    f.write(f"Example {i + 1}:\n")
                    f.write(f"  Input: {inp}\n")
                    f.write(f"  Target: {target}\n")
                    f.write(f"  Prediction: {pred}\n")
                    f.write(f"  Correct: {pred.strip() == target.strip()}\n\n")
            print(f"Results saved to {args.output_file}")

    elif args.input:
        # Single input
        prediction = inference_model.predict_single(args.input, max_len=args.max_length)
        print(f"Input: {args.input}")
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()