import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.data.processing.dataset import ArithmeticDataset
from src.data.processing.tokenizer import ArithmeticTokenizer
from src.models.transformer import ArithmeticTransformer
from src.models.transformer_config import ArithmeticTransformerConfig


class ArithmeticTransformerInference:
    def __init__(
            self,
            model: ArithmeticTransformer,
            tokenizer: ArithmeticTokenizer,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """initialize the inference module for ArithmeticTransformer."""
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.pad_idx = tokenizer.get_pad_token_id()
        self.sos_token = tokenizer.get_sos_token_id()
        self.eos_token = tokenizer.get_eos_token_id()

        # ensure model is in evaluation mode
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config: ArithmeticTransformerConfig = None,
                        max_seq_length: int = None, device: str = None):
        """create inference module from a checkpoint file."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model configuration
        if model_config is None:
            model_config = ArithmeticTransformerConfig()

        if max_seq_length is None:
            max_seq_length = model_config.max_seq_length

        # create tokenizer
        tokenizer = ArithmeticTokenizer(max_length=max_seq_length)

        # load the checkpoint first to check its structure (try with and without weights_only)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Error loading with weights_only=True: {e}")
            print("Trying with weights_only=False (legacy mode)...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # check if the checkpoint was saved with the old positional encoding (backward compatibility with my old model type)
        old_style_checkpoint = any('pos_encoder.pe' in key for key in checkpoint['model_state_dict'].keys())

        # initialize model with the appropriate positional encoding type
        pos_encoding_type = 'standard' if old_style_checkpoint else 'adaptive'
        print(f"Detected checkpoint format: Using {pos_encoding_type} positional encoding")

        # initialize empty model with the appropriate encoding type
        model = ArithmeticTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_config.d_model,
            num_encoder_layers=model_config.num_encoder_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            max_seq_length=max_seq_length,
            dropout=model_config.dropout,
            pos_encoding_type=pos_encoding_type
        )

        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from checkpoint: {checkpoint_path}")

        return cls(model, tokenizer, device)

    def predict_single(self, input_text: str, max_len: int = None):
        """generate prediction for a single input."""
        if max_len is None:
            max_len = self.tokenizer.max_length

        # tokenize input
        encoded = self.tokenizer.encode(input_text, padding='max_length')
        src = torch.tensor([encoded['input_ids']], dtype=torch.long, device=self.device)

        self.model.eval()

        # create src padding mask
        src_padding_mask = self.model.create_padding_mask(src, self.pad_idx).to(self.device)

        # encode source sequence
        memory = self.model.encoder(src, padding_mask=src_padding_mask)

        # initialize target sequence with SOS token
        tgt = torch.full((1, 1), self.sos_token, dtype=torch.long, device=self.device)

        # generate sequence
        for i in range(max_len - 1):
            # create target padding mask
            tgt_padding_mask = self.model.create_padding_mask(tgt, self.pad_idx).to(self.device)

            # create causal mask
            tgt_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_len).to(self.device)

            # decode
            output = self.model.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=src_padding_mask
            )

            # get next token prediction
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=1, keepdim=True)

            # concatenate with target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # check if we've generated an EOS token
            if next_token.item() == self.eos_token:
                break

        # find the position of EOS token, if any
        eos_pos = (tgt[0] == self.eos_token).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            # only keep tokens up to the first EOS
            pred_tokens = tgt[0, 1:eos_pos[0]].cpu().numpy().tolist()  # skip SOS token
        else:
            # keep all tokens if no EOS is found
            pred_tokens = tgt[0, 1:].cpu().numpy().tolist()  # skip SOS token

        # convert tokens to text
        pred_text = self.tokenizer.decode(pred_tokens)
        return pred_text

    def evaluate_csv_file_faster(self, csv_file: str, max_len: int = None, batch_size: int = 32, num_workers: int = 4):
        """evaluate the model on data from a CSV file using parallel batch processing."""
        if max_len is None:
            max_len = self.tokenizer.max_length

        # create a dataset directly using ArithmeticDataset
        dataset = ArithmeticDataset(
            csv_file=csv_file,
            tokenizer=self.tokenizer,
            max_length=max_len
        )

        # create dataloader for efficient batch processing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Loaded {len(dataset)} samples from {csv_file}")
        print(f"Processing in {len(dataloader)} batches")

        # ensure model is in evaluation mode
        self.model.eval()

        # initialize metrics tracking
        total_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0
        all_predictions = []
        all_inputs = []
        all_targets = []

        # set up criterion
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating")
            for batch in pbar:
                # extract input_ids and labels from batch
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # store original inputs and targets for reporting
                for i in range(input_ids.size(0)):
                    # get tokens up to padding or EOS
                    input_tokens = input_ids[i].cpu().numpy().tolist()
                    eos_pos = input_tokens.index(self.eos_token) if self.eos_token in input_tokens else -1
                    pad_pos = input_tokens.index(self.pad_idx) if self.pad_idx in input_tokens else len(input_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(input_tokens), pad_pos)
                    input_text = self.tokenizer.decode(input_tokens[:end_pos])
                    all_inputs.append(input_text)

                    # get target tokens up to padding or EOS
                    target_tokens = labels[i].cpu().numpy().tolist()
                    eos_pos = target_tokens.index(self.eos_token) if self.eos_token in target_tokens else -1
                    pad_pos = target_tokens.index(self.pad_idx) if self.pad_idx in target_tokens else len(target_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(target_tokens), pad_pos)
                    target_text = self.tokenizer.decode(target_tokens[1:end_pos])  # skip SOS token
                    all_targets.append(target_text)

                # create input/output for teacher forcing evaluation
                src = input_ids
                tgt = labels[:, :-1]  # remove last token for decoder input
                tgt_output = labels[:, 1:]  # remove first token (SOS) for targets

                # forward pass
                outputs = self.model(src, tgt)

                # reshape for loss calculation
                batch_size_current, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.contiguous().view(batch_size_current * seq_len, vocab_size)
                tgt_output_flat = tgt_output.contiguous().view(-1)

                # calculate loss
                loss = criterion(outputs_flat, tgt_output_flat)
                current_loss = loss.item()
                total_loss += current_loss

                # reshape back for accuracy calculation
                outputs = outputs.view(batch_size_current, seq_len, vocab_size)
                tgt_output = tgt_output.view(batch_size_current, seq_len)

                # calculate predictions
                predictions = outputs.argmax(dim=2)

                # store predictions for reporting
                for i in range(predictions.size(0)):
                    pred_tokens = predictions[i].cpu().numpy().tolist()
                    mask = (tgt_output[i] != self.pad_idx).cpu().numpy()

                    # get only valid tokens (before padding)
                    valid_length = sum(mask)
                    valid_pred_tokens = [t for t, m in zip(pred_tokens[:valid_length], mask[:valid_length]) if m]
                    pred_text = self.tokenizer.decode(valid_pred_tokens)
                    all_predictions.append(pred_text)

                # calculate metrics
                mask = (tgt_output != self.pad_idx)

                # exact match (entire sequence correct)
                correct_sequences = ((predictions == tgt_output) | ~mask).all(dim=1).sum().item()
                all_correct_sequences += correct_sequences
                total_sequences += batch_size_current

                # character-level accuracy
                correct_chars = ((predictions == tgt_output) & mask).sum().item()
                total_chars += mask.sum().item()
                all_correct_chars += correct_chars

                # update progress bar
                pbar.set_postfix(loss=f"{current_loss:.4f}")

        # calculate final metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

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
    """run inference with a trained arithmetic transformer model."""
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
                        help="Number of workers for data loading")

    args = parser.parse_args()

    # check for valid input options
    if not args.input and not args.csv_file:
        parser.error("At least one of --input or --csv_file must be provided")

    # set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # load model for inference
    inference_model = ArithmeticTransformerInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_seq_length=args.max_length,
        device=device
    )

    # perform inference based on input type
    if args.csv_file:
        # use a specific CSV file
        print(f"Evaluating model on {args.csv_file}...")
        results = inference_model.evaluate_csv_file_faster(
            csv_file=args.csv_file,
            max_len=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # save predictions if output file is provided
        if args.output_file:
            # create base filename without extension
            base_filename = os.path.splitext(args.output_file)[0]
            correct_filename = f"{base_filename}_correct.txt"
            incorrect_filename = f"{base_filename}_incorrect.txt"
            summary_filename = args.output_file

            # count of correct and incorrect predictions
            correct_count = 0
            incorrect_count = 0

            # write summary file with overall metrics
            with open(summary_filename, 'w') as f:
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Character Accuracy: {results['char_accuracy']:.4f}\n")
                f.write(f"Perplexity: {results['perplexity']:.4f}\n\n")
                f.write(f"Total examples: {results['num_samples']}\n")

            # write correct predictions to one file
            with open(correct_filename, 'w') as correct_file, open(incorrect_filename, 'w') as incorrect_file:
                # write headers to both files
                for file in [correct_file, incorrect_file]:
                    file.write("Example\tInput\tTarget\tPrediction\n")
                    file.write("-" * 60 + "\n")

                # write individual examples
                for i, (inp, target, pred) in enumerate(zip(
                        results['inputs'],
                        results['targets'],
                        results['predictions'])):

                    is_correct = pred.strip() == target.strip()
                    example_text = f"Example {i + 1}:\n"
                    example_text += f"  Input: {inp}\n"
                    example_text += f"  Target: {target}\n"
                    example_text += f"  Prediction: {pred}\n\n"

                    if is_correct:
                        correct_file.write(example_text)
                        correct_count += 1
                    else:
                        incorrect_file.write(example_text)
                        incorrect_count += 1

            # update summary file with count information
            with open(summary_filename, 'a') as f:
                f.write(f"Correct predictions: {correct_count}\n")
                f.write(f"Incorrect predictions: {incorrect_count}\n\n")
                f.write(f"Correct predictions saved to: {os.path.basename(correct_filename)}\n")
                f.write(f"Incorrect predictions saved to: {os.path.basename(incorrect_filename)}\n")

            print(f"Results summary saved to {summary_filename}")
            print(f"Correct predictions ({correct_count}) saved to {correct_filename}")
            print(f"Incorrect predictions ({incorrect_count}) saved to {incorrect_filename}")

    elif args.input:
        # single input
        prediction = inference_model.predict_single(args.input, max_len=args.max_length)
        print(f"Input: {args.input}")
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
