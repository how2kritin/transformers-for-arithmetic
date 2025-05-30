from typing import Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ArithmeticTransformer.ArithmeticTransformer import ArithmeticTransformer
from models.ArithmeticTransformer.ArithmeticTransformerConfig import ArithmeticTransformerConfig
from src.data.processing.tokenizer import ArithmeticTokenizer


class BaseModelHandler:
    """base class for common model operations shared between training and inference."""

    def __init__(self, model: ArithmeticTransformer, tokenizer: ArithmeticTokenizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """initialize the base model handler.

        :arg model: the transformer model
        :arg tokenizer: the tokenizer for encoding/decoding
        :arg device: device to use for operations
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.pad_idx = tokenizer.get_pad_token_id()
        self.sos_token = tokenizer.get_sos_token_id()
        self.eos_token = tokenizer.get_eos_token_id()

        # common criterion for evaluation
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    @staticmethod
    def create_device(no_cuda: bool = False) -> torch.device:
        """create device with standard logic.

        :arg no_cuda: whether to disable cuda

        returns torch device
        """
        return torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")

    @classmethod
    def load_model_from_checkpoint(cls, checkpoint_path: str,
                                   model_config: Optional[ArithmeticTransformerConfig] = None,
                                   max_seq_length: Optional[int] = None,
                                   device: Optional[str] = None) -> Tuple[ArithmeticTransformer, ArithmeticTokenizer]:
        """load model and tokenizer from checkpoint with unified logic.

        :arg checkpoint_path: path to the checkpoint file
        :arg model_config: model configuration (creates default if none)
        :arg max_seq_length: maximum sequence length
        :arg device: device to load model on

        returns tuple of (model, tokenizer)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model configuration
        if model_config is None:
            model_config = ArithmeticTransformerConfig()

        if max_seq_length is None:
            max_seq_length = model_config.max_seq_length

        # create tokenizer
        tokenizer = ArithmeticTokenizer(max_length=max_seq_length)

        # load checkpoint with error handling
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"error loading with weights_only=True: {e}")
            print("trying with weights_only=False (legacy mode)...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # check if the checkpoint was saved with the old positional encoding (backward compatibility with old model type)
        old_style_checkpoint = any('pos_encoder.pe' in key for key in checkpoint['model_state_dict'].keys())
        pos_encoding_type: Literal['standard', 'adaptive'] = 'standard' if old_style_checkpoint else 'adaptive'
        print(f"detected checkpoint format: using {pos_encoding_type} positional encoding")

        # initialize model
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
        print(f"model loaded from checkpoint: {checkpoint_path}")

        return model, tokenizer

    def calculate_batch_metrics(self, outputs: torch.Tensor, tgt_output: torch.Tensor) -> Tuple[int, int, int, int]:
        """calculate metrics for a single batch - shared logic.

        :arg outputs: model outputs [batch_size, seq_len, vocab_size]
        :arg tgt_output: target output [batch_size, seq_len]

        returns tuple of (correct_sequences, batch_size, correct_chars, total_chars)
        """
        batch_size, seq_len, vocab_size = outputs.shape
        predictions = outputs.argmax(dim=2)
        mask = (tgt_output != self.pad_idx)

        # exact match (entire sequence correct)
        correct_sequences = ((predictions == tgt_output) | ~mask).all(dim=1).sum().item()

        # character-level accuracy
        correct_chars = ((predictions == tgt_output) & mask).sum().item()
        total_chars = mask.sum().item()

        return correct_sequences, batch_size, correct_chars, total_chars

    def process_batch_for_loss_and_metrics(self, input_ids: torch.Tensor,
                                           labels: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """process a batch to get loss and metrics - shared logic.

        :arg input_ids: input token ids [batch_size, seq_len]
        :arg labels: target labels [batch_size, seq_len]

        returns tuple of (loss, metrics_tuple)
        """
        # create input/output for teacher forcing
        src = input_ids
        tgt = labels[:, :-1]  # remove last token for decoder input
        tgt_output = labels[:, 1:]  # remove first token (sos) for targets

        # forward pass
        outputs = self.model(src, tgt)  # [batch_size, tgt_len-1, vocab_size]

        # reshape for loss calculation
        batch_size, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.contiguous().view(batch_size * seq_len, vocab_size)
        tgt_output_flat = tgt_output.contiguous().view(-1)

        # calculate loss
        loss = self.criterion(outputs_flat, tgt_output_flat)

        # reshape back for metrics
        outputs = outputs.view(batch_size, seq_len, vocab_size)
        tgt_output = tgt_output.view(batch_size, seq_len)

        # calculate metrics
        metrics = self.calculate_batch_metrics(outputs, tgt_output)

        return loss, metrics

    def evaluate_dataloader(self, dataloader: DataLoader, desc: str = "evaluating") -> Tuple[
        float, Tuple[float, float, float]]:
        """evaluate model on a dataloader - shared logic.

        :arg dataloader: dataloader for evaluation
        :arg desc: description for progress bar

        returns tuple of (avg_loss, (accuracy, perplexity, char_accuracy))
        """
        self.model.eval()
        total_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=desc, leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                loss, (correct_sequences, batch_size, correct_chars, batch_total_chars) = \
                    self.process_batch_for_loss_and_metrics(input_ids, labels)

                total_loss += loss.item()
                all_correct_sequences += correct_sequences
                total_sequences += batch_size
                all_correct_chars += correct_chars
                total_chars += batch_total_chars

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # calculate final metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, (accuracy, perplexity, char_accuracy)

    def predict_greedy(self, src: torch.Tensor, max_len: int) -> torch.Tensor:
        """generate predictions using greedy decoding - shared logic.

        :arg src: source sequence [batch_size, src_len]
        :arg max_len: maximum length of the generated sequence

        returns predicted sequences [batch_size, generated_len]
        """
        self.model.eval()
        batch_size = src.size(0)
        src = src.to(self.device)

        # create src padding mask
        src_padding_mask = self.model.create_padding_mask(src, self.pad_idx).to(self.device)

        # encode source sequence
        memory = self.model.encoder(src, padding_mask=src_padding_mask)

        # initialize target sequence with sos token
        tgt = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=self.device)

        for _ in range(max_len - 1):
            # create target padding mask
            tgt_padding_mask = self.model.create_padding_mask(tgt, self.pad_idx).to(self.device)

            # create causal mask
            tgt_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_len).to(self.device)

            # decode
            output = self.model.decoder(tgt, memory, tgt_mask=tgt_mask,
                                        tgt_padding_mask=tgt_padding_mask,
                                        memory_padding_mask=src_padding_mask)

            # get next token prediction
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=1, keepdim=True)

            # concatenate with target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # check if all sequences have eos token
            if (next_token == self.eos_token).all():
                break

        return tgt
