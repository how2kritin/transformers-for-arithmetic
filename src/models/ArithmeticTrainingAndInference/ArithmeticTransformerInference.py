from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.data.processing.dataset import ArithmeticDataset
from src.models.ArithmeticTrainingAndInference.BaseModelHandler import BaseModelHandler


class ArithmeticTransformerInference(BaseModelHandler):
    """inference class for the arithmetic transformer. inherits common functionality from base handler."""

    def __init__(self, model, tokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """initialize the inference module for arithmetic transformer.

        :arg model: the transformer model
        :arg tokenizer: the tokenizer for encoding/decoding
        :arg device: device to use for inference
        """
        super().__init__(model, tokenizer, device)
        # ensure model is in evaluation mode
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config=None, max_seq_length: int = None, device: str = None):
        """create inference module from a checkpoint file using inherited loader.

        :arg checkpoint_path: path to the checkpoint file
        :arg model_config: model configuration
        :arg max_seq_length: maximum sequence length
        :arg device: device to load model on

        returns inference module instance
        """
        if device is None:
            device = cls.create_device()

        model, tokenizer = cls.load_model_from_checkpoint(
            checkpoint_path, model_config, max_seq_length, device
        )

        return cls(model, tokenizer, device)

    def predict_single(self, input_text: str, max_len: int = None) -> str:
        """generate prediction for a single input.

        :arg input_text: input text to predict on
        :arg max_len: maximum length of the generated sequence

        returns predicted text
        """
        if max_len is None:
            max_len = self.tokenizer.max_length

        # tokenize input
        encoded = self.tokenizer.encode(input_text, padding='max_length')
        src = torch.tensor([encoded['input_ids']], dtype=torch.long, device=self.device)

        # use inherited greedy prediction
        tgt = self.predict_greedy(src, max_len)

        # process output to text
        eos_pos = (tgt[0] == self.eos_token).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            pred_tokens = tgt[0, 1:eos_pos[0]].cpu().numpy().tolist()  # skip sos token
        else:
            pred_tokens = tgt[0, 1:].cpu().numpy().tolist()  # skip sos token

        pred_text = self.tokenizer.decode(pred_tokens)
        return pred_text

    def evaluate_csv_file(self, csv_file: str, max_len: int = None,
                                 batch_size: int = 32, num_workers: int = 4) -> Dict[str, Any]:
        """evaluate the model on data from a csv file using parallel batch processing.

        :arg csv_file: path to csv file with data
        :arg max_len: maximum length of sequences
        :arg batch_size: batch size for processing
        :arg num_workers: number of workers for data loading

        returns dictionary with evaluation results
        """
        if max_len is None:
            max_len = self.tokenizer.max_length

        # create dataset and dataloader
        dataset = ArithmeticDataset(
            csv_file=csv_file,
            tokenizer=self.tokenizer,
            max_length=max_len
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"loaded {len(dataset)} samples from {csv_file}")
        print(f"processing in {len(dataloader)} batches")

        # use inherited evaluation method for core metrics
        avg_loss, (accuracy, perplexity, char_accuracy) = self.evaluate_dataloader(dataloader, "evaluating")

        # additional processing for predictions and detailed output
        all_predictions = []
        all_inputs = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="processing for detailed output")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # store original inputs and targets for reporting
                for i in range(input_ids.size(0)):
                    # process input tokens
                    input_tokens = input_ids[i].cpu().numpy().tolist()
                    eos_pos = input_tokens.index(self.eos_token) if self.eos_token in input_tokens else -1
                    pad_pos = input_tokens.index(self.pad_idx) if self.pad_idx in input_tokens else len(input_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(input_tokens), pad_pos)
                    input_text = self.tokenizer.decode(input_tokens[:end_pos])
                    all_inputs.append(input_text)

                    # process target tokens
                    target_tokens = labels[i].cpu().numpy().tolist()
                    eos_pos = target_tokens.index(self.eos_token) if self.eos_token in target_tokens else -1
                    pad_pos = target_tokens.index(self.pad_idx) if self.pad_idx in target_tokens else len(target_tokens)
                    end_pos = min(eos_pos if eos_pos != -1 else len(target_tokens), pad_pos)
                    target_text = self.tokenizer.decode(target_tokens[1:end_pos])  # skip sos token
                    all_targets.append(target_text)

                # get predictions using model forward pass
                src = input_ids
                tgt = labels[:, :-1]
                outputs = self.model(src, tgt)
                predictions = outputs.argmax(dim=2)

                # process predictions
                for i in range(predictions.size(0)):
                    pred_tokens = predictions[i].cpu().numpy().tolist()
                    tgt_output = labels[i, 1:].cpu().numpy()  # remove sos token
                    mask = (tgt_output != self.pad_idx)

                    # get only valid tokens (before padding)
                    valid_length = sum(mask)
                    valid_pred_tokens = [t for t, m in zip(pred_tokens[:valid_length], mask[:valid_length]) if m]
                    pred_text = self.tokenizer.decode(valid_pred_tokens)
                    all_predictions.append(pred_text)

        print(f"evaluation on {csv_file}:")
        print(f"  loss: {avg_loss:.4f}")
        print(f"  accuracy: {accuracy:.4f}")
        print(f"  character accuracy: {char_accuracy:.4f}")
        print(f"  perplexity: {perplexity:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "char_accuracy": char_accuracy,
            "perplexity": perplexity,
            "predictions": all_predictions,
            "inputs": all_inputs,
            "targets": all_targets,
            "num_samples": len(all_inputs)
        }
