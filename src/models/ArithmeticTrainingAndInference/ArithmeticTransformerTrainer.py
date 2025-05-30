import os
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data.processing.tokenizer import ArithmeticTokenizer
from models.ArithmeticTrainingAndInference.BaseModelHandler import BaseModelHandler
from models.ArithmeticTransformer.ArithmeticTransformer import ArithmeticTransformer


class ArithmeticTransformerTrainer(BaseModelHandler):
    """trainer class for the arithmetic transformer. inherits common functionality from base handler."""

    def __init__(self, model: ArithmeticTransformer, tokenizer: ArithmeticTokenizer,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None,
                 criterion=None, optimizer=None, lr: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: str = "checkpoints"):
        """initialize the trainer for the arithmetic transformer.

        :arg model: the transformer model
        :arg tokenizer: the tokenizer for encoding/decoding
        :arg train_loader: dataloader for training data
        :arg val_loader: dataloader for validation data
        :arg test_loader: dataloader for test data (optional)
        :arg criterion: loss function (if none, crossentropyloss will be used by default)
        :arg optimizer: optimizer (if none, adam will be used by default)
        :arg lr: learning rate (used if optimizer is none to initialize the default optimizer)
        :arg device: device to use for training
        :arg checkpoint_dir: directory to save checkpoints
        """
        super().__init__(model, tokenizer, device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # override criterion if provided
        if criterion is not None:
            self.criterion = criterion

        # define optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.train_char_accuracies = []
        self.val_char_accuracies = []
        self.best_val_loss = float('inf')

    def train(self, epochs: int, early_stopping_patience: int = 5) -> ArithmeticTransformer:
        """train the model for a specified number of epochs.

        :arg epochs: number of epochs to train for
        :arg early_stopping_patience: number of epochs to wait for improvement before stopping

        returns trained model
        """
        print(f"starting training on device: {self.device}")
        print(f"training for {epochs} epochs with early stopping patience: {early_stopping_patience}")

        no_improvement_count = 0

        epoch_iterator = trange(epochs, desc="epochs")
        for epoch in epoch_iterator:
            start_time = time.time()

            # training phase
            train_loss, train_metrics = self._train_epoch()
            train_accuracy, train_perplexity, train_char_accuracy = train_metrics

            # validation phase - use inherited method
            val_loss, val_metrics = self.evaluate_dataloader(self.val_loader, "validating")
            val_accuracy, val_perplexity, val_char_accuracy = val_metrics

            # record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            self.train_char_accuracies.append(train_char_accuracy)
            self.val_char_accuracies.append(val_char_accuracy)

            epoch_time = time.time() - start_time

            epoch_iterator.set_description(f"epoch {epoch + 1}/{epochs} | "
                                           f"train loss: {train_loss:.4f} | "
                                           f"val loss: {val_loss:.4f} | "
                                           f"acc: {val_accuracy:.4f}")

            # print detailed epoch results
            print(f"\nepoch {epoch + 1}/{epochs} | time: {epoch_time:.2f}s")
            print(f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
            print(f"train accuracy: {train_accuracy:.4f} | val accuracy: {val_accuracy:.4f}")
            print(f"train perplexity: {train_perplexity:.4f} | val perplexity: {val_perplexity:.4f}")
            print(f"train char accuracy: {train_char_accuracy:.4f} | val char accuracy: {val_char_accuracy:.4f}")
            print("-" * 50)

            # save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # save regular checkpoint (once every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"model_epoch_{epoch + 1}.pt")

            # early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"early stopping after {epoch + 1} epochs without improvement.")
                break

        # test the model if test data is provided
        if self.test_loader is not None:
            self.load_checkpoint("best_model.pt")
            print("\nevaluating on test set...")
            test_loss, test_metrics = self.evaluate_dataloader(self.test_loader, "testing")
            test_accuracy, test_perplexity, test_char_accuracy = test_metrics

            print("\ntest results:")
            print(f"test loss: {test_loss:.4f}")
            print(f"test accuracy: {test_accuracy:.4f}")
            print(f"test perplexity: {test_perplexity:.4f}")
            print(f"test char accuracy: {test_char_accuracy:.4f}")

        # plot and save training history
        self.plot_training_history()

        return self.model

    def _train_epoch(self) -> Tuple[float, Tuple[float, float, float]]:
        """train the model for one epoch.

        returns tuple of (avg_loss, (accuracy, perplexity, char_accuracy))
        """
        self.model.train()
        epoch_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0

        pbar = tqdm(self.train_loader, desc="training", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # reset gradients
            self.optimizer.zero_grad()

            # forward pass and get loss/metrics using inherited method
            loss, (correct_sequences, batch_size, correct_chars, batch_total_chars) = \
                self.process_batch_for_loss_and_metrics(input_ids, labels)

            # backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            all_correct_sequences += correct_sequences
            total_sequences += batch_size
            all_correct_chars += correct_chars
            total_chars += batch_total_chars

            pbar.set_postfix(loss=f"{current_loss:.4f}")

        # calculate average metrics
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, (accuracy, perplexity, char_accuracy)

    def predict(self, src: torch.Tensor, max_len: int = None) -> torch.Tensor:
        """generate predictions using inherited greedy decoding.

        :arg src: source sequence [batch_size, src_len]
        :arg max_len: maximum length of the generated sequence

        returns predicted sequences
        """
        if max_len is None:
            max_len = self.tokenizer.max_length
        return self.predict_greedy(src, max_len)

    def save_checkpoint(self, filename: str) -> None:
        """save model checkpoint.

        :arg filename: name of the checkpoint file
        """
        # convert any numpy arrays to pytorch tensors to ensure compatibility
        train_losses = torch.tensor(self.train_losses) if self.train_losses else []
        val_losses = torch.tensor(self.val_losses) if self.val_losses else []
        train_accuracies = torch.tensor(self.train_accuracies) if self.train_accuracies else []
        val_accuracies = torch.tensor(self.val_accuracies) if self.val_accuracies else []
        train_perplexities = torch.tensor(self.train_perplexities) if self.train_perplexities else []
        val_perplexities = torch.tensor(self.val_perplexities) if self.val_perplexities else []
        train_char_accuracies = torch.tensor(self.train_char_accuracies) if self.train_char_accuracies else []
        val_char_accuracies = torch.tensor(self.val_char_accuracies) if self.val_char_accuracies else []

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities,
            'train_char_accuracies': train_char_accuracies,
            'val_char_accuracies': val_char_accuracies,
            'best_val_loss': float(self.best_val_loss)
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"checkpoint saved to {self.checkpoint_dir / filename}")

    def load_checkpoint(self, filename: str) -> None:
        """load model checkpoint.

        :arg filename: name of the checkpoint file
        """
        checkpoint_path = filename
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except Exception as e:
                print(f"error loading with weights_only=True: {e}")
                print("trying with weights_only=False (legacy mode)...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', []).cpu().numpy()
            self.val_losses = checkpoint.get('val_losses', []).cpu().numpy()
            self.train_accuracies = checkpoint.get('train_accuracies', []).cpu().numpy()
            self.val_accuracies = checkpoint.get('val_accuracies', []).cpu().numpy()
            self.train_perplexities = checkpoint.get('train_perplexities', []).cpu().numpy()
            self.val_perplexities = checkpoint.get('val_perplexities', []).cpu().numpy()
            self.train_char_accuracies = checkpoint.get('train_char_accuracies', []).cpu().numpy()
            self.val_char_accuracies = checkpoint.get('val_char_accuracies', []).cpu().numpy()
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"checkpoint loaded from {checkpoint_path}")
        else:
            print(f"no checkpoint found at {checkpoint_path}")

    def plot_training_history(self) -> None:
        """plot and save training history."""
        plt.figure(figsize=(20, 12))

        # plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='validation loss')
        plt.title('loss over epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        # plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='train accuracy')
        plt.plot(self.val_accuracies, label='validation accuracy')
        plt.title('exact match accuracy over epochs')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

        # plot perplexity
        plt.subplot(2, 2, 3)
        plt.plot(self.train_perplexities, label='train perplexity')
        plt.plot(self.val_perplexities, label='validation perplexity')
        plt.title('perplexity over epochs')
        plt.xlabel('epoch')
        plt.ylabel('perplexity')
        plt.legend()

        # plot character-level accuracy
        plt.subplot(2, 2, 4)
        plt.plot(self.train_char_accuracies, label='train char accuracy')
        plt.plot(self.val_char_accuracies, label='validation char accuracy')
        plt.title('character-level accuracy over epochs')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png')
        plt.close()
