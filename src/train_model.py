import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange  # Added tqdm import

from src.data.processing.dataloader import create_dataloaders
from src.data.processing.tokenizer import ArithmeticTokenizer
from src.models.transformer import ArithmeticTransformer
from src.models.transformer_config import ArithmeticTransformerConfig


class ArithmeticTransformerTrainer:
    def __init__(
            self,
            model,
            tokenizer,
            train_loader,
            val_loader,
            test_loader=None,
            criterion=None,
            optimizer=None,
            lr=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_dir="checkpoints"
    ):
        """Initialize the trainer for the ArithmeticTransformer.

        Args:
            model: The transformer model
            tokenizer: The tokenizer for encoding/decoding
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            criterion: Loss function (if None, CrossEntropyLoss will be used)
            optimizer: Optimizer (if None, Adam will be used)
            lr: Learning rate (used if optimizer is None)
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.pad_idx = tokenizer.get_pad_token_id()
        self.sos_token = tokenizer.get_sos_token_id()
        self.eos_token = tokenizer.get_eos_token_id()

        # Define loss function that ignores padding tokens
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        else:
            self.criterion = criterion

        # Define optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.train_char_accuracies = []
        self.val_char_accuracies = []
        self.best_val_loss = float('inf')

    def train(self, epochs, early_stopping_patience=5):
        """Train the model for a specified number of epochs.

        Args:
            epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement before stopping

        Returns:
            Trained model and training history
        """
        print(f"Starting training on device: {self.device}")
        print(f"Training for {epochs} epochs with early stopping patience: {early_stopping_patience}")

        no_improvement_count = 0

        # Use tqdm for epoch iteration
        epoch_iterator = trange(epochs, desc="Epochs")
        for epoch in epoch_iterator:
            start_time = time.time()

            # Training phase
            train_loss, train_metrics = self._train_epoch()
            train_accuracy, train_perplexity, train_char_accuracy = train_metrics

            # Validation phase
            val_loss, val_metrics = self._evaluate(self.val_loader)
            val_accuracy, val_perplexity, val_char_accuracy = val_metrics

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            self.train_char_accuracies.append(train_char_accuracy)
            self.val_char_accuracies.append(val_char_accuracy)

            epoch_time = time.time() - start_time

            # Update tqdm description with metrics
            epoch_iterator.set_description(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {val_accuracy:.4f}"
            )

            # Print detailed epoch results
            print(f"\nEpoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}")
            print(f"Train Perplexity: {train_perplexity:.4f} | Val Perplexity: {val_perplexity:.4f}")
            print(f"Train Char Accuracy: {train_char_accuracy:.4f} | Val Char Accuracy: {val_char_accuracy:.4f}")
            print("-" * 50)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"model_epoch_{epoch + 1}.pt")

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs without improvement.")
                break

        # Test the model if test data is provided
        if self.test_loader is not None:
            self.load_checkpoint("best_model.pt")  # Load the best model for testing
            print("\nEvaluating on test set...")
            test_loss, test_metrics = self._evaluate(self.test_loader)
            test_accuracy, test_perplexity, test_char_accuracy = test_metrics

            print("\nTest Results:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Perplexity: {test_perplexity:.4f}")
            print(f"Test Char Accuracy: {test_char_accuracy:.4f}")

        # Plot and save training history
        self.plot_training_history()

        return self.model

    def _train_epoch(self):
        """Train the model for one epoch.

        Returns:
            Average loss for the epoch and metrics tuple
        """
        self.model.train()
        epoch_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0

        # Add tqdm progress bar for batches
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Extract input_ids and labels from batch
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Create input/output for teacher forcing
            # Use input sequences for encoder and target sequences (shifted) for decoder
            src = input_ids
            tgt = labels[:, :-1]  # Remove last token for decoder input
            tgt_output = labels[:, 1:]  # Remove first token (usually SOS) for targets

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(src, tgt)  # [batch_size, tgt_len-1, vocab_size]

            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.contiguous().view(batch_size * seq_len, vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = self.criterion(outputs, tgt_output)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate metrics
            current_loss = loss.item()
            epoch_loss += current_loss

            # Reshape back for accuracy calculation
            outputs = outputs.view(batch_size, seq_len, vocab_size)
            tgt_output = tgt_output.view(batch_size, seq_len)

            # Calculate exact match accuracy
            predictions = outputs.argmax(dim=2)
            mask = (tgt_output != self.pad_idx)

            # Exact match (entire sequence correct)
            correct_sequences = ((predictions == tgt_output) | ~mask).all(dim=1).sum().item()
            all_correct_sequences += correct_sequences
            total_sequences += batch_size

            # Character-level accuracy
            correct_chars = ((predictions == tgt_output) & mask).sum().item()
            total_chars += mask.sum().item()
            all_correct_chars += correct_chars

            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        # Calculate average metrics
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, (accuracy, perplexity, char_accuracy)

    def _evaluate(self, dataloader):
        """Evaluate the model on the given dataloader.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Average loss and metrics tuple
        """
        self.model.eval()
        total_loss = 0
        all_correct_sequences = 0
        all_correct_chars = 0
        total_sequences = 0
        total_chars = 0

        with torch.no_grad():
            # Add tqdm progress bar for evaluation batches
            pbar = tqdm(dataloader, desc="Evaluating", leave=False)
            for batch in pbar:
                # Extract input_ids and labels from batch
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Create input/output for teacher forcing during evaluation
                src = input_ids
                tgt = labels[:, :-1]
                tgt_output = labels[:, 1:]

                # Forward pass
                outputs = self.model(src, tgt)

                # Reshape for loss calculation
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.contiguous().view(batch_size * seq_len, vocab_size)
                tgt_output = tgt_output.contiguous().view(-1)

                # Calculate loss
                loss = self.criterion(outputs, tgt_output)
                current_loss = loss.item()
                total_loss += current_loss

                # Reshape back for accuracy calculation
                outputs = outputs.view(batch_size, seq_len, vocab_size)
                tgt_output = tgt_output.view(batch_size, seq_len)

                # Calculate exact match accuracy
                predictions = outputs.argmax(dim=2)
                mask = (tgt_output != self.pad_idx)

                # Exact match (entire sequence correct)
                correct_sequences = ((predictions == tgt_output) | ~mask).all(dim=1).sum().item()
                all_correct_sequences += correct_sequences
                total_sequences += batch_size

                # Character-level accuracy
                correct_chars = ((predictions == tgt_output) & mask).sum().item()
                total_chars += mask.sum().item()
                all_correct_chars += correct_chars

                # Update progress bar with current loss
                pbar.set_postfix(loss=f"{current_loss:.4f}")

        # Calculate average metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = all_correct_sequences / total_sequences if total_sequences > 0 else 0
        char_accuracy = all_correct_chars / total_chars if total_chars > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, (accuracy, perplexity, char_accuracy)

    def predict(self, src, max_len=None):
        """Generate predictions using greedy decoding.

        Args:
            src: Source sequence [batch_size, src_len]
            max_len: Maximum length of the generated sequence

        Returns:
            Predicted sequences
        """
        if max_len is None:
            max_len = self.tokenizer.max_length

        self.model.eval()
        batch_size = src.size(0)
        src = src.to(self.device)

        # Create src padding mask
        src_padding_mask = self.model.create_padding_mask(src, self.pad_idx).to(self.device)

        # Encode source sequence
        memory = self.model.encoder(src, padding_mask=src_padding_mask)

        # Initialize target sequence with SOS token
        tgt = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=self.device)

        # Use tqdm for decoding steps
        for i in tqdm(range(max_len - 1), desc="Generating", leave=False):
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

            # Check if all sequences have EOS token
            if (next_token == self.eos_token).all():
                break

        return tgt

    def save_checkpoint(self, filename):
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        # Convert any NumPy arrays to PyTorch tensors to ensure compatibility
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
        print(f"Checkpoint saved to {self.checkpoint_dir / filename}")

    def load_checkpoint(self, filename):
        """Load model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = filename
        if os.path.exists(checkpoint_path):
            try:
                # First try with weights_only=True (new default in PyTorch 2.6+)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except Exception as e:
                print(f"Error loading with weights_only=True: {e}")
                print("Trying with weights_only=False (legacy mode)...")
                # If that fails, try with weights_only=False (old behavior)
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
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def plot_training_history(self):
        """Plot and save training history."""
        plt.figure(figsize=(20, 12))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Exact Match Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot perplexity
        plt.subplot(2, 2, 3)
        plt.plot(self.train_perplexities, label='Train Perplexity')
        plt.plot(self.val_perplexities, label='Validation Perplexity')
        plt.title('Perplexity over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()

        # Plot character-level accuracy
        plt.subplot(2, 2, 4)
        plt.plot(self.train_char_accuracies, label='Train Char Accuracy')
        plt.plot(self.val_char_accuracies, label='Validation Char Accuracy')
        plt.title('Character-level Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png')
        plt.close()


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description="Train Arithmetic Transformer")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="./datasets",
                        help="Path to the dataset directory")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=ArithmeticTransformerConfig.d_model,
                        help="Embedding dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=ArithmeticTransformerConfig.num_encoder_layers,
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=ArithmeticTransformerConfig.num_decoder_layers,
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=ArithmeticTransformerConfig.num_heads,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=ArithmeticTransformerConfig.d_ff,
                        help="Feedforward dimension")
    parser.add_argument("--max_seq_length", type=int, default=ArithmeticTransformerConfig.max_seq_length,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=ArithmeticTransformerConfig.dropout,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    parser.add_argument("--resume", type=str,
                        help="Resume from checkpoint file")

    # Other arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Create tokenizer
    tokenizer = ArithmeticTokenizer(max_length=args.max_seq_length)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=args.num_workers
    )

    # Print dataset information
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")

    # Initialize model
    model = ArithmeticTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )

    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Initialized Arithmetic Transformer with:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Embedding dimension: {args.d_model}")
    print(f"- Encoder layers: {args.num_encoder_layers}")
    print(f"- Decoder layers: {args.num_decoder_layers}")
    print(f"- Attention heads: {args.num_heads}")
    print(f"- Feed-forward dimension: {args.d_ff}")

    # Initialize trainer
    trainer = ArithmeticTransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # Load checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        trainer.plot_training_history()

    # Train model
    trainer.train(epochs=args.epochs, early_stopping_patience=args.patience)

    print("Training completed!")


if __name__ == "__main__":
    main()