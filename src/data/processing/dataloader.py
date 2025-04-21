import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from data.processing.dataset import ArithmeticDataset
from data.processing.tokenizer import ArithmeticTokenizer


def create_dataloaders(dataset_path, tokenizer=None, batch_size=32, max_length=64, num_workers=4):
    """create dataloaders for training, validation and testing.

    args:
        dataset_path: path to the dataset directory
        tokenizer: tokenizer to use (creates a new one if None)
        batch_size: batch size for training
        max_length: maximum sequence length
        num_workers: number of workers for data loading

    returns:
        tuple of (train_loader, val_loader, test_loader)
    """
    # create tokenizer if not provided
    if tokenizer is None:
        tokenizer = ArithmeticTokenizer(max_length=max_length)

    # get dataset files
    train_file = os.path.join(dataset_path, 'dataset_train.csv')
    val_file = os.path.join(dataset_path, 'dataset_val.csv')
    test_file = os.path.join(dataset_path, 'dataset_test.csv')

    # check if files exist
    if not os.path.isfile(train_file):
        raise FileNotFoundError(f"training dataset not found: {train_file}")

    # create datasets
    train_dataset = ArithmeticDataset(
        csv_file=train_file,
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = ArithmeticDataset(
        csv_file=val_file,
        tokenizer=tokenizer,
        max_length=max_length
    ) if os.path.isfile(val_file) else None

    test_dataset = ArithmeticDataset(
        csv_file=test_file,
        tokenizer=tokenizer,
        max_length=max_length
    ) if os.path.isfile(test_file) else None

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=num_workers,
        pin_memory=True
    ) if val_dataset else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=num_workers,
        pin_memory=True
    ) if test_dataset else None

    return train_loader, val_loader, test_loader