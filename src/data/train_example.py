"""example script showing how to use the dataset and dataloader."""
import os
import torch
from src.data.processing.tokenizer import ArithmeticTokenizer
from src.data.processing.dataloader import create_dataloaders


def main():
    """main function to demonstrate dataset and dataloader usage."""
    # define paths
    dataset_path = '../datasets'

    # create tokenizer
    tokenizer = ArithmeticTokenizer(max_length=64)

    # create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=32,
        max_length=64,
        num_workers=4
    )

    # print dataset information
    print(f"training batches: {len(train_loader)}")
    if val_loader:
        print(f"validation batches: {len(val_loader)}")
    if test_loader:
        print(f"test batches: {len(test_loader)}")

    # example of iterating through the dataloader
    print("\nexample batch:")
    for batch in train_loader:
        # print batch shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")

        # display an example from the batch
        idx = 0
        print(f"\nexpression: {tokenizer.decode(batch['input_ids'][idx])}")
        print(f"target result: {tokenizer.decode(batch['labels'][idx])}")

        # only process one batch for the example
        break


if __name__ == "__main__":
    main()