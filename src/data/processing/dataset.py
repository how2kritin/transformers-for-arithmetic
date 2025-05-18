import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class ArithmeticDataset(Dataset):
    """dataset class for arithmetic operations' data.

    this class loads arithmetic expression data from csv files
    and prepares it for transformer model training.
    """

    def __init__(self, csv_file: str, tokenizer, max_length: int = 64):
        """initialize the arithmetic dataset.

        args:
            csv_file: path to the csv file containing expression and result
            tokenizer: tokenizer instance for processing text
            max_length: maximum sequence length
        """
        # ensure file exists
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"dataset file not found: {csv_file}")

        # load data
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """fetch a sample from the dataset.

        args:
            idx: index of the sample to fetch

        returns:
            dict containing input_ids, attention_mask, labels
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get expression and result
        expression = str(self.data.loc[idx, 'expression'])
        result = str(self.data.loc[idx, 'result'])

        # tokenize input (the expression)
        inputs = self.tokenizer.encode(
            expression,
            max_length=self.max_length,
            padding='max_length'
        )

        # tokenize target (the result)
        targets = self.tokenizer.encode(
            result,
            max_length=self.max_length,
            padding='max_length'
        )

        # create sample
        sample = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(targets['input_ids'], dtype=torch.long)
        }

        return sample
