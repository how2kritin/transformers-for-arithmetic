import torch


class ArithmeticTokenizer:
    """custom tokenizer for arithmetic expressions.

    this class provides a simple character-level tokenizer
    tailored for arithmetic expressions.
    """

    def __init__(self, max_length=64):
        """initialize the arithmetic tokenizer.

        args:
            max_length: maximum sequence length
        """
        # define vocabulary for arithmetic operations
        self.vocab = {
            '<pad>': 0,  # padding token
            '<sos>': 1,  # start of sequence
            '<eos>': 2,  # end of sequence
            '<unk>': 3,  # unknown token
            '0': 4,
            '1': 5,
            '2': 6,
            '3': 7,
            '4': 8,
            '5': 9,
            '6': 10,
            '7': 11,
            '8': 12,
            '9': 13,
            '+': 14,
            '-': 15,
        }

        # create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length

    def encode(self, text, max_length=None, padding='do_not_pad'):
        """encode input text to token ids.

        args:
            text: input text to tokenize
            max_length: maximum length (overrides instance value if provided)
            padding: padding strategy ('max_length' or 'do_not_pad')

        returns:
            dict containing input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        # add start token
        tokens = ['<sos>']

        # tokenize character by character
        for char in text:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append('<unk>')

        # add end token
        tokens.append('<eos>')

        # convert to ids
        input_ids = [self.vocab[token] for token in tokens]

        # truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1] + [self.vocab['<eos>']]

        # create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # handle padding
        if padding == 'max_length':
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.vocab['<pad>']] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # prepare output
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        return output

    def decode(self, token_ids, skip_special_tokens=True):
        """decode a list of token ids back to text.

        args:
            token_ids: list of token ids
            skip_special_tokens: whether to skip special tokens

        returns:
            decoded text
        """
        # convert tensor to list if needed
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        # decode tokens
        tokens = [self.id_to_token.get(id, '<unk>') for id in token_ids]

        # filter out special tokens if requested
        if skip_special_tokens:
            special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
            tokens = [t for t in tokens if t not in special_tokens]

        # join tokens to form text
        text = ''.join(tokens)
        return text

    def get_vocab_size(self):
        """return the size of the vocabulary."""
        return len(self.vocab)