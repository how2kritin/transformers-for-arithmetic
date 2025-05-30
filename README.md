# Transformers for Arithmetic

A Transformer-based encoder-decoder model for solving arithmetic operations. This project implements a
sequence-to-sequence transformer that learns to perform addition and subtraction with various complexities (carry/borrow
operations, negative numbers). The model is trained on synthetic arithmetic datasets and evaluated on both
in-distribution and out-of-distribution generalization tasks.

Implemented in Python using PyTorch. Followed the architecture defined in the ["Attention is All You Need" paper (Vaswani et al., 2023)](https://arxiv.org/abs/1706.03762). This corresponds to Assignment-5 of the Introduction to Natural Language Processing
course at IIIT Hyderabad, taken in the Spring'25 semester.

Please check the [Report.md](Report.md) if you'd like to take a look at the evaluation results.

---

# Pre-requisites

1. `python 3.12`
2. A python package manager such as `pip` or `conda`.
3. [pytorch](https://pytorch.org/get-started/locally/)
4. (OPTIONAL) `virtualenv` to create a virtual environment.
5. All the python libraries mentioned in `requirements.txt`.

---

# Dataset Generation

## Instructions to generate arithmetic datasets

### Usage

```bash
python -m src.data.datagen.dataset_generator [--num_samples NUM_SAMPLES] [--min_digits MIN_DIGITS] [--max_digits MAX_DIGITS] 
                                            [--output_dir OUTPUT_DIR] [--filename FILENAME] [--gen_min_digits GEN_MIN_DIGITS] 
                                            [--gen_max_digits GEN_MAX_DIGITS] [--gen_filename GEN_FILENAME] [--seed SEED]
```

### Arguments

- `--num_samples <count>` : Total number of examples to generate (defaults to 200,000, divided by 8 for each category).
- `--min_digits <count>` : Minimum number of digits for regular dataset (defaults to 1).
- `--max_digits <count>` : Maximum number of digits for regular dataset (defaults to 10).
- `--output_dir <path>` : Directory to save the dataset files (defaults to './datasets').
- `--filename <name>` : Filename for the regular dataset (defaults to 'dataset.csv').
- `--gen_min_digits <count>` : Minimum number of digits for generalization dataset (optional).
- `--gen_max_digits <count>` : Maximum number of digits for generalization dataset (optional).
- `--gen_filename <name>` : Filename for the generalization dataset (optional).
- `--seed <value>` : Random seed for reproducibility (defaults to 42).

### Dataset Categories

The generator creates 8 different categories of arithmetic problems:

- **a + b (no carry)** : Addition without carry operations
- **a + b (carry)** : Addition with carry operations
- **-a - b (no borrow)** : Negative subtraction without borrowing
- **-a - b (borrow)** : Negative subtraction with borrowing
- **a - b (no borrow)** : Positive subtraction without borrowing
- **a - b (borrow)** : Positive subtraction with borrowing
- **-a + b (no carry)** : Addition with negative first operand, no carry
- **-a + b (carry)** : Addition with negative first operand, with carry

### Example Usage

Generate a basic training dataset:

```bash
python -m src.data.datagen.dataset_generator --num_samples 160000 --max_digits 8 --filename training_dataset.csv
```

Generate both training and generalization datasets:

```bash
python -m src.data.datagen.dataset_generator --num_samples 200000 --max_digits 10 --gen_filename dataset_generalization.csv
```

Generate with custom parameters:

```bash
python -m src.data.datagen.dataset_generator --num_samples 100000 --min_digits 2 --max_digits 6 --gen_min_digits 7 --gen_max_digits 12 --gen_filename generalization_test.csv
```

---

# Model Training

## Instructions to train the arithmetic transformer

### Usage

```bash
python -m src.train_model [--dataset_path DATASET_PATH] [--d_model D_MODEL] [--num_encoder_layers NUM_ENCODER_LAYERS] 
                         [--num_decoder_layers NUM_DECODER_LAYERS] [--num_heads NUM_HEADS] [--d_ff D_FF] 
                         [--max_seq_length MAX_SEQ_LENGTH] [--dropout DROPOUT] [--pos_encoding_type {standard,adaptive}]
                         [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--no_cuda] [--learning_rate LEARNING_RATE] 
                         [--epochs EPOCHS] [--patience PATIENCE] [--resume RESUME] [--checkpoint_dir CHECKPOINT_DIR]
```

### Arguments

#### Data Arguments

- `--dataset_path <path>` : Path to the dataset directory (defaults to './datasets').

#### Model Architecture Arguments

- `--d_model <size>` : Dimensionality of the embeddings (defaults to 128).
- `--num_encoder_layers <count>` : Number of encoder layers (defaults to 3).
- `--num_decoder_layers <count>` : Number of decoder layers (defaults to 3).
- `--num_heads <count>` : Number of attention heads (defaults to 8).
- `--d_ff <size>` : Dimensionality of the feed-forward network (defaults to 512).
- `--max_seq_length <length>` : Maximum sequence length (defaults to 64).
- `--dropout <rate>` : Dropout probability (defaults to 0.1).
- `--pos_encoding_type {standard,adaptive}` : Type of positional encoding (defaults to 'standard').

#### Training Arguments

- `--batch_size <size>` : Training batch size (defaults to 32).
- `--num_workers <count>` : Number of data loading workers (defaults to 4).
- `--no_cuda` : Disable CUDA and use CPU only.
- `--learning_rate <rate>` : Learning rate (defaults to 1e-4).
- `--epochs <count>` : Number of training epochs (defaults to 50).
- `--patience <count>` : Early stopping patience (defaults to 5).
- `--resume <path>` : Resume from checkpoint file (optional).
- `--checkpoint_dir <path>` : Directory to save checkpoints (defaults to 'checkpoints').

### Model Architecture

The transformer uses an encoder-decoder architecture with:

- **Vocabulary size**: 16 tokens (digits 0-9, operators +/-, special tokens)
- **Encoder**: Processes the arithmetic expression
- **Decoder**: Generates the result autoregressively
- **Attention mechanism**: Multi-head self-attention and cross-attention
- **Positional encoding**: Standard or adaptive positional embeddings

### Example Usage

Train with default settings:

```bash
python -m src.train_model --epochs 100
```

Train with custom architecture:

```bash
python -m src.train_model --d_model 256 --num_encoder_layers 6 --num_decoder_layers 6 --num_heads 16 --epochs 100
```

Train with modified training parameters:

```bash
python -m src.train_model --batch_size 64 --learning_rate 5e-4 --epochs 150 --patience 10
```

Resume training from checkpoint:

```bash
python -m src.train_model --resume checkpoints/best_model.pt --epochs 100
```

Train with adaptive positional encoding:

```bash
python -m src.train_model --pos_encoding_type adaptive --d_model 256 --epochs 100
```

---

# Model Inference

## Instructions to run inference with trained models

### Usage

```bash
python -m src.inference --checkpoint CHECKPOINT [--input INPUT] [--csv_file CSV_FILE] [--output_file OUTPUT_FILE] 
                       [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--max_length MAX_LENGTH] [--no_cuda]
```

### Arguments

- `--checkpoint <path>` : Path to the model checkpoint file (required).
- `--input <expression>` : Input arithmetic expression for single inference (e.g., '2+3').
- `--csv_file <path>` : CSV file containing 'expression' and 'result' columns for batch evaluation.
- `--output_file <path>` : File to save output predictions and evaluation results.
- `--batch_size <size>` : Batch size for inference (defaults to 256).
- `--num_workers <count>` : Number of data loading workers (defaults to 4).
- `--max_length <length>` : Maximum sequence length (defaults to 64).
- `--no_cuda` : Disable CUDA and use CPU only.

### Output Format

When evaluating a CSV file with `--output_file`, the inference generates:

- **Summary file** (`output_file`): Overall metrics including accuracy, character accuracy, and perplexity
- **Correct predictions file** (`output_file_correct.txt`): All correctly predicted examples
- **Incorrect predictions file** (`output_file_incorrect.txt`): All incorrectly predicted examples

### Evaluation Metrics

- **Accuracy**: Percentage of completely correct sequence predictions
- **Character Accuracy**: Percentage of correctly predicted characters
- **Perplexity**: Model's uncertainty measure

### Example Usage

Single expression inference:

```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --input "123+456"
```

Evaluate on a test dataset:

```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --csv_file datasets/dataset_generalization.csv
```

Evaluate and save detailed results:

```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --csv_file datasets/test_data.csv --output_file results/evaluation_results.txt
```

Batch inference with custom settings:

```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --csv_file datasets/large_test.csv --batch_size 512 --output_file results/large_test_results.txt
```

Test multiple expressions:

```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --input "999+1"
python -m src.inference --checkpoint checkpoints/best_model.pt --input "1000-999"
python -m src.inference --checkpoint checkpoints/best_model.pt --input "-50+25"
```

---

# Quick Start

1. **Generate training data**:

    ```bash
    python -m src.data.datagen.dataset_generator --num_samples 200000 --max_digits 10
    ```

2. **Generate generalization test data**:

    ```bash
    python -m src.data.datagen.dataset_generator --gen_filename dataset_generalization.csv --gen_min_digits 11 --gen_max_digits 15
    ```

3. **Train the model**:

    ```bash
    python -m src.train_model --epochs 100 --batch_size 64
    ```

4. **Run inference**:

    ```bash
    python -m src.inference --checkpoint checkpoints/best_model.pt --csv_file datasets/dataset_generalization.csv --output_file results/generalization_test.txt
    ```

---