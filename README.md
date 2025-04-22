# transformers-for-arithmetic

200k element dataset, 1 to 10 train size and 11 to 15 gen size.

Evaluation on datasets/dataset_train.csv:
  Loss: 0.0013
  Accuracy: 0.9975
  Character Accuracy: 0.9996
  Perplexity: 1.0013

Evaluation on datasets/dataset_val.csv:
  Loss: 0.0018
  Accuracy: 0.9962
  Character Accuracy: 0.9994
  Perplexity: 1.0018

Evaluation on datasets/dataset_test.csv:
  Loss: 0.0021
  Accuracy: 0.9960
  Character Accuracy: 0.9994
  Perplexity: 1.0021

Evaluation on datasets/dataset_generalization.csv:
  Loss: 7.7232
  Accuracy: 0.0558
  Character Accuracy: 0.2575
  Perplexity: 2260.1382


d_model = 128  # 128-dimensional embeddings
num_encoder_layers = 3
num_decoder_layers = 3
num_heads = 8
d_ff = 512  # 4 * d_model
max_seq_length = 64
dropout = 0.1
pos_encoding_type = standard