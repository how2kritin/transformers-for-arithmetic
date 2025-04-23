## Name: Kritin Maddireddy

## Roll Number: 2022101071

## Link to checkpoints: https://drive.google.com/drive/folders/1FBGA5FTwvXcSlBgd_qAUCaTHyFZEFIwN?usp=drive_link

---

# Analysis

## Quantitative Performance and Generalization

For best performing model (1 to 10 train/test/val size, 11 to 15 generalization size, 65 epochs).

**Hyperparameters used:**

```
d_model = 128
num_encoder_layers = 3
num_decoder_layers = 3
num_heads = 8
d_ff = 512  # 4 * d_model
max_seq_length = 64
dropout = 0.1
pos_encoding_type = standard
```

```
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
```

Clearly, the model is very good at performing arithmetic on sequences of lengths 1 to 10 (because that's what it has
been trained on), however, it is absolutely garbage at performing arithmetic on larger sequences (which it hasn't been
trained on).

Essentially, it is not "learning" to performing arithmetic, rather, it's memorizing patterns in addition and it is
overfitting to sequences of length 1 to 10. It is unable to generalize effectively. Thus, it fails on longer sequences.

---

## Error Analysis

### Incorrect prediction categories

### Correlation of errors with input characteristics

---

## Ablation/Sensitivity Study

Default (base) model config (1 to 10 train/test/val size, 11 to 15 generalization size).

**Hyperparameters used:**

```
d_model = 128
num_encoder_layers = 3
num_decoder_layers = 3
num_heads = 8
d_ff = 512  # 4 * d_model
max_seq_length = 64
dropout = 0.1
pos_encoding_type = standard
```

**I will be comparing each for 15 epochs of training.**

```
Evaluation on datasets/dataset_test.csv:
  Loss: 0.0421
  Accuracy: 0.9227
  Character Accuracy: 0.9856
  Perplexity: 1.0430

Evaluation on datasets/dataset_generalization.csv:
  Loss: 4.6615
  Accuracy: 0.0435
  Character Accuracy: 0.2750
  Perplexity: 105.7952
```

### Changing the positional encoding method

I changed the positional encoding method to adaptive encoding.

```
Evaluation on datasets/dataset_test.csv:
  Loss: 0.0578
  Accuracy: 0.8861
  Character Accuracy: 0.9799
  Perplexity: 1.0595

Evaluation on datasets/dataset_generalization.csv:
  Loss: 4.8966
  Accuracy: 0.0310
  Character Accuracy: 0.2985
  Perplexity: 133.8381
```

The adaptive positional encoding slightly hurt the model's in-distribution performance.

While the sequence-level generalization accuracy decreased, there was a small improvement in character-level accuracy
for out-of-distribution examples.

This suggests that standard positional encoding was better for exact matches, but adaptive encoding might help the model
better understand individual digit positions for new number ranges.

### Changing $d_{model}$ (and thereby, $d_{ff}$ as $d_{ff}$ is $4 * d_{model}$)

I changed $d_{model}$ to 256 (and thus, $d_{ff}$ to 1024).
This transformer took much longer to train (about 2.5x as long).

```
Evaluation on datasets/dataset_test.csv:
  Loss: 0.0212
  Accuracy: 0.9600
  Character Accuracy: 0.9928
  Perplexity: 1.0215
  
Evaluation on datasets/dataset_generalization.csv:
  Loss: 5.7076
  Accuracy: 0.0312
  Character Accuracy: 0.2763
  Perplexity: 301.1495
```

Doubling the model dimension significantly improved in-distribution performance.

However, it didn't help with generalization at the sequence level and provided minimal improvement at the character
level.

The increased model capacity allowed better memorization of training patterns but not better abstraction of the
underlying arithmetic rules.

The much higher generalization perplexity (301.15 vs 105.80) suggests the larger model is actually more confident in its
wrong answers for out-of-distribution examples, indicating possible overfitting.

### Changing the number of encoder and decoder layers

I doubled the number of encoder and decoder layers (from 3 to 6).

This transformer took much longer to train (about 2x as long).

```
Evaluation on datasets/dataset_test.csv:
  Loss: 0.0166
  Accuracy: 0.9744
  Character Accuracy: 0.9952
  Perplexity: 1.0167

Evaluation on datasets/dataset_generalization.csv:
  Loss: 4.9184
  Accuracy: 0.0384
  Character Accuracy: 0.2844
  Perplexity: 136.7872
```

Doubling the network depth provided the greatest improvement in test accuracy among all modifications.

The model achieved the highest in-distribution character accuracy (99.52%).

While it still struggled with full sequence accuracy on the generalization set, it showed modest improvements in
character-level accuracy.

This suggests deeper networks can better learn the patterns within the training distribution but still face challenges
with systematic generalization.

### General findings

In general, deeper and wider networks provided better in-distribution performance, but this was at the cost of training
time and potentially increased overfitting as well.

The deeper model (more layers) showed better performance as compared to the wider network (higher dimensional layers).

No configuration has dramatically improved generalization performance. This suggests that hyperparameter tuning alone
may not be sufficient for systematic generalization in arithmetic tasks.

---

## Discussion

---