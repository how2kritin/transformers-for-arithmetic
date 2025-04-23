## Name: Kritin Maddireddy

## Roll Number: 2022101071

## Link to checkpoints: https://drive.google.com/drive/folders/1FBGA5FTwvXcSlBgd_qAUCaTHyFZEFIwN?usp=drive_link

---

# Analysis

## Quantitative Performance and Generalization

For best performing model (1 to 10 train/test/val size, 11 to 15 generalization size, 65 epochs).
**Learning rate of 1e-4 for everything reported in this analysis.**

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

### Justification for these hyperparameter choices:

Firstly, arithmetic operations are structurally simpler as compared to typical language modelling tasks. Hence, I will
be making a transformer that's smaller than the one proposed in the "Attention Is All You Need" paper.

**How do I do this?**
 - Using a moderate embedding dimension (128) instead of larger sizes like 512 or 768
 - Using fewer layers (3 encoder, 3 decoder) instead of 6+ layers
 - Maintaining a reasonable feed-forward dimension (512) for pattern capture
 - **Attention Mechanism:** I am using 8 attention heads to allow the model to focus on different
   portions of the sequence, so that it can attempt to learn positions and significance of the symbols and digits.
 - **Sequence Length:** A max sequence length of 64 so that there's enough space to provide larger numbers later to test.

These choices result in a model with approximately:

- 3 encoder layers × [(128×128×8 for attention) + (128×512×2 for FF)] = ~787K parameters in encoder
- 3 decoder layers × [(128×128×8×2 for attention) + (128×512×2 for FF)] = ~1.18M parameters in decoder
- Plus embeddings and output projection: ~4K parameters

Total: ~2M parameters - A reasonable size that can be trained efficiently on most hardware
while having sufficient capacity to learn arithmetic operations.

### Best model accuracy

![training_history.png](checkpoints/checkpoints_200k_65_1to10/training_history.png)

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

Please check the file: [test_output_incorrect.txt](results/test_output_incorrect.txt)

### Incorrect prediction categories

1. **Single Digit Substitution Errors**

   The model often substitutes just one digit in the middle or end of the sequence:

   ```
   Example 508:
     Input: -1769405888+900878847
     Target: -868527041
     Prediction: -868527011
   ```
   
   _A special case of this_: **Leading Digit Errors**

      The model predicts the wrong leading digit while maintaining most of the remaining digits correctly.

      ```
      Example 506:
        Input: 32575826+3967525661
        Target: 4000101487
        Prediction: 3000101487
      ```

2. **Order of Magnitude Errors**

   The model sometimes adds/misses a zero, thereby changing the order of magnitude of the result.

   ```
   Example 4241:
     Input: -4988986778+4986462558
     Target: -2524220
     Prediction: -25242200
   ```

3. **Small number Errors**

   For very small numbers (especially negative single digit numbers), the model has high error rates.

   ```
   Example 5231:
     Input: -107+98
     Target: -9
     Prediction: -39
   ```

### Correlation of errors with input characteristics

**Impact of input length**  
1. Clearly, on the generalization set, the model is garbage. However, even otherwise, the model seems to make more significant errors on very large numbers as it tends to get the order of magnitude wrong, or randomly gets one of the intermediate digits wrong.
2. Even for very small numbers, it tends to get the order of magnitude wrong.

**Impact of carry/borrow operations**  
A lot of errors occur in cases which require multiple carry/borrow operations.
```
Example 13719:
  Input: -977447889+977306144
  Target: -141745
  Prediction: -1067555
```

**Position dependent errors**  
Most errors seem to occur in the first few places (leftmost), suggesting that there might be an issue in carry/borrow propagation from left to right.

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

While it still struggled with full sequence accuracy on the generalization set, it showed slight improvements in
character-level accuracy.

This suggests deeper networks can better learn the patterns within the training distribution but still face challenges
with proper generalization.

### General findings

All my changes improved accuracy for in-distribution sequence lengths, though this is not what we want. We still want a good, generalizable model.

In general, deeper and wider networks provided better in-distribution performance, but this was at the cost of training
time and potentially increased overfitting as well.

The deeper model (more layers) showed better performance as compared to the wider network (higher dimensional layers).

No configuration has dramatically improved generalization performance. This suggests that hyperparameter tuning alone
may not be sufficient for systematic generalization in arithmetic tasks.

---

## Discussion

After all this analysis, it's clear that our transformer model hasn't really "learned" to do arithmetic. What it's done instead is essentially memorize patterns it saw during training. The huge performance gap between the test set (99.60% accuracy for exact match) and generalization set (5.58% accuracy for exact match) makes this extremely obvious.

The model is great at spitting out answers for number lengths it's seen before, but completely falls apart when faced with longer numbers.Then again, are we supposed to be surprised? I'd say not at all, since transformers are pattern-matching machines, and they're just doing what they do best: finding statistical regularities in the training data.

The ablation studies further confirm this. Making the network deeper or wider improved in-distribution performance, but did basically nothing for generalization. The deeper model (more layers) performed better than the wider one, but neither configuration was better than the model that I started out with, in the generalization problem. Perhaps no amount of hyperparameter tuning can fix the most fundamental issue here: the model is memorizing, not learning.

When we compare this to how humans do arithmetic, the differences are stark. We learn a generalizable algorithm that works regardless of number length. Once we understand carrying and borrowing, we can add or subtract numbers of any length (though we are kinda slow at processing larger numbers). We also process digits sequentially (usually right-to-left), explicitly tracking carries as we go.

The transformer meanwhile, seems as though it attempts to process the entire expression at once, looking for patterns it recognizes from training. It has no explicit procedure and no concept of place value that generalizes beyond what it's seen. Its attention mechanism theoretically gives it access to all input tokens simultaneously, but it still can't effectively handle the long-range dependencies needed for multi-digit arithmetic.

Honestly though, why even are we using transformers for this task? Just use the `add` and `sub` assembly instructions instead, which are extremely effective at this task. Why introduce neural nets into an already deterministically solvable task?

---