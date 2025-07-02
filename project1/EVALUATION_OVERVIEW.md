# BEATs Deepfake Audio Detection Evaluation Overview

## Overview

This document explains how evaluation metrics are calculated for deepfake audio detection using the BEATs model, with special focus on batch processing and score computation.

## Core Metrics

### 1. EER (Equal Error Rate)
- **Definition**: The point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR)
- **Calculation**: 
  1. Sort all scores from both classes
  2. For each threshold, compute FAR and FRR
  3. Find the threshold where |FAR - FRR| is minimized
  4. EER = (FAR + FRR) / 2 at this threshold

### 2. FAR (False Acceptance Rate)
- **Definition**: Rate at which fake/deepfake samples are incorrectly classified as real
- **Formula**: FAR = FP / (FP + TN)
- **Where**: FP = False Positives, TN = True Negatives

### 3. FRR (False Rejection Rate)
- **Definition**: Rate at which real samples are incorrectly classified as fake
- **Formula**: FRR = FN / (FN + TP)
- **Where**: FN = False Negatives, TP = True Positives

### 4. Additional Metrics
- **AUC**: Area Under the ROC Curve
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: (Sensitivity + Specificity) / 2

## Batch Processing Workflow

### 1. BEATs Model Evaluation with Batches

```python
def evaluate_beats_model(model, dataloader, device="cuda", ...):
    all_labels = []
    all_scores = []
    
    for batch_data in dataloader:
        # Process batch
        audio_input, labels = batch_data
        
        # Forward pass through BEATs
        batch_output = model(audio_input)
        
        # Extract scores using BEATs-specific extraction
        batch_scores = extract_scores_from_beats_output(batch_output)
        
        # Store results
        all_labels.extend(labels)
        all_scores.extend(batch_scores)
    
    return np.array(all_labels), np.array(all_scores)
```

### 2. BEATs Score Extraction

BEATs model outputs are processed as follows:

#### Multi-class Output (2+ classes)
```python
# Output: [fake_score, real_score, ...] per sample
probs = torch.softmax(model_output, dim=1)
scores = probs[:, 1]  # Take positive class probability (real/genuine)
```

#### Binary Output (single logit)
```python
# Output: single logit per sample
scores = torch.sigmoid(model_output.squeeze())  # Convert to probability
```

### 3. Batch Processing Considerations

#### Audio Length Handling
```python
if cut_length is not None:
    if audio_input.shape[-1] < cut_length:
        # Pad by repeating audio
        repeats = int(cut_length / audio_input.shape[-1]) + 1
        audio_input = audio_input.repeat(1, repeats)[:, :cut_length]
    elif audio_input.shape[-1] > cut_length:
        # Truncate audio
        audio_input = audio_input[:, :cut_length]
```

#### Augmentation Application
```python
if augmentations is not None:
    if not manipulation_on_real:
        # Only apply to fake samples
        fake_mask = labels == 0
        if fake_mask.any():
            audio_input[fake_mask] = augmentations(audio_input[fake_mask])
    else:
        # Apply to all samples
        audio_input = augmentations(audio_input)
```

## Score Calculation Process

### 1. Individual Sample Scoring
For each audio sample in a batch:
1. **Input**: Raw audio waveform (typically 1D tensor)
2. **Model Forward Pass**: Audio → Model → Raw output
3. **Score Extraction**: Extract positive class probability/score
4. **Output**: Single float value (higher = more likely real)

### 2. Batch Aggregation
```python
# Collect all scores and labels
all_labels = []
all_scores = []

for batch in dataloader:
    batch_labels, batch_scores = process_batch(batch)
    all_labels.extend(batch_labels)
    all_scores.extend(batch_scores)

# Convert to numpy arrays
labels = np.array(all_labels)  # Shape: (N,)
scores = np.array(all_scores)  # Shape: (N,)
```

### 3. Metric Computation
```python
# Separate scores by class
real_scores = scores[labels == 1]    # Shape: (N_real,)
fake_scores = scores[labels == 0]    # Shape: (N_fake,)

# Compute EER
eer, eer_threshold = compute_eer(real_scores, fake_scores)

# Compute FAR/FRR at threshold
far, frr = compute_far_frr(labels, scores, eer_threshold)
```

## Key Implementation Details

### 1. Memory Efficiency
- Process batches instead of loading all data at once
- Use `torch.no_grad()` during evaluation
- Move tensors to CPU before converting to numpy

### 2. Device Handling
```python
# Move to GPU for computation
audio_input = audio_input.to(device)
labels = labels.to(device)

# Move back to CPU for numpy conversion
scores = scores.cpu().numpy()
```

### 3. Error Handling
- Validate input shapes and types
- Handle edge cases (empty batches, single samples)
- Check for NaN/Inf values in scores

### 4. Score Normalization
- Most models output logits or probabilities
- Higher scores typically indicate higher confidence in real class
- Scores are used directly without additional normalization

## Usage Examples

### Basic Evaluation
```python
from eval import evaluate_beats_model, compute_all_metrics

# Evaluate BEATs model
labels, scores = evaluate_beats_model(model, dataloader)

# Compute metrics
results = compute_all_metrics(labels, scores)
print(f"EER: {results['EER']:.4f}")
```

### Evaluation with Custom Threshold
```python
# Use custom threshold instead of EER threshold
results = compute_all_metrics(labels, scores, threshold=0.5)
```

### Evaluation from Saved Scores
```python
from eval import evaluate_from_file

# Load and evaluate from file
results = evaluate_from_file("scores.txt", plot_results=True)
```

## File Format for Saved Scores

Scores are saved in a simple text format:
```
- - real 0.85
- - fake 0.23
- - real 0.92
- - fake 0.15
...
```

Where:
- `real` = genuine/real audio (label = 1)
- `fake` = deepfake/fake audio (label = 0)
- Number = model prediction score

## Performance Considerations

### 1. Batch Size
- Larger batches = faster processing but more memory usage
- Typical batch sizes: 32-128 for evaluation
- Adjust based on GPU memory and model size

### 2. Data Loading
- Use multiple workers for data loading
- Pin memory for faster GPU transfer
- Prefetch data to avoid GPU waiting

### 3. Model Optimization
- Use `model.eval()` for evaluation mode
- Disable gradient computation with `torch.no_grad()`
- Consider model quantization for faster inference

## Common Issues and Solutions

### 1. Memory Issues
- Reduce batch size
- Use gradient checkpointing
- Process in smaller chunks

### 2. Score Range Issues
- Ensure scores are in expected range (typically [0,1] or [-∞,∞])
- Check for NaN/Inf values
- Verify model output format

### 3. Label Mismatch
- Ensure labels are binary (0 or 1)
- Verify label encoding (0=fake, 1=real)
- Check for class imbalance

This evaluation framework provides a comprehensive and efficient way to assess deepfake detection models with proper batch processing support.