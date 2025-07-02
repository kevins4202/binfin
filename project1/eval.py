"""
Evaluation metrics for deepfake audio detection.

This module provides functions to compute various evaluation metrics including:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate) 
- EER (Equal Error Rate)
- AUC, F1, Balanced Accuracy

Author: Assistant
"""

import numpy as np
import torch
import os
from typing import Tuple, Dict, Optional, Union, List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from tqdm import tqdm


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Detection Error Tradeoff (DET) curve.
    
    Args:
        target_scores: Scores for genuine/real samples
        nontarget_scores: Scores for deepfake/fake samples
        
    Returns:
        frr: False rejection rates
        far: False acceptance rates  
        thresholds: Corresponding thresholds
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and corresponding threshold.
    
    Args:
        target_scores: Scores for genuine/real samples
        nontarget_scores: Scores for deepfake/fake samples
        
    Returns:
        eer: Equal error rate
        threshold: Threshold at EER
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_far_frr(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) at a given threshold.
    
    Args:
        labels: Binary labels (1 for genuine/real, 0 for deepfake/fake)
        scores: Prediction scores (higher values indicate more likely to be genuine)
        threshold: Decision threshold
        
    Returns:
        far: False acceptance rate
        frr: False rejection rate
    """
    # Convert to binary predictions
    predictions = (scores > threshold).astype(int)
    
    # Compute confusion matrix elements
    tp = np.sum((labels == 1) & (predictions == 1))  # True positives
    tn = np.sum((labels == 0) & (predictions == 0))  # True negatives
    fp = np.sum((labels == 0) & (predictions == 1))  # False positives
    fn = np.sum((labels == 1) & (predictions == 0))  # False negatives
    
    # Compute rates
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Rejection Rate
    
    return far, frr


def extract_scores_from_model_output(model_output: torch.Tensor) -> np.ndarray:
    """
    Extract scores from model output.
    
    Args:
        model_output: Raw model output
        
    Returns:
        scores: Extracted scores as numpy array
    """
    # Model typically outputs logits for binary classification
    # Apply softmax to get probabilities and take the positive class probability
    if model_output.dim() > 1 and model_output.shape[1] > 1:
        # Multi-class output, apply softmax and take positive class
        probs = torch.softmax(model_output, dim=1)
        scores = probs[:, 1]  # Positive class probability (real/genuine)
    else:
        # Single output, apply sigmoid for binary classification
        scores = torch.sigmoid(model_output.squeeze())
    
    return scores.cpu().numpy().ravel()


def evaluate_model_batch(model: torch.nn.Module, 
                        dataloader: torch.utils.data.DataLoader,
                        device: str = "cuda",
                        augmentations: Optional[callable] = None,
                        augmentations_on_cpu: Optional[callable] = None,
                        manipulation_on_real: bool = True,
                        cut_length: Optional[int] = None,
                        save_scores: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a model on a dataset using batch processing.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        augmentations: Augmentation function to apply on GPU
        augmentations_on_cpu: Augmentation function to apply on CPU
        manipulation_on_real: Whether to apply augmentations to real samples
        cut_length: Length to cut/pad audio to (if None, no processing)
        save_scores: Path to save scores (if None, scores not saved)
        
    Returns:
        all_labels: All labels from the dataset
        all_scores: All scores from the model
    """
    model.eval()
    all_labels = []
    all_scores = []
    
    # Prepare score file if saving
    if save_scores:
        with open(save_scores, 'w') as f:
            pass  # Create/clear the file
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Handle different data formats
            if len(batch_data) == 2:
                audio_input, labels = batch_data
                spks = None
            elif len(batch_data) == 3:
                audio_input, spks, labels = batch_data
            else:
                raise ValueError(f"Unexpected batch data format with {len(batch_data)} elements")
            
            # Process audio input
            if audio_input.dim() > 2:
                audio_input = audio_input.squeeze(1)  # Remove extra dimension
            
            # Apply CPU augmentations if specified
            if augmentations_on_cpu is not None:
                audio_input = augmentations_on_cpu(audio_input)
            
            # Move to device
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            
            # Apply GPU augmentations if specified
            if augmentations is not None:
                if not manipulation_on_real:
                    # Only apply to fake samples
                    audio_length = audio_input.shape[-1]
                    fake_mask = labels == 0
                    if fake_mask.any():
                        audio_input[fake_mask] = pad_or_clip_batch(
                            augmentations(audio_input[fake_mask]), 
                            audio_length, 
                            random_clip=False
                        )
                else:
                    # Apply to all samples
                    audio_input = augmentations(audio_input)
            
            # Handle audio length
            if cut_length is not None:
                if audio_input.shape[-1] < cut_length:
                    # Repeat audio to reach desired length
                    repeats = int(cut_length / audio_input.shape[-1]) + 1
                    audio_input = audio_input.repeat(1, repeats)[:, :cut_length]
                elif audio_input.shape[-1] > cut_length:
                    # Truncate audio
                    audio_input = audio_input[:, :cut_length]
            
            # Forward pass through model
            batch_output = model(audio_input)
            
            # Extract scores using model-specific extraction
            batch_scores = extract_scores_from_model_output(batch_output)
            
            # Convert labels to numpy
            batch_labels = labels.cpu().numpy()
            
            # Store results
            all_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
            
            # Save scores if requested
            if save_scores:
                with open(save_scores, 'a') as f:
                    for label, score in zip(batch_labels, batch_scores):
                        label_str = 'real' if label == 1 else 'fake'
                        f.write(f'- - {label_str} {score}\n')
    
    return np.array(all_labels), np.array(all_scores)


def pad_or_clip_batch(audio_batch: torch.Tensor, target_length: int, random_clip: bool = False) -> torch.Tensor:
    """
    Pad or clip a batch of audio to a target length.
    
    Args:
        audio_batch: Batch of audio tensors
        target_length: Target length to pad/clip to
        random_clip: Whether to clip randomly or from start
        
    Returns:
        Processed audio batch
    """
    batch_size, audio_length = audio_batch.shape
    
    if audio_length == target_length:
        return audio_batch
    elif audio_length < target_length:
        # Pad with zeros
        padding = torch.zeros(batch_size, target_length - audio_length, device=audio_batch.device)
        return torch.cat([audio_batch, padding], dim=1)
    else:
        # Clip
        if random_clip:
            start_indices = torch.randint(0, audio_length - target_length + 1, (batch_size,))
            return torch.stack([audio_batch[i, start:start + target_length] 
                              for i, start in enumerate(start_indices)])
        else:
            return audio_batch[:, :target_length]


def load_scores_from_file(score_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load scores and labels from a saved score file.
    
    Args:
        score_file_path: Path to the score file
        
    Returns:
        labels: Binary labels (1 for real, 0 for fake)
        scores: Prediction scores
    """
    if not os.path.exists(score_file_path):
        raise FileNotFoundError(f"Score file not found: {score_file_path}")
    
    data = np.genfromtxt(score_file_path, dtype=str)
    labels_str = data[:, 2]  # Assuming format: "- - label score"
    scores = data[:, 3].astype(float)
    
    # Convert string labels to binary
    labels = np.where(labels_str == 'real', 1, 0)
    
    return labels, scores


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray, 
                       threshold: Optional[float] = None,
                       print_results: bool = True) -> Dict[str, float]:
    """
    Compute all evaluation metrics for deepfake detection.
    
    Args:
        labels: Binary labels (1 for genuine/real, 0 for deepfake/fake)
        scores: Prediction scores (higher values indicate more likely to be genuine)
        threshold: Decision threshold (if None, EER threshold will be used)
        print_results: Whether to print results
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Input validation
    labels = np.array(labels)
    scores = np.array(scores)
    
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same length")
    
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be binary (0 or 1)")
    
    # Separate scores by class
    real_scores = scores[labels == 1]
    fake_scores = scores[labels == 0]
    
    if len(real_scores) == 0 or len(fake_scores) == 0:
        raise ValueError("Both classes must have at least one sample")
    
    # Compute EER
    eer, eer_threshold = compute_eer(real_scores, fake_scores)
    
    # Use EER threshold if no threshold provided
    if threshold is None:
        threshold = eer_threshold
    
    # Compute FAR and FRR at the given threshold
    far, frr = compute_far_frr(labels, scores, threshold)
    
    # Compute additional metrics
    auc = roc_auc_score(labels, scores)
    predictions = (scores > threshold).astype(int)
    f1 = f1_score(labels, predictions)
    acc = balanced_accuracy_score(labels, predictions)
    
    # Prepare results
    results = {
        'EER': eer,
        'EER_threshold': eer_threshold,
        'FAR': far,
        'FRR': frr,
        'threshold': threshold,
        'AUC': auc,
        'F1_score': f1,
        'balanced_accuracy': acc
    }
    
    # Print results if requested
    if print_results:
        print(f"Evaluation Results:")
        print(f"  EER: {eer:.5f}")
        print(f"  FAR: {far:.5f}")
        print(f"  FRR: {frr:.5f}")
        print(f"  AUC: {auc:.5f}")
        print(f"  F1 Score: {f1:.5f}")
        print(f"  Balanced Accuracy: {acc:.5f}")
        print(f"  Threshold: {threshold:.5f}")
    
    return results


def plot_score_distributions(labels: np.ndarray, scores: np.ndarray, 
                           threshold: Optional[float] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot score distributions for real and fake samples.
    
    Args:
        labels: Binary labels (1 for real, 0 for fake)
        scores: Prediction scores
        threshold: Decision threshold to mark on plot
        save_path: Path to save the plot (if None, plot will be displayed)
    """
    real_scores = scores[labels == 1]
    fake_scores = scores[labels == 0]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(real_scores, bins=50, alpha=0.7, density=True, 
             label='Real', color='blue', histtype='step', linewidth=2)
    plt.hist(fake_scores, bins=50, alpha=0.7, density=True, 
             label='Fake/Deepfake', color='red', histtype='step', linewidth=2)
    
    # Mark threshold if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})', linewidth=2)
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_det_curve(labels: np.ndarray, scores: np.ndarray, 
                  save_path: Optional[str] = None) -> None:
    """
    Plot Detection Error Tradeoff (DET) curve.
    
    Args:
        labels: Binary labels (1 for real, 0 for fake)
        scores: Prediction scores
        save_path: Path to save the plot (if None, plot will be displayed)
    """
    real_scores = scores[labels == 1]
    fake_scores = scores[labels == 0]
    
    frr, far, thresholds = compute_det_curve(real_scores, fake_scores)
    eer, eer_threshold = compute_eer(real_scores, fake_scores)
    
    plt.figure(figsize=(8, 8))
    
    # Plot DET curve
    plt.plot(far, frr, 'b-', linewidth=2, label='DET Curve')
    
    # Mark EER point
    eer_far, eer_frr = compute_far_frr(labels, scores, eer_threshold)
    plt.plot(eer_far, eer_frr, 'ro', markersize=8, label=f'EER ({eer:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def evaluate_from_file(score_file_path: str, 
                      threshold: Optional[float] = None,
                      plot_results: bool = True,
                      save_plots: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate results from a saved score file.
    
    Args:
        score_file_path: Path to the score file
        threshold: Decision threshold (if None, EER threshold will be used)
        plot_results: Whether to generate plots
        save_plots: Directory to save plots (if None, plots will be displayed)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load scores and labels
    labels, scores = load_scores_from_file(score_file_path)
    
    # Compute metrics
    results = compute_all_metrics(labels, scores, threshold)
    
    # Generate plots if requested
    if plot_results:
        if save_plots:
            os.makedirs(save_plots, exist_ok=True)
            plot_score_distributions(labels, scores, results['threshold'], 
                                   os.path.join(save_plots, 'score_distribution.png'))
            plot_det_curve(labels, scores, 
                          os.path.join(save_plots, 'det_curve.png'))
        else:
            plot_score_distributions(labels, scores, results['threshold'])
            plot_det_curve(labels, scores)
    
    return results


# Example usage and testing function
def example_usage():
    """Example usage of the evaluation functions."""
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_real = 1000
    n_fake = 1000
    
    # Generate scores (higher scores for real, lower for fake)
    real_scores = np.random.normal(0.7, 0.2, n_real)
    fake_scores = np.random.normal(0.3, 0.2, n_fake)
    
    # Create labels and scores arrays
    labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
    scores = np.concatenate([real_scores, fake_scores])
    
    print("Example Evaluation:")
    print("=" * 50)
    
    # Compute all metrics
    results = compute_all_metrics(labels, scores)
    
    # Plot distributions
    plot_score_distributions(labels, scores, threshold=results['threshold'])
    
    # Plot DET curve
    plot_det_curve(labels, scores)
    
    return results


if __name__ == "__main__":
    # Run example if script is executed directly
    example_usage()
