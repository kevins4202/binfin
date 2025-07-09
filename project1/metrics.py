from sklearn.metrics import accuracy_score
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def compute_eer(scores, labels):
    """Compute EER and its threshold using numpy + sklearn"""
    # Check for single class or constant scores
    if np.all(scores == scores[0]) or len(np.unique(labels)) < 2:
        print("Warning: ROC curve cannot be computed (constant scores or single class).")
        return np.nan, np.nan
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, thresholds[eer_idx]

def get_eval_metrics(logits: torch.Tensor, 
                     labels: torch.Tensor, 
                     plot_figure: bool = False, 
                     threshold: float = None, 
                     print_result: bool = False):
    """
    logits: torch.FloatTensor, model logits (higher = more likely deepfake), shape [N]
    labels: torch.LongTensor, ground truth labels (0 = genuine, 1 = deepfake), shape [N]
    """

    # Ensure CPU and convert to numpy for sklearn
    logits = logits.numpy()
    true_labels = labels.numpy()
    
    # Apply sigmoid to get probabilities for better interpretability
    probabilities = torch.sigmoid(torch.from_numpy(logits))
    
    # EER computation
    # EER needs labels: 1 = genuine, 0 = spoof — so we invert for EER only
    eer_labels = 1 - true_labels
    eer_scores = -logits  # reverse score: higher = more genuine
    eer, threshold = compute_eer(eer_scores, eer_labels)

    threshold = threshold if threshold is not None else 0.5

    # Predicted classes using probability threshold (0.5 is natural threshold for sigmoid)
    y_pred = (probabilities > threshold).float()

    # AUC, F1, balanced accuracy
    auc = roc_auc_score(true_labels, logits)
    f1 = f1_score(true_labels, y_pred)
    acc = accuracy_score(true_labels, y_pred)

    # FAR: False Acceptance Rate (genuine predicted as deepfake)
    # FRR: False Rejection Rate (deepfake predicted as genuine)
    FAR = np.sum((true_labels == 0) & (y_pred == 1)) / np.sum(true_labels == 0)
    FRR = np.sum((true_labels == 1) & (y_pred == 0)) / np.sum(true_labels == 1)

    if print_result:
        print(f"EER: {eer:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}, "
              f"Threshold: {threshold:.4f}, FAR: {FAR:.4f}, FRR: {FRR:.4f}")

    if plot_figure:
        genuine_probs = probabilities[true_labels == 0]
        deepfake_probs = probabilities[true_labels == 1]
        sns.histplot(genuine_probs, label='Genuine', stat="density", element="step", fill=False, bins='auto')
        sns.histplot(deepfake_probs, label='Deepfake', stat="density", element="step", fill=False, bins='auto')
        plt.legend()
        plt.xlabel('Prediction probability')
        plt.title('Prediction probability histogram')
        plt.show()

    return eer, auc, f1, acc, threshold, FAR, FRR
