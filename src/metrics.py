"""
Metrics computation for NIDS models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Union


def compute_metrics(predictions: Union[torch.Tensor, np.ndarray],
                   labels: Union[torch.Tensor, np.ndarray]) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: Predicted labels (torch.Tensor or np.ndarray)
        labels: True labels (torch.Tensor or np.ndarray)

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - macro_f1: Macro-averaged F1 score (treats all classes equally)
            - weighted_f1: Weighted F1 score (weighted by class frequency)
            - per_class_f1: F1 score for each class
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - confusion_matrix: Confusion matrix
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)

    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(labels, predictions)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_f1': per_class_f1,
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'confusion_matrix': conf_matrix
    }


def compute_per_class_metrics(predictions: Union[torch.Tensor, np.ndarray],
                              labels: Union[torch.Tensor, np.ndarray],
                              num_classes: int = 10) -> Dict:
    """
    Compute detailed per-class metrics.

    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes

    Returns:
        Dictionary with per-class precision, recall, and F1
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute per-class metrics
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    f1 = f1_score(labels, predictions, average=None, zero_division=0)

    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = {int(cls): int(count) for cls, count in zip(unique, counts)}

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_counts': class_counts
    }


def print_classification_report(predictions: Union[torch.Tensor, np.ndarray],
                                labels: Union[torch.Tensor, np.ndarray],
                                class_names: Dict[int, str] = None):
    """
    Print detailed classification report.

    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Optional mapping of class indices to names
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get class names
    if class_names is None:
        class_names = {i: f"Class {i}" for i in range(10)}

    target_names = [class_names[i] for i in sorted(class_names.keys())]

    # Print report
    print("\nClassification Report:")
    print("=" * 70)
    print(classification_report(
        labels,
        predictions,
        target_names=target_names,
        zero_division=0
    ))


def print_confusion_matrix(conf_matrix: np.ndarray,
                          class_names: Dict[int, str] = None):
    """
    Print formatted confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        class_names: Optional mapping of class indices to names
    """
    if class_names is None:
        class_names = {i: f"Class {i}" for i in range(conf_matrix.shape[0])}

    print("\nConfusion Matrix:")
    print("=" * 70)

    # Header
    print(f"{'True/Pred':<15}", end="")
    for i in range(conf_matrix.shape[1]):
        print(f"{class_names[i]:<12}", end="")
    print()
    print("-" * 70)

    # Rows
    for i in range(conf_matrix.shape[0]):
        print(f"{class_names[i]:<15}", end="")
        for j in range(conf_matrix.shape[1]):
            print(f"{conf_matrix[i, j]:<12}", end="")
        print()
    print("=" * 70)


class MetricsTracker:
    """Track metrics across epochs."""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_macro_f1': [],
            'val_macro_f1': [],
        }

    def update(self, train_metrics: Dict, val_metrics: Dict):
        """Update history with new metrics."""
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['train_acc'].append(train_metrics.get('accuracy', 0))
        self.history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.history['train_macro_f1'].append(train_metrics.get('macro_f1', 0))
        self.history['val_macro_f1'].append(val_metrics.get('macro_f1', 0))

    def get_best_epoch(self, metric: str = 'val_macro_f1') -> int:
        """Get epoch with best metric value."""
        if metric not in self.history:
            raise ValueError(f"Unknown metric: {metric}")

        values = self.history[metric]
        return int(np.argmax(values))

    def get_best_value(self, metric: str = 'val_macro_f1') -> float:
        """Get best value for a metric."""
        if metric not in self.history:
            raise ValueError(f"Unknown metric: {metric}")

        values = self.history[metric]
        return float(np.max(values))
