"""
Evaluation metrics for NIDS project.

This module provides comprehensive metrics calculation including
accuracy, precision, recall, F1-score, and confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for classification.

    Computes per-class and macro-averaged metrics including:
    - Accuracy
    - Precision, Recall, F1-score (per class and macro)
    - Confusion matrix

    Args:
        num_classes: Number of classes
        class_names: Optional dictionary mapping class_id -> class_name

    Example:
        >>> calc = MetricsCalculator(num_classes=10)
        >>> metrics = calc.compute_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.856
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_names: Optional[Dict[int, str]] = None
    ):
        self.num_classes = num_classes

        # Default class names if not provided
        if class_names is None:
            self.class_names = {
                0: "Benign",
                1: "Analysis",
                2: "Backdoor",
                3: "DoS",
                4: "Exploits",
                5: "Fuzzers",
                6: "Generic",
                7: "Reconnaissance",
                8: "Shellcode",
                9: "Worms"
            }
        else:
            self.class_names = class_names

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            y_true: True labels (shape: [N,])
            y_pred: Predicted labels (shape: [N,])
            average: Averaging method ('macro', 'weighted', 'micro')

        Returns:
            Dictionary containing all metrics
        """
        # Ensure numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        # Flatten if needed (remove extra dimensions)
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, recall, F1 (macro-averaged)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average=None,
                zero_division=0,
                labels=list(range(self.num_classes))
            )

        metrics = {
            'accuracy': float(accuracy),
            f'precision_{average}': float(precision),
            f'recall_{average}': float(recall),
            f'f1_{average}': float(f1),
        }

        # Add per-class metrics
        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f"Class_{i}")
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
            metrics[f'support_{class_name}'] = int(support_per_class[i])

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix (shape: [num_classes, num_classes])
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            normalize=normalize
        )

        return cm

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report as string.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Formatted classification report
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            target_names=[self.class_names[i] for i in range(self.num_classes)],
            zero_division=0
        )

        return report

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary mapping class_name -> {precision, recall, f1, support}
        """
        y_true = self._to_numpy(y_true).flatten()
        y_pred = self._to_numpy(y_pred).flatten()

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
            labels=list(range(self.num_classes))
        )

        per_class = {}
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        return per_class

    def _to_numpy(self, tensor_or_array) -> np.ndarray:
        """Convert PyTorch tensor or numpy array to numpy array."""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.cpu().detach().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        else:
            return np.array(tensor_or_array)

    def print_metrics_summary(self, metrics: Dict[str, float]):
        """
        Pretty print metrics summary.

        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)

        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")

        for avg_type in ['macro', 'weighted', 'micro']:
            if f'precision_{avg_type}' in metrics:
                print(f"\n{avg_type.capitalize()}-averaged:")
                print(f"  Precision:          {metrics[f'precision_{avg_type}']:.4f}")
                print(f"  Recall:             {metrics[f'recall_{avg_type}']:.4f}")
                print(f"  F1-score:           {metrics[f'f1_{avg_type}']:.4f}")

        # Per-class metrics
        print(f"\nPer-class F1-scores:")
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            f1_key = f'f1_{class_name}'
            support_key = f'support_{class_name}'
            if f1_key in metrics:
                f1 = metrics[f1_key]
                support = metrics.get(support_key, 0)
                print(f"  {class_name:15s}: {f1:.4f} (support: {support:>6})")

        print("="*60 + "\n")


def compute_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metrics_calc: Optional[MetricsCalculator] = None
) -> Dict[str, float]:
    """
    Compute metrics from model logits.

    Args:
        logits: Model output logits (shape: [N, num_classes])
        labels: True labels (shape: [N,] or [N, 1])
        metrics_calc: MetricsCalculator instance (creates default if None)

    Returns:
        Dictionary of computed metrics
    """
    # Get predictions from logits
    predictions = torch.argmax(logits, dim=1)

    # Flatten labels if needed
    if len(labels.shape) > 1:
        labels = labels.squeeze()

    # Convert to numpy
    y_pred = predictions.cpu().detach().numpy()
    y_true = labels.cpu().detach().numpy()

    # Compute metrics
    if metrics_calc is None:
        metrics_calc = MetricsCalculator()

    return metrics_calc.compute_metrics(y_true, y_pred)
