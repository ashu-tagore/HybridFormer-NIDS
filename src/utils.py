"""
Training utilities for NIDS project.

This module provides helper functions for:
- Checkpoint saving and loading
- Early stopping
- Model state management
- Device management
- Random seed setting
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_state: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        metrics: Dictionary of metrics at this checkpoint
        path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        extra_state: Optional extra state to save

    Example:
        >>> save_checkpoint(
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'val_f1': 0.85},
        ...     path='saved_models/checkpoint.pth'
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if extra_state is not None:
        checkpoint['extra_state'] = extra_state

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map checkpoint to

    Returns:
        Dictionary containing checkpoint information

    Example:
        >>> checkpoint_info = load_checkpoint(
        ...     path='saved_models/checkpoint.pth',
        ...     model=my_model,
        ...     optimizer=optimizer
        ... )
        >>> print(f"Loaded from epoch {checkpoint_info['epoch']}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model state loaded from {path}")

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state loaded")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'extra_state': checkpoint.get('extra_state', {})
    }


def early_stopping_check(
    val_losses: List[float],
    patience: int = 10,
    min_delta: float = 0.0
) -> bool:
    """
    Check if training should stop based on validation loss.

    Args:
        val_losses: List of validation losses (most recent last)
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement

    Returns:
        True if training should stop, False otherwise

    Example:
        >>> losses = [0.5, 0.4, 0.38, 0.37, 0.37, 0.37, 0.37]
        >>> should_stop = early_stopping_check(losses, patience=3)
        >>> print(should_stop)
        True
    """
    if len(val_losses) < patience + 1:
        return False

    # Get best loss and current loss
    best_loss = min(val_losses[:-patience])
    current_losses = val_losses[-patience:]

    # Check if any recent loss is better than best_loss - min_delta
    for loss in current_losses:
        if loss < best_loss - min_delta:
            return False

    # No improvement in patience epochs
    logger.info(f"Early stopping triggered - no improvement for {patience} epochs")
    return True


class EarlyStopping:
    """
    Early stopping handler with more features.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/f1

    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_one_epoch()
        ...     if early_stop(val_loss):
        ...         print(f"Stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        # Initialize best score
        if self.best_score is None:
            self.best_score = score
            return False

        # Check if score improved
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get available device (CUDA or CPU).

    Args:
        prefer_cuda: Prefer CUDA if available

    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")

    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary.

    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("="*60)

    param_counts = count_parameters(model)
    print(f"\nParameter Summary:")
    print(f"  Total parameters:      {param_counts['total']:,}")
    print(f"  Trainable parameters:  {param_counts['trainable']:,}")
    print(f"  Non-trainable params:  {param_counts['non_trainable']:,}")
    print("="*60 + "\n")
